import casadi as ca
import numpy as np
from datetime import datetime, timedelta

class MPCTeacher:
    def __init__(self, R=0.0, C=0.0, Q_gain=0.0, A_solar=0.0, dt=900, target_mode='middle'):
        """
        初始化 MPC 老师
        :param target_mode:
            'middle' -> Peak (冬季) 策略
            'lower'  -> Typical (春季) 策略
        """
        self.R = R
        self.C = C
        self.Q_gain = Q_gain
        self.A_solar = A_solar
        self.dt = dt
        self.target_mode = target_mode
        self.c_hp = [-6000.0, 12000.0, 493.7]
        # 预测时域
        self.N = 24
        self.T_CONV = 273.15  # 定义转换常数

        # 权重设置
        self.w_comfort = 100000.0
        self.w_price = 1.0
        self.w_smooth = 10000.0

        # 初始化求解器
        self._setup_solver()
        self._setup_mhe_solver()

    def _to_K(self, t):
        """自动识别摄氏度并转为开尔文"""
        return t + self.T_CONV if t < 100.0 else t

    def _to_C(self, t):
        """自动识别开尔文并转为摄氏度"""
        return t - self.T_CONV if t > 100.0 else t


    def step_disturbance_observer(self, t_meas_deg, t_pred_deg):
        
        t_meas_c = self._to_C(t_meas_deg)
        t_pred_c = self._to_C(t_pred_deg)
        """
        误差比例法自适应扰动观测器
        核心逻辑：根据建筑物理特性(C/dt)计算基准增益，并随误差大小动态缩放因子。
        """
        # 1. 计算当前预测误差
        error = t_meas_c - t_pred_c
        abs_err = abs(error)

        # 2. 计算物理基准增益 (理论上 15 分钟抹平 1℃ 偏差所需的功率)
      
        base_gain = self.C / self.dt

        # 3. 计算自适应因子 factor (建议范围 0.01 ~ 0.15)
        # - 误差极小时 (<0.05℃)：可能是传感器噪声，使用极小因子 (0.01) 保持稳定。
        # - 误差较大时 (>0.5℃)：可能是开窗或强日照，使用较大因子 (0.12) 快速追踪。
        if abs_err < 0.05:
            factor = 0.01
        elif abs_err > 0.5:
            factor = 0.12
        else:
            # 中间区域：线性平滑过渡
            factor = 0.01 + (0.12 - 0.01) * (abs_err - 0.05) / (0.5 - 0.05)

        # 4. 计算最终观测增益
        # obs_gain = 物理基准 * 活跃度因子
        obs_gain = base_gain * factor

        # 5. 更新 Q_gain (单位为 W)
      
        self.Q_gain = self.Q_gain + obs_gain * error

        # 6. 物理边界裁剪，防止由于极端数据导致系统崩溃
        self.Q_gain = np.clip(self.Q_gain, -5000.0, 5000.0)

        # 调试打印，观察 factor 和 Q_gain 的变化
        # print(f"[Observer] Err: {error:.3f}, Factor: {factor:.3f}, New Qg: {self.Q_gain:.1f}")

    def _setup_solver(self):
        """
        配置 MPC 求解器，包括对齐物理逻辑的成本函数和软约束目标。
        """
        # 1. 符号变量定义
        x = ca.MX.sym('x')  # 室内温度 (K)
        u = ca.MX.sym('u')  # 控制信号 [0, 1]

        # 参数矩阵扩展为 6 维: [Tout, Qsol, Price, Tmin, Tmax, T_target]
        p_data = ca.MX.sym('p_data', 7)
        Tout, Qsol, Price, Tmin, Tmax, T_target, Qg_val = (p_data[0], p_data[1], p_data[2],
                                                           p_data[3], p_data[4], p_data[5], p_data[6])

        # 2. 物理模型演化 (f_step)
        # 计算热泵主机产生的热量 (W)
        Q_hp_W = self.c_hp[0] * (u ** 2) + self.c_hp[1] * u + self.c_hp[2]

        # 室内温度状态转移方程
        x_next = x + (self.dt / self.C) * (
                (Tout - x) / self.R + Q_hp_W + Qg_val + self.A_solar * Qsol
        )
        self.f_step = ca.Function('f_step', [x, u, p_data], [x_next])

        # 3. 成本函数定义 (f_cost) - 对齐 ActionLinkWrapper
        # A. 主机电功率 (kW)
        P_hp_kW = Q_hp_W / 1000.0

        # B. 辅机电功率 (kW): 包含风机 (1.0kW) 和 泵 (0.5kW) = 总计 1.5kW
        # 逻辑逻辑：当 u > 0.05 时，辅机运行功率为 max(0.2, u) * 1.5
        # --- 修改前 (存在断崖跳变) ---
        # P_aux_kW = ca.if_else(u > 0.05, ca.fmax(0.2, u) * 1.5, 0.0)

        # --- 修改后 (Sigmoid 平滑切换) ---
        # 定义平滑切换因子：当 u 越过 0.05 时，switch 从 0 平滑过渡到 1
        # 100 是陡峭系数，数值越大越接近硬切换，数值越小越平滑
        switch = 1 / (1 + ca.exp(-100 * (u - 0.05)))

        # 使用平滑因子代替 if_else
        P_aux_kW = (ca.fmax(0.2, u) * 1.5) * switch
        # C. 总消耗电能 (kWh) 与成本计算
        P_total_kW = P_hp_kW + P_aux_kW
        cost_elec = Price * P_total_kW * (self.dt / 3600.0)  # 将功率转化为步长内的电费
        self.f_cost = ca.Function('f_cost', [u, p_data], [cost_elec])

        # 4. 构建优化问题 (NLP)
        U = ca.MX.sym('U', self.N)  # 未来动作序列
        X = ca.MX.sym('X', self.N + 1)  # 未来状态序列
        Eps = ca.MX.sym('Eps', self.N)  # 用于软约束的松弛变量

        P = ca.MX.sym('P', 7, self.N)  # 预测时域内的参数
        X0 = ca.MX.sym('X0')  # 初始温度

        obj = 0
        g = []
        g.append(X[0] - X0)  # 初始条件约束

        for k in range(self.N):
            st = X[k]
            con = U[k]
            eps = Eps[k]
            param = P[:, k]

            # 读取当前时刻的约束边界和目标
            t_min_k, t_max_k, t_target_k = param[3], param[4], param[5]

            # 状态演化约束
            st_next = self.f_step(st, con, param)
            g.append(X[k + 1] - st_next)

            # --- 目标函数累加 ---
            # A. 经济成本 (电费)
            obj += self.w_price * self.f_cost(con, param)

            # B. 舒适度追踪：追求接近 T_target
            obj += self.w_comfort * (st_next - t_target_k) ** 2

            # C. 软约束惩罚：防止硬约束导致求解器无解
            obj += 1e6 * eps ** 2

            # 温度红线约束（软约束形式）
            g.append(st_next - (t_min_k - eps))  # 室内温不低于下限
            g.append((t_max_k + eps) - st_next)  # 室内温不高于上限

            # D. 平滑惩罚：减少执行器频繁大幅跳变
            if k > 0:
                obj += self.w_smooth * (U[k] - U[k - 1]) ** 2

        # NLP 定义与求解器配置
        opt_vars = ca.vertcat(U, X, Eps)
        g_all = ca.vertcat(*g)

        nlp_prob = {'f': obj, 'x': opt_vars, 'g': g_all, 'p': ca.vertcat(X0, ca.vec(P))}
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-4,
            'ipopt.warm_start_init_point': 'no'
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def _setup_mhe_solver(self, n_mhe=96):
        """构建 MHE 求解器，辨识 6 个核心物理参数"""
        # 1. 定义待辨识的变量 (待优化的 x)
        v_R = ca.MX.sym('R')
        v_C = ca.MX.sym('C')
        v_As = ca.MX.sym('A_solar')
        v_c1 = ca.MX.sym('c1')
        v_c2 = ca.MX.sym('c2')
        v_c3 = ca.MX.sym('c3')
        v_Qg = ca.MX.sym('Q_gain')

        vars_mhe = ca.vertcat(v_R, v_C, v_As, v_Qg, v_c1, v_c2, v_c3)

        # 2. 定义观测到的历史数据 (参数 p)
        p_T_meas = ca.MX.sym('T_meas', n_mhe)
        p_T_out = ca.MX.sym('T_out', n_mhe)
        p_Q_sol = ca.MX.sym('Q_sol', n_mhe)
        p_U_past = ca.MX.sym('U_past', n_mhe)

        p_last_params = ca.MX.sym('p_last', 7)
        p_mhe = ca.vertcat(p_T_meas, p_T_out, p_Q_sol, p_U_past, p_last_params)
        # 3. 构造仿真轨迹与目标函数
        obj = 0
        t_sim = p_T_meas[0]  # 以历史第一帧温度为起点进行迭代

        for k in range(n_mhe - 1):
         
            q_hp = v_c1 * (p_U_past[k] ** 2) + v_c2 * p_U_past[k] + v_c3
         
            t_next = t_sim + (self.dt / v_C) * (
                    (p_T_out[k] - t_sim) / v_R + q_hp + v_Qg + v_As * p_Q_sol[k]
            )
           
            obj += (t_next - p_T_meas[k + 1]) ** 2
            t_sim = t_next  # 自回归迭代
       
        obj += 10.0 * ((v_R - p_last_params[0]) / p_last_params[0]) ** 2
        obj += 10.0 * ((v_C - p_last_params[1]) / p_last_params[1]) ** 2
        obj += 1.0 * (v_Qg - p_last_params[3]) ** 2
           
        obj += 1e-3 * ca.sumsqr(ca.vertcat(v_c1, v_c2, v_c3) - p_last_params[4:7])
        # 4. 配置求解器
        nlp = {'x': vars_mhe, 'f': obj, 'p': p_mhe}
        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-6}
        self.mhe_solver = ca.nlpsol('mhe_solver', 'ipopt', nlp, opts)

    def update_model_online(self, t_meas, t_out, q_sol, u_past):
        """执行辨识并更新 MPC 内部模型（含扰动项更新）"""
      
        lbw = [0.0005, 5e6, 0.0, -2000.0, -15000, 10000, 0]
        ubw = [0.05, 5e8, 200.0, 2000.0, -5000, 30000, 1500]
        # 初始猜想值使用当前对象的实时状态
        x0 = [self.R, self.C, self.A_solar, self.Q_gain, self.c_hp[0], self.c_hp[1], self.c_hp[2]]
        p_last = np.array(x0)
        p_in = np.concatenate([t_meas, t_out, q_sol, u_past, p_last])

        try:
            sol = self.mhe_solver(x0=x0, lbx=lbw, ubx=ubw, p=p_in)
            new_params = sol['x'].full().flatten()
            mhe_loss = float(sol['f'])  # 获取目标函数值，即拟合残差
            # 映射回对象属性
            self.R = new_params[0]
            self.C = new_params[1]
            self.A_solar = new_params[2]
            self.Q_gain = new_params[3]  # <--- 观测器更新的核心
            self.c_hp = [new_params[4], new_params[5], new_params[6]]


            q_hp_last = self.c_hp[0] * (u_past[-1] ** 2) + self.c_hp[1] * u_past[-1] + self.c_hp[2]
            self.last_mhe_temp_k = t_meas[-1] + (self.dt / new_params[1]) * (
                    (t_out[-1] - t_meas[-1]) / new_params[0] + q_hp_last + new_params[3] + new_params[2] * q_sol[-1]
            )
            self._setup_solver()
            return True, mhe_loss
        except Exception as e:
            print(f">>> [MHE Error]: {e}")
            return False, 0.0

    def _get_comfort_bounds_at_time(self, time_seconds):
        base_date = datetime(2019, 1, 1)
        current_dt = base_date + timedelta(seconds=time_seconds)
        weekday = current_dt.weekday()
        hour = current_dt.hour + current_dt.minute / 60.0

        # 默认：有人在家 (21-24)
        t_lower, t_upper = 21.0, 24.0

        # 工作日白天 (07:00 - 20:00)：人去上班 (15-30)
        if weekday < 5:
            if 7.0 <= hour < 20.0:
                t_lower, t_upper = 15.0, 30.0

        return t_lower, t_upper

    def get_action(self, state_norm, current_time_seconds, forecast_data=None, external_target_temp=None):
        """
        :param external_target_temp: 外部传入的目标温度 (摄氏度)，如果提供，将覆盖内部逻辑
        """
        # 1. 从 teacher-190 观测中取当前量（不再反归一化）
        # 观测结构: [Tset, Tin, eT, u_prev, eThigh, eTlow] + Tout[LEN] + Solar[LEN] + Price[LEN] + Thigh[LEN] + Tlow[LEN] + time feats
        Tin_C = float(state_norm[1])
        T_zone_K = Tin_C + self.T_CONV

        # 当前时刻的预测外温/太阳（取序列第 0 个）
        LEN = int(self.N)  # MPC 预测步数（通常 12）
        # teacher-190 的 forecast LEN=36；这里取第0个足够用
        Tout_C_current = float(state_norm[6])
        T_out_K_current = Tout_C_current + self.T_CONV

        Q_sol_current = float(state_norm[6 + 36])

        # 2. 构建预测矩阵 (6维: [Tout, Qsol, Price, Tmin, Tmax, T_target])
        p_matrix = np.zeros((7, self.N))

        for k in range(self.N):
            future_time = current_time_seconds + k * self.dt

            # A. 获取红线
            t_low, t_high = self._get_comfort_bounds_at_time(future_time)
            p_matrix[3, k] = t_low + self.T_CONV
            p_matrix[4, k] = t_high + self.T_CONV

            # B. 填充天气
            if forecast_data and 'Tout' in forecast_data and len(forecast_data['Tout']) >= self.N:
                t_out_val = self._to_K(forecast_data['Tout'][k])  # ✅ 自动识别°C/K，强制转K
                p_matrix[0, k] = t_out_val
                p_matrix[1, k] = forecast_data['Qsol'][k]
                p_matrix[2, k] = forecast_data['Price'][k]
            else:
                p_matrix[0, k] = T_out_K_current
                p_matrix[1, k] = Q_sol_current
                p_matrix[2, k] = 0.25

            p_matrix[6, k] = self.Q_gain  # ✅ 放到 if/else 外面，避免漏填

            # C. 【核心逻辑】计算目标温度
            target_temp_c = 0.0

            # === 1. 如果有外部传入的规则温度，优先使用 ===
            if external_target_temp is not None:
                # 不要在 MPC 内部再改目标温度，否则会和 main 的 t_target 不一致
                target_temp_c = np.clip(external_target_temp, t_low, t_high)
            # === 2. 否则使用原来的内部逻辑 (Fallback) ===
            else:
                if self.target_mode == 'middle':
                    target_temp_c = np.clip((t_low + t_high) / 2.0 - 0.5, t_low, t_high)
                elif self.target_mode == 'lower':
                    self.w_upper_violation = self.w_comfort * 2.0
                    if t_low > 15.0:
                        target_temp_c =t_low + 0.2
                    else:
                        target_temp_c =t_low + 0.5

            p_matrix[5, k] = target_temp_c + self.T_CONV
        # 3. 求解
        x0_val = T_zone_K
        param_flat = np.concatenate(([x0_val], p_matrix.flatten('F')))

        lb_u, ub_u = [0.0] * self.N, [1.0] * self.N
        lb_x, ub_x = [270.0] * (self.N + 1), [320.0] * (self.N + 1)
        lb_eps, ub_eps = [0.0] * self.N, [np.inf] * self.N
        lbg, ubg = [0.0], [0.0]
        for _ in range(self.N):
            lbg.extend([0.0, 0.0, 0.0])
            ubg.extend([0.0, np.inf, np.inf])

        try:
            sol = self.solver(
                x0=[0.5] * self.N + [x0_val] * (self.N + 1) + [0.0] * self.N,
                lbx=lb_u + lb_x + lb_eps, ubx=ub_u + ub_x + ub_eps,
                lbg=lbg, ubg=ubg, p=param_flat
            )
            return float(sol['x'].full().flatten()[0])

        except Exception as e:
           
            print(f"!!! [MPC Error] Solver failed at {current_time_seconds}: {e}")

            if self.target_mode == 'middle':
              
                return 0.6
            else:
                
                return 0.1

    # mpc_teacher.py 增加或修改

    def get_one_step_prediction(self, T_zone_K, T_out_K, Q_sol, action, R, C, Q_gain):
        """
        根据离散化的 RC 方程，计算下一时刻的预测温度 (K)
        """

        # Q_hp = c1*u^2 + c2*u + c3
        u = float(action)
        Q_hp = self.c_hp[0] * (u ** 2) + self.c_hp[1] * u + self.c_hp[2]

        # 离散化 RC 方程: dT = (dt/C) * [ (To - Tz)/R + Q_sol*A + Q_gain + Q_hp ]
        dT = (self.dt / C) * (
                (T_out_K - T_zone_K) / R +
                Q_sol * self.A_solar +
                Q_gain +
                Q_hp
        )
        return T_zone_K + dT