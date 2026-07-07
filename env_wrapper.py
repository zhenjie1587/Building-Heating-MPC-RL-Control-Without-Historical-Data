import numpy as np
import gymnasium as gym
from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper
import requests
from collections import deque
from datetime import datetime, timedelta

import math
# BOPTEST服务器地址
BOPTEST_URL = "http://127.0.0.1:8080"

# ================= 配置区域 =================

class ClipObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return np.clip(obs, -1e6, 1e6).astype(np.float32)

def get_grid_signals(time_seconds):
    return 10000.0

# =============================================================

class Teacher190NormalizeWrapper(gym.ObservationWrapper):
    """
    把 teacher-190 从原始物理量纲缩放到 [-1, 1]
    使用 SuperAugmentedObservationWrapper 里定义的 observation_space.low/high
    """
    def __init__(self, env, eps: float = 1e-6):
        super().__init__(env)
        low = np.asarray(env.observation_space.low, dtype=np.float32)
        high = np.asarray(env.observation_space.high, dtype=np.float32)
        scale = high - low
        scale = np.where(scale < eps, 1.0, scale)

        self._low = low
        self._scale = scale

        # 新的 obs space：全部 [-1, 1]
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        x = (x - self._low) / self._scale
        x = x * 2.0 - 1.0
        return np.clip(x, -1.0, 1.0).astype(np.float32)

class SuperAugmentedObservationWrapper(gym.ObservationWrapper):
    """
    输出 teacher-190 维观测（与你老师 PreTrain.py 里的 GetNetX 对齐）：

    X = [Tset, Tin, eT, u_prev, eThigh, eTlow]
        + Tout[LEN] + Solar[LEN] + Price[LEN] + Thigh[LEN] + Tlow[LEN]
        + [Tsin, Tcos, eTsin, eTcos]

    其中 LEN=36（9小时，每步15min），单位全部是：
    - 温度: 摄氏度 (°C)
    - Solar: W/m²
    - Price: 与 BOPTEST forecast 一致
    """
    def __init__(self, env, n_forecast: int = 36, dt: int = 900):
        super().__init__(env)
        self.n_forecast = int(n_forecast)
        self.dt = int(dt)
        self.price_point = "PriceElectricPowerHighlyDynamic"


        # 外部注入变量（主循环每步 update_super_wrapper_vars）
        self.external_vars = {
            "T_target": 22.0,   # °C
            "u_mpc": 0.0,
            "u_rl": 0.0,
            "R": 0.003,
            "C": 6.9e7,
            "Q_gain": 0.0,
        }

        # 上一步动作（对齐 teacher: u = U[k-1]）
        self.prev_u = 0.0

        # 预先算好 sin8/cos8（与你老师 CreateBuffer 保持一致：20步=5小时偏移）
        self._sin8 = math.sin((20 * self.dt / 86400.0) * 2 * math.pi)
        self._cos8 = math.cos((20 * self.dt / 86400.0) * 2 * math.pi)

        # teacher-190 的 obs space（范围用“足够宽”的物理合理区间即可）
        low = np.array(
            [15.0, 15.0, -15.0, 0.0, -15.0, -15.0] +                 # 6
            [-50.0] * self.n_forecast +                                # Tout
            [0.0] * self.n_forecast +                                  # Solar
            [0.0] * self.n_forecast +                                  # Price
            [10.0] * self.n_forecast +                                 # Thigh
            [10.0] * self.n_forecast +                                 # Tlow
            [-1.0, -1.0, -2.0, -2.0]                                   # sin/cos & errors
            + [-50.0, 0.0],                                            #  Tout_max_9h, Solar_max_9h
            dtype=np.float32
        )
        high = np.array(
            [30.0, 30.0, 15.0, 1.0, 15.0, 15.0] +                      # 6
            [50.0] * self.n_forecast +                                  # Tout
            [1500.0] * self.n_forecast +                                # Solar
            [10.0] * self.n_forecast +                                  # Price
            [40.0] * self.n_forecast +                                  # Thigh
            [40.0] * self.n_forecast +                                  # Tlow
            [1.0, 1.0, 2.0, 2.0] +
            [50.0, 1500.0],                                             # Tout_max_9h, Solar_max_9h
            dtype=np.float32
        )
        assert low.shape[0] == 6 + 5 * self.n_forecast + 4+ 2
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.last_forecast_dict = None  # 仍保留给 main/teacher 用

    def update_external_vars(self, updates: dict):
        self.external_vars.update(updates)

    def _get_forecast(self):
        try:
            url = f"{BOPTEST_URL}/forecast/{self.env.unwrapped.testid}"  # 顺便避免 testid 警告
            payload = {
                "point_names": [
                    "TDryBul",
                    "HGloHor",
                    self.price_point,
                    "EmissionsElectricPower",
                ],
                "horizon": self.n_forecast * self.dt,
                "interval": self.dt,
            }
            r = requests.put(url, json=payload, timeout=10)
            if r.status_code != 200:
                print("[forecast] HTTP", r.status_code, "text=", r.text[:200])
                return None

            data = r.json()["payload"]
            tout = data["TDryBul"]
            qsol = data["HGloHor"]
            price = data[self.price_point]
            emissions = data.get("EmissionsElectricPower", [0.0] * len(tout))
            return {"Tout": tout, "Qsol": qsol, "Price": price, "Emissions": emissions}
        except Exception as e:
            print("[forecast] EXCEPTION:", repr(e))
            return None

    def _get_comfort_seq(self, time_seconds: float):
        from main import get_comfort_bounds
        t_low_seq = []
        t_high_seq = []
        for k in range(self.n_forecast):
            t_low, t_high = get_comfort_bounds(time_seconds + k * self.dt)
            t_low_seq.append(float(t_low))
            t_high_seq.append(float(t_high))
        return t_low_seq, t_high_seq

    def observation(self, obs: np.ndarray) -> np.ndarray:
        # obs: 来自 base env 的原始观测（含 time, Tzone(K), Tout(K), Qsol, ...）
        time_seconds = float(obs[0])
        self.prev_u = float(self.external_vars.get("u_rl", self.prev_u))
        # 当前室内温度 Tin (°C)：reaTZon_y 在 make_env 里是第 2 个
        Tin_C = float(obs[1] - 273.15)

        # 当前 setpoint（上一时刻主循环注入的目标温度；在决策时它就是“上一时刻设定”）
        Tset_C = float(self.external_vars.get("T_target", 22.0))

        # 当前误差 eT = Tset - Tin
        eT = Tset_C - Tin_C


        u_prev = float(self.prev_u)

        # forecast
        fc = self._get_forecast()
        if fc is None:
            # fallback：用当前测量值重复填充
            Tout_C_seq = [float(obs[2] - 273.15)] * self.n_forecast
            Solar_seq = [float(obs[3])] * self.n_forecast
            Price_seq = [0.25] * self.n_forecast
            print(f"[forecast] FAILED, using fallback price=0.25, price_point={self.price_point}")


        else:
            self.last_forecast_dict = fc
            Tout_C_seq = [float(x - 273.15) for x in fc["Tout"][: self.n_forecast]]
            Solar_seq = [float(x) for x in fc["Qsol"][: self.n_forecast]]
            Price_seq = [float(x) for x in fc["Price"][: self.n_forecast]]

            # pad
            if len(Tout_C_seq) < self.n_forecast:
                pad = self.n_forecast - len(Tout_C_seq)
                Tout_C_seq += [Tout_C_seq[-1]] * pad
                Solar_seq += [Solar_seq[-1]] * pad
                Price_seq += [Price_seq[-1]] * pad

        # comfort bounds sequence（°C）
        Tlow_seq, Thigh_seq = self._get_comfort_seq(time_seconds)


        # eThigh/eTlow：改成当前时刻（k 时刻用 k 的舒适上下限）
        Thigh_now = Thigh_seq[0]
        Tlow_now = Tlow_seq[0]
        eThigh = Thigh_now - Tin_C
        eTlow = Tin_C - Tlow_now

        # time features
        day_seconds = 24.0 * 3600.0
        norm_time = (time_seconds % day_seconds) / day_seconds
        Tsin = math.sin(2.0 * math.pi * norm_time)
        Tcos = math.cos(2.0 * math.pi * norm_time)
        eTsin = self._sin8 - Tsin
        eTcos = self._cos8 - Tcos

        Tout_max_9h = float(np.max(Tout_C_seq))
        Solar_max_9h = float(np.max(Solar_seq))

        x = (
            [Tset_C, Tin_C, eT, u_prev, eThigh, eTlow]
            + Tout_C_seq
            + Solar_seq
            + Price_seq
            + Thigh_seq
            + Tlow_seq
            + [Tsin, Tcos, eTsin, eTcos]
            + [Tout_max_9h, Solar_max_9h]
        )

        return np.asarray(x, dtype=np.float32)
class ActionLinkWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def action(self, action):
        u_hp = np.clip(action[0], 0.0, 1.0)
        if u_hp > 0.05:
            u_fan, u_pump = max(0.2, u_hp), max(0.2, u_hp)
        else:
            u_hp, u_fan, u_pump = 0.0, 0.0, 0.0
        return np.array([u_hp, u_fan, u_pump], dtype=np.float32)
# =============================================================

class HighPenaltyEnv(BoptestGymEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_u = 0.0
        self.current_u = 0.0
        self.t_target = 22.0
        self.objective_integrand = 0.0


    def reset(self, **kwargs):
        self.last_u = 0.0
        self.current_u = 0.0
        self.objective_integrand = 0.0   #  防止跨场景切换奖励突刺
        return super().reset(**kwargs)

    def step(self, action):
        self.last_u = self.current_u
        # action 经过 ActionLinkWrapper 后变成了 3维，这里取第0位(热泵)
        self.current_u = float(action[0])
        return super().step(action)

    def get_reward(self):
        """
        集成了非对称惩罚逻辑的奖励函数
        目标：严厉打击 Typical 赛季的过热违约（Upper Violation）
        """
        import requests
        import numpy as np
        from main import get_comfort_bounds

        # --- 1. 获取基础数据 ---
        # 获取当前仿真时间（秒）
        res = self.unwrapped.last_res
        if isinstance(res, dict):
            current_time = float(res.get("time", 0))
        else:
            # 打印异常信息，方便排查，同时赋默认值不崩溃
            print(f"[Reward异常] res类型错误：{type(res)}，内容：{res}")
            current_time = 0.0
            # 获取当前室内温度 (Celsius)
        Tz_deg = float(res.get("reaTZon_y", 293.15)) - 273.15

        # 获取当前时刻的舒适度红线
        t_low, t_high = get_comfort_bounds(current_time)

        # --- 2. 计算基础奖励 (基于 BOPTEST KPI) ---
        # w 是 BOPTEST 默认的不舒适度权重
        w = 1000.0
        kpis = requests.get('{0}/kpi/{1}'.format(self.url, self.testid)).json()['payload']
        objective_integrand = kpis['cost_tot'] + w * kpis['tdis_tot']

        # 计算本步长内总指标的变化负值
        base_reward = -(objective_integrand - self.objective_integrand)
        self.objective_integrand = objective_integrand

        # --- 3. 动作平滑惩罚 ---
        lambda_smooth = 5.0
        smooth_penalty = -lambda_smooth * np.power(self.current_u - self.last_u, 2)

        # --- 4. 非对称温度追踪惩罚 (核心改进) ---
        # 逻辑：在 Typical 赛季，过热的代价远高于过冷
        k_track_base = 20.0  # 基础追踪权重

        # 定义非对称系数
        # 如果温度超过上限，惩罚力度放大 3 倍（具体倍数可根据实验效果微调）
        asymmetric_factor = 1.0
        if Tz_deg > t_high:
            asymmetric_factor = 15.0  # 严厉惩罚过热
        elif Tz_deg < t_low:
            asymmetric_factor = 1.0  # 维持标准惩罚

        # 计算当前温度与目标设定点 (t_target) 的偏差惩罚
        # 使用超额权重 asymmetric_factor
        track_penalty = - (k_track_base * asymmetric_factor) * abs(self.t_target - Tz_deg)

        # --- 5. 综合奖励输出 ---
        # 总奖励 = 基础 KPI 奖励 + 平滑惩罚 + 非对称追踪惩罚
        return base_reward + smooth_penalty + track_penalty




def make_env():
    observations = {
        "time": (0.0, 365.0 * 24 * 3600.0),
        "reaTZon_y": (280.0, 310.0),
        "weaSta_reaWeaTDryBul_y": (253.15, 313.15),
        "weaSta_reaWeaHGloHor_y": (0.0, 1000.0),
        "reaPHeaPum_y": (0.0, 5000.0),
        "reaPFan_y": (0.0, 1000.0),
        "reaPPumEmi_y": (0.0, 500.0),
        "reaTSetHea_y": (280.0, 310.0),
    }
    actions = ["oveHeaPumY_u", "oveFan_u", "ovePum_u"]

    env = HighPenaltyEnv(
        url=BOPTEST_URL,
        testcase="bestest_hydronic_heat_pump",
        actions=actions,
        observations=observations,
        reward=["reward"],
        max_episode_length=24 * 3600,
        random_start_time=False,
        start_time=0,
        step_period=900,
        warmup_period=12 * 3600,
        render_episodes=False,
        scenario={"electricity_price": "highly_dynamic"},
    )

    env = ActionLinkWrapper(env)
    env = SuperAugmentedObservationWrapper(env)  # 使用新的超强包装器
    env = Teacher190NormalizeWrapper(env)
    env = ClipObsWrapper(env)
    return env