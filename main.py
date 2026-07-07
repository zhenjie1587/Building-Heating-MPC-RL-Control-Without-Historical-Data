import matplotlib
matplotlib.use('Agg')  # 必须放在最前
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
import os
import requests
import pandas as pd
HOME_UPPER_BUFFER = 0.2   # °C，给在家上限留余量
LEN = 36
def tout_to_c_scalar(t):
    """Tout 可能是 K 或 ℃，统一转成 ℃"""
    t = float(t)
    return (t - 273.15) if t > 150.0 else t

def tout_to_c_array(arr):
    """Tout 序列可能是 K 或 ℃，统一转成 ℃"""
    a = np.asarray(arr, dtype=float)
    return np.where(a > 150.0, a - 273.15, a)

def debug_print_teacher190(x, tag="state"):
    x = np.asarray(x, dtype=float).reshape(-1)
    print(f"\n[{tag}] shape={x.shape} dtype={x.dtype}")

    print("  head6 [Tset,Tin,eT,u_prev,eThigh,eTlow] =", x[:6])

    Tout  = x[6:6+LEN]
    Solar = x[6+LEN:6+2*LEN]
    Price = x[6+2*LEN:6+3*LEN]
    Thigh = x[6+3*LEN:6+4*LEN]
    Tlow  = x[6+4*LEN:6+5*LEN]

    print("  Tout[0:3] =", Tout[:3],  " ... last =", Tout[-1])
    print("  Solar[0:3]=", Solar[:3], " ... last =", Solar[-1])
    print("  Price[0:3]=", Price[:3], " ... last =", Price[-1])
    print("  Thigh[0:3]=", Thigh[:3], " ... last =", Thigh[-1])
    print("  Tlow[0:3] =", Tlow[:3],  " ... last =", Tlow[-1])

    print("  time4 [Tsin,Tcos,eTsin,eTcos] =", x[186:190])


from collections import defaultdict
from collections import deque
from datetime import datetime, timedelta

from algorithmsDDPGT.buffer import MemoryBuffer
from algorithmsDDPGT.ddpgt import DDPGTAgent
import torch
from algorithmsDDPGT.high_level_agent import HighLevelAgent, _rc_to_feat_np

from teacher.mpc_teacher import MPCTeacher
from teacher.rule_temp_teacher import get_adaptive_setpoint
# main.py

from env_wrapper import make_env, BOPTEST_URL, SuperAugmentedObservationWrapper

PARAMS_DICT = {
    'Peak':    {'R': 0.01, 'C': 1.0e7, 'A_solar': 5, 'Q_gain': 0.0, 'dt': 900},
    'Typical': {'R': 0.01, 'C': 1.0e7, 'A_solar': 5, 'Q_gain': 0.0, 'dt': 900},
}

LOG_DIR = './Results_Direct_Test'
os.makedirs(LOG_DIR, exist_ok=True)

TEST_SCENARIOS = {
    "Peak": {"start_day": 16, "days": 14},
    "Typical": {"start_day": 108, "days": 14}
}

STEPS_PER_DAY = 96
MEMORY_SIZE = 20000
BATCH_SIZE = 64
# ===== UTD (Updates-To-Data) 相关超参数 =====
UPDATES_PER_STEP = 20            # 低层DDPG：每个真实step更新10次（先从5/10试）
HL_UPDATES_PER_STEP = 2          # 高层设定点：每步更新更少（1~2比较稳）


def get_forecast_from_wrapper(env):
    curr_env = env
    while hasattr(curr_env, 'env'):
        if isinstance(curr_env, SuperAugmentedObservationWrapper):
            return curr_env.last_forecast_dict
        curr_env = curr_env.env
    print("Warning: AugmentedObservationWrapper not found in environment stack!")
    return None


def compute_safety_metric(hist):
    Tz = np.array(hist["t_zon"])
    Tlow = np.array(hist["t_lower"])
    Thigh = np.array(hist["t_upper"])

    min_len = min(len(Tz), len(Tlow), len(Thigh))
    Tz, Tlow, Thigh = Tz[:min_len], Tlow[:min_len], Thigh[:min_len]


    outside = (Tz < Tlow) | (Tz > Thigh)
    r_time = outside.mean()


    denom_high_K = Thigh + 273.15
    denom_low_K = Tlow + 273.15

    overshoot = np.maximum(Tz - Thigh, 0.0) / denom_high_K
    undershoot = np.maximum(Tlow - Tz, 0.0) / denom_low_K


    r_sev = float(np.max(np.maximum(overshoot, undershoot))) if len(overshoot) > 0 else 0.0


    return r_time + r_sev


# main.py

def get_comfort_bounds(time_seconds):
    base_date = datetime(2019, 1, 1)

    current_dt = base_date + timedelta(seconds=float(time_seconds))
    weekday = current_dt.weekday()
    hour = current_dt.hour + current_dt.minute / 60.0

    t_low, t_high = 21.0, 24.0
    if weekday < 5 and 7.0 <= hour < 20.0:
        t_low, t_high = 15.0, 30.0
    return t_low, t_high

def plot_time_series(csv_path, scenario_name):
    print(f">>> 正在为 {scenario_name} 生成 5 行时间序列图...")
    df = pd.read_csv(csv_path)
    t = df['Time_Step'] / 96.0

    fig, axes = plt.subplots(5, 1, figsize=(15, 22), sharex=True)
    plt.subplots_adjust(hspace=0.25)

    axes[0].step(t, df['Price'], color='black', where='post', label='Real Price')
    axes[0].set_ylabel('Price ($/kWh)', fontweight='bold')
    axes[0].set_title(f'1. Electricity Price Signal ({scenario_name})', loc='left', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].fill_between(t, df['T_Lower'], df['T_Upper'], color='gray', alpha=0.2, label='Comfort Range')
    axes[1].plot(t, df['T_Zone'], color='#d62728', linewidth=1.5, label='Zone Temp')
    axes[1].plot(t, df['T_Upper_Ctrl'], linestyle='--', linewidth=1.2, label='Upper (Ctrl)')
    axes[1].set_ylabel('Temp (°C)', fontweight='bold')
    axes[1].set_ylim(14, 30)
    axes[1].set_title('2. Thermal Comfort Tracking', loc='left', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    axes[2].plot(t, df['Action_MPC'], color='green', linestyle='--', alpha=0.6, label='MPC Action')
    axes[2].plot(t, df['Action_Agent'], color='blue', linestyle=':', alpha=0.8, label='RL Action')
    axes[2].plot(t, df['Action_Final'], color='black', linewidth=1.5, alpha=0.9, label='Final Action')
    axes[2].set_ylabel('Pump Control (u)', fontweight='bold')
    axes[2].set_title('3. Pump Operation', loc='left', fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right', ncol=3)

    axes[3].plot(t, df['Alpha_Action'], color='purple', linewidth=2, label='Alpha_MPC (Action)')
    axes[3].set_ylabel('Guidance Weight', fontweight='bold')
    axes[3].set_title('4. Dual-Teacher Learning Process (Guidance Weights)', loc='left', fontweight='bold')
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right')

    axes[4].plot(t, df['T_Out'], color='tab:blue', label='Outdoor Temp')
    axes[4].set_ylabel('Outdoor Temp (°C)', fontweight='bold', color='tab:blue')
    axes[4].tick_params(axis='y', labelcolor='tab:blue')

    ax5t = axes[4].twinx()
    ax5t.fill_between(t, 0, df['Q_Sol'], color='orange', alpha=0.3, label='Solar')
    ax5t.set_ylabel('Solar (W/m²)', fontweight='bold', color='tab:orange')
    ax5t.tick_params(axis='y', labelcolor='tab:orange')
    ax5t.set_ylim(0, 1100)

    axes[4].set_title('5. Environmental Disturbances', loc='left', fontweight='bold')
    axes[4].set_xlabel('Time (Days)', fontsize=14)
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, f'{scenario_name}_TimeSeries.png'), dpi=300)
    plt.close(fig)


def plot_kpis_comparison(kpi_peak, ms_peak, kpi_typ, ms_typ, savepath):
    print(f">>> 正在生成 KPI 对比图...")
    labels = ["Cost", "Energy", "Emissions", "Discomfort", "MS"]

    p_data = [kpi_peak['cost_tot'], kpi_peak['ener_tot'], kpi_peak['emis_tot'], kpi_peak['tdis_tot'], ms_peak]
    t_data = [kpi_typ['cost_tot'], kpi_typ['ener_tot'], kpi_typ['emis_tot'], kpi_typ['tdis_tot'], ms_typ]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=150)
    ax2 = ax1.twinx()

    idx_dis = 3
    idx_others = [0, 1, 2, 4]

    ax1.bar(x[idx_dis] - width / 2, p_data[idx_dis], width, color='tab:blue', hatch='//', alpha=0.9)
    for i in idx_others:
        ax2.bar(x[i] - width / 2, p_data[i], width, color='tab:blue')

    ax1.bar(x[idx_dis] + width / 2, t_data[idx_dis], width, color='tab:orange', hatch='//', alpha=0.9)
    for i in idx_others:
        ax2.bar(x[i] + width / 2, t_data[i], width, color='tab:orange')

    for i in range(5):
        ax = ax1 if i == idx_dis else ax2
        ax.text(x[i] - width / 2, p_data[i], f'{p_data[i]:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(x[i] + width / 2, t_data[i], f'{t_data[i]:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel("Discomfort (K·h)", color='tab:red', fontweight='bold')
    ax2.set_ylabel("Cost / Energy / Emissions / MS", color='black', fontweight='bold')

    legend_patches = [
        mpatches.Patch(color='tab:blue', label='Peak Season'),
        mpatches.Patch(color='tab:orange', label='Typical Season'),
        mpatches.Patch(facecolor='gray', hatch='//', alpha=0.5, label='Discomfort')
    ]
    ax1.legend(handles=legend_patches, loc='upper left')
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close(fig)


def plot_diagnostic_analysis(csv_path, scenario_name):
    """
    系统诊断分析绘图：包含估计误差、控制误差、高层决策对比及物理参数演化
    """
    print(f">>> 正在为 {scenario_name} 生成诊断分析图...")
    df = pd.read_csv(csv_path)

    t = df['Time_Step'] / 96.0


    fig, axes = plt.subplots(4, 1, figsize=(16, 22), sharex=True)
    plt.subplots_adjust(hspace=0.28)


    axes[0].plot(t, df['RC_Error'], color='tab:blue', alpha=0.8,
                 label='Step-wise RC Prediction Error (T_zone - T_pred_rc)')
    axes[0].plot(t, df['MHE_Error'], color='tab:orange', linestyle='--', alpha=0.6, label='MHE Static Error (Residual)')
    axes[0].axhline(0, color='black', linewidth=1.0, linestyle='-')
    axes[0].set_ylabel('Estimation Error (K)', fontweight='bold')
    axes[0].set_title('1. Physical Model Estimation Fidelity (Is the Model Accurate?)', loc='left', fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)


    axes[1].fill_between(t, df['T_Lower'], df['T_Upper'], color='gray', alpha=0.1, label='Comfort Zone')
    axes[1].plot(t, df['Temp_Rule'], color='tab:green', linestyle='--', label='Rule-based Setpoint')
    axes[1].plot(t, df['Tset_AI'], color='tab:red', alpha=0.8, label='High-Level RL Setpoint')
    axes[1].plot(t, df['Temp_Target'], color='black', linewidth=1.5, label='Final Fused Target (T_target)')
    axes[1].set_ylabel('Temperature (°C)', fontweight='bold')
    axes[1].set_title('2. High-Level Setpoint Comparison', loc='left', fontweight='bold')
    axes[1].legend(loc='upper right', ncol=2)
    axes[1].grid(True, alpha=0.3)


    control_err = df['Temp_Target'] - df['T_Zone']
    consistency_err = df['Temp_Target'] - df['T_Pred_RC']

    axes[2].plot(t, control_err, color='tab:purple', linewidth=1.5, label='Control Error (T_target - T_zone)')
    axes[2].plot(t, consistency_err, color='tab:pink', linestyle=':', alpha=0.7,
                 label='Consistency Discrepancy (Target - MPC_Pred)')
    axes[2].axhline(0, color='black', linewidth=1.0, linestyle='-')
    axes[2].set_ylabel('Tracking Error (K)', fontweight='bold')
    axes[2].set_title('3. Control Performance & Reachability (Is the Target Met?)', loc='left', fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)


    ax4_right = axes[3].twinx()
    p1, = axes[3].plot(t, df['R_est'], color='blue', label='R_est (Thermal Resistance)')
    p2, = ax4_right.plot(t, df['C_est'], color='tab:orange', alpha=0.7, label='C_est (Thermal Capacity)')


    update_pts = df[df['MHE_Updated'] == 1].index
    if not update_pts.empty:
        axes[3].scatter(t[update_pts], df['R_est'][update_pts], color='red', marker='v', s=50, label='MHE Update Point')

    axes[3].set_ylabel('R Value', color='blue', fontweight='bold')
    ax4_right.set_ylabel('C Value', color='tab:orange', fontweight='bold')
    axes[3].set_title('4. Online Parameter Identification Stability (MHE)', loc='left', fontweight='bold')


    lines = [p1, p2]
    labels = [l.get_label() for l in lines]
    axes[3].legend(lines, labels, loc='upper right')
    axes[3].grid(True, alpha=0.3)

    axes[3].set_xlabel('Time (Days)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, f'{scenario_name}_Diagnostic_Analysis.png'), dpi=300)
    plt.close(fig)


def update_super_wrapper_vars(env, var_dict):
    curr_env = env
    while hasattr(curr_env, 'env'):
        if curr_env.__class__.__name__ == 'SuperAugmentedObservationWrapper':
            curr_env.update_external_vars(var_dict)   # ✅ 用 wrapper 自带方法
            return
        curr_env = curr_env.env
def norm_scalar(x, low, high):
    x = float(x)
    return (x - low) / (high - low) * 2.0 - 1.0
def _find_wrapper_by_name(env, cls_name: str):
    cur = env
    while True:
        if cur.__class__.__name__ == cls_name:
            return cur
        if not hasattr(cur, "env"):
            return None
        cur = cur.env

def denorm_teacher190(env, x_norm):
    """
    把 Teacher190NormalizeWrapper 输出的 [-1,1] teacher-190 还原回物理量纲（°C / W/m² / Price）
    """
    w = _find_wrapper_by_name(env, "Teacher190NormalizeWrapper")
    if w is None:
        return np.asarray(x_norm, dtype=np.float32)

    x = np.asarray(x_norm, dtype=np.float32)
    x = (x + 1.0) * 0.5 * w._scale + w._low
    return x.astype(np.float32)

def run_scenario_test(env, agent, teacher, high_level_agent, scenario_name, config, memory: MemoryBuffer):

    w_ai_smooth = 0.0
    GATE_BETA = 1.0  
    GATE_LAMBDA = 0.10 
    GATE_MIN_AI = 0.05  
    GATE_MAX_AI = 0.95
    GATE_START_AFTER = 500 

    start_day = config['start_day']
    days = config['days']
    print(f"\n>>> [Test Start] {scenario_name} (Start Day: {start_day}, Duration: {days} days)")

    start_seconds = start_day * 24 * 3600
    env.unwrapped.start_time = start_seconds
    print(f">>> 正在同步 BOPTEST 服务器场景设置...")
    scenario_params = {
        "time_period": f"{days}d",  # 测试持续天数
        "start_time": start_seconds,  # 开始的时间点（秒）
        "electricity_price": "highly_dynamic"  # 切换到高度动态电价
    }

    env.unwrapped.scenario = {"electricity_price": "highly_dynamic"}


   
    res_scenario = requests.put(f"{BOPTEST_URL}/scenario/{env.unwrapped.testid}", json=scenario_params).json()

 
    if res_scenario['status'] == 200:
        initial_payload = res_scenario['payload']
     
        env.unwrapped.last_res = initial_payload

        env.unwrapped.start_time = start_seconds
        env.unwrapped.scenario = {"electricity_price": "highly_dynamic"}
        state, _ = env.reset()

        print("state shape:", np.asarray(state).shape)  # teacher-190
        print("env obs space:", env.observation_space.shape)

    else:
        print("Warning: Scenario setup failed! Fallback to standard reset.")
        state, _ = env.reset()
    agent.noise.reset()
    agent.reset_hidden()
    high_level_agent.reset_hidden()

    detail_log = defaultdict(list)
    hist_for_ms = defaultdict(list)
    total_steps = days * STEPS_PER_DAY
    mhe_window = 96
    update_every = 96
    mhe_history = {
        't_zon': deque(maxlen=mhe_window),
        't_out': deque(maxlen=mhe_window),
        'q_sol': deque(maxlen=mhe_window),
        'u_act': deque(maxlen=mhe_window),
    }
    last_mhe_loss = 0.0
    home_step = 0
    away_step = 0
    last_u_rl = 0.0
    last_step_pred_deg = None  
    for step in range(total_steps):
        current_time = start_seconds + step * 900
        forecast_data = get_forecast_from_wrapper(env)

        if forecast_data and ('Qsol' in forecast_data) and ('Tout' in forecast_data):
            qsol_arr = np.array(forecast_data['Qsol'], dtype=float)  # W/m2
            tout_arr_c = tout_to_c_array(forecast_data['Tout'])  # ✅ 自动识别 K/℃


            H6 = min(len(qsol_arr), 24)  
            H24 = min(len(qsol_arr), 96)  

            future6_qsol_max = float(np.max(qsol_arr[:H6])) if H6 > 0 else 0.0
            future6_tout_max = float(np.max(tout_arr_c[:H6])) if H6 > 0 else 0.0

            future24_qsol_avg = float(np.mean(qsol_arr[:H24])) if H24 > 0 else 0.0
            future24_qsol_max = float(np.max(qsol_arr[:H24])) if H24 > 0 else 0.0

            future24_tout_max = float(np.max(tout_arr_c[:H24])) if H24 > 0 else 0.0
        else:
            future6_qsol_max = 0.0
            future6_tout_max = 0.0
            future24_qsol_avg = 0.0
            future24_qsol_max = 0.0
            future24_tout_max = 0.0
        # =============================================================

        mhe_updated = 0

        res_prev = env.unwrapped.last_res or {}
        Tz_prev_deg = float(res_prev.get("reaTZon_y", 293.15)) - 273.15
        Tout_prev_deg = float(res_prev.get("weaSta_reaWeaTDryBul_y", 283.15)) - 273.15
        Qsol_prev = float(res_prev.get("weaSta_reaWeaHGloHor_y", 0.0))
        Tz_now_deg = float(res_prev.get("reaTZon_y", 293.15)) - 273.15

        mhe_history['t_zon'].append(Tz_prev_deg + 273.15)
        mhe_history['t_out'].append(Tout_prev_deg + 273.15)
        mhe_history['q_sol'].append(Qsol_prev)

  
        if forecast_data and 'Qsol' in forecast_data:
            qsol_arr = np.array(forecast_data['Qsol'], dtype=float)
            H = min(len(qsol_arr), 96)  # 96步=24小时
            future_q_sol_avg = float(np.mean(qsol_arr[:H]))
            future_q_sol_max = float(np.max(qsol_arr[:H]))
        else:
            future_q_sol_avg, future_q_sol_max = 0.0, 0.0
   
        if last_step_pred_deg is not None:
          
            teacher.step_disturbance_observer(
                t_meas_deg=Tz_now_deg,
                t_pred_deg=last_step_pred_deg
            )

        r_now, c_now = teacher.R, teacher.C
        if step < 3:
            debug_print_teacher190(state, tag=f"state_before_high@step{step}")

        t_set_ai = high_level_agent.get_t_set(
            state,
            r=float(r_now),
            c=float(c_now)

        )

        curr_t_low, curr_t_high = get_comfort_bounds(current_time)
        is_away = (curr_t_low < 20.5)
        curr_t_high_ctrl = curr_t_high - (HOME_UPPER_BUFFER if (not is_away) else 0.0)

        t_set_ai = float(np.clip(t_set_ai, curr_t_low, curr_t_high_ctrl))
        if forecast_data and 'Tout' in forecast_data:
            curr_t_out_c = tout_to_c_scalar(forecast_data['Tout'][0])  # ✅ 自动识别 K/℃
            curr_q_sol = float(forecast_data['Qsol'][0])
        else:
            curr_t_out_c, curr_q_sol = 10.0, 0.0
        if forecast_data and 'Tout' in forecast_data:
            tout_arr_c = tout_to_c_array(forecast_data['Tout'])  # ✅ 自动识别 K/℃
            future_tout_avg = float(np.mean(tout_arr_c))
            future_tout_max = float(np.max(tout_arr_c))
        else:
            future_tout_avg, future_tout_max = 0.0, 0.0

        if forecast_data and 'Price' in forecast_data:
            # print("Price head:", forecast_data['Price'][:5])
            curr_price_real = float(forecast_data['Price'][0])
            avg_price_24h = float(np.mean(forecast_data['Price']))
        else:
            curr_price_real, avg_price_24h = 0.20, 0.25

        t_set_rule = get_adaptive_setpoint(
            curr_price_real, curr_t_out_c, curr_q_sol,
            curr_t_low, curr_t_high_ctrl,
            avg_price=avg_price_24h,
            future_q_sol_avg=future24_qsol_avg,
            future_q_sol_max=future24_qsol_max,
            future_tout_max=future24_tout_max,
            time_seconds=current_time
        )


        is_away = (curr_t_low < 20.5)

        # 1. 更新步数计数器
        if is_away:
            away_step += 1
            home_step = 0
            # 离家时的退火系数（3天内从规则过渡到AI）
            warmup_steps = 3 * 96
            current_step_count = away_step
        else:
            home_step += 1
            away_step = 0
          
            warmup_steps = 2 * 96
            current_step_count = home_step

       
        ai_floor = t_set_rule - (2.0 if is_away else 0.5)

      
        ai_floor = max(ai_floor, curr_t_low + 0.1)

       
        t_set_ai_safe = float(np.clip(t_set_ai, ai_floor, curr_t_high_ctrl))
       
        alpha_warm = max(0.0, 1.0 - current_step_count / float(warmup_steps))  # 1->0
        
      
        use_q_gate = (step >= GATE_START_AFTER) and (len(memory) >= BATCH_SIZE)

        if use_q_gate:
           
            seq_len = high_level_agent.seq_len
            s_hist = list(high_level_agent.state_hist)

            if len(s_hist) < seq_len:
                pad_s = [s_hist[0]] * (seq_len - len(s_hist))
                s_seq = pad_s + s_hist
            else:
                s_seq = s_hist[-seq_len:]

            s_t = torch.from_numpy(np.stack(s_seq, axis=0)[None, :, :]).float()
            dev = next(high_level_agent.critic.parameters()).device
            s_t = s_t.to(dev)

            physics_t = None
            if high_level_agent.critic.physical_dim > 0:
               
                p_hist = list(high_level_agent.physics_hist)
                if len(p_hist) < seq_len:
                    pad_p = [p_hist[0]] * (seq_len - len(p_hist))
                    p_seq = pad_p + p_hist
                else:
                    p_seq = p_hist[-seq_len:]
                physics_t = torch.from_numpy(np.stack(p_seq, axis=0)[None, :, :]).float().to(dev)

            with torch.no_grad():
                t_ai_t = torch.tensor([[float(t_set_ai_safe)]], dtype=torch.float32, device=dev)
                t_rule_t = torch.tensor([[float(t_set_rule)]], dtype=torch.float32, device=dev)

                q_ai, _ = high_level_agent.critic(s_t, physics=physics_t, t_set=t_ai_t, hidden=None)
                q_rule, _ = high_level_agent.critic(s_t, physics=physics_t, t_set=t_rule_t, hidden=None)

                q_ai = float(q_ai.item())
                q_rule = float(q_rule.item())

            # softmax -> w_ai_new (0~1)
         
            m = max(q_ai, q_rule)
            ea = np.exp(GATE_BETA * (q_ai - m))
            er = np.exp(GATE_BETA * (q_rule - m))
            w_ai_new = float(ea / (ea + er + 1e-12))

          
            w_ai_new = (1.0 - alpha_warm) * w_ai_new

           
            w_ai_new = float(np.clip(w_ai_new, GATE_MIN_AI, GATE_MAX_AI))
            w_ai_smooth = (1.0 - GATE_LAMBDA) * w_ai_smooth + GATE_LAMBDA * w_ai_new
            w_ai_smooth = float(np.clip(w_ai_smooth, GATE_MIN_AI, GATE_MAX_AI))

            alpha_hl = 1.0 - w_ai_smooth

        else:
          
            alpha_hl = alpha_warm

     
        t_target = alpha_hl * t_set_rule + (1.0 - alpha_hl) * t_set_ai_safe


        
        solar_risk = (future6_qsol_max > 600 and curr_t_out_c > 10.0)

        if solar_risk:
            t_target = min(t_target, curr_t_high_ctrl )  
            t_target = float(np.clip(t_target, curr_t_low, curr_t_high_ctrl))
        state_phys = denorm_teacher190(env, state) 
        action_teacher = teacher.get_action(
            state_phys, current_time,
            forecast_data=forecast_data,
            external_target_temp=t_target
        )


        u_mpc = float(action_teacher[0]) if not np.isscalar(action_teacher) else float(action_teacher)

   

        update_super_wrapper_vars(env, {
            "R": float(r_now),
            "C": float(c_now),
            "Q_gain": float(teacher.Q_gain),  
            "T_target": float(t_target),
            "u_mpc": u_mpc,
        })
        if np.isscalar(action_teacher):
            action_teacher = np.array([action_teacher], dtype=np.float32)



        state_ll = state.copy()


        Tin_now_C = Tz_prev_deg

        tset_norm = norm_scalar(t_target, low=15.0, high=30.0)
        eT_norm = norm_scalar(t_target - Tin_now_C, low=-15.0, high=15.0)

        state_ll[0] = tset_norm
        state_ll[2] = eT_norm

        action_student, pred_temp_student = agent.get_action_and_temp_combined(state_ll, use_noise=False)
        if np.isscalar(action_student):
            action_student = np.array([action_student], dtype=np.float32)

        future_upper_list = []
        for kk in range(1, 13):
            l_k, h_k = get_comfort_bounds(current_time + kk * 900)
            is_away_k = (l_k < 20.5)
            h_k_ctrl = h_k - (HOME_UPPER_BUFFER if (not is_away_k) else 0.0)
            future_upper_list.append(h_k_ctrl)

        future_upper_min = float(min(future_upper_list)) if len(future_upper_list) > 0 else float(curr_t_high_ctrl)
        # ====================================================================

        D_action = float(np.linalg.norm(action_student - action_teacher))
        alpha_online = 1.0 / (1.0 + np.exp(-agent.lambda_param * (D_action - agent.tau_param)))
        alpha_online = float(np.clip(alpha_online, 0.05, 0.95))

      
        margin = 0.6  
        k_risk = 10.0 

        risk_signal = (Tz_prev_deg - (future_upper_min - margin)) / max(margin, 1e-6)

        risk_gate = 1.0 / (1.0 + np.exp(-k_risk * risk_signal))  # 0~1

       
        alpha_final = alpha_online + (1.0 - alpha_online) * risk_gate
        alpha_final = float(np.clip(alpha_final, 0.05, 0.99))

      
        agent.alpha = alpha_final

        action_final = alpha_final * action_teacher + (1.0 - alpha_final) * action_student
        action_final = np.clip(action_final, 0.0, 1.0)


        update_super_wrapper_vars(env, {"u_rl": float(action_final[0])})
        env.unwrapped.t_target = t_target

        t_rc_pred_k = teacher.get_one_step_prediction(
            T_zone_K=Tz_prev_deg + 273.15,
            T_out_K=Tout_prev_deg + 273.15,
            Q_sol=Qsol_prev,
            action=action_final[0],
            R=teacher.R,
            C=teacher.C,
            Q_gain=teacher.Q_gain
        )
        last_step_pred_deg = t_rc_pred_k - 273.15
        t_rc_pred_deg = t_rc_pred_k - 273.15
    
        t_mhe_pred_deg = getattr(teacher, 'last_mhe_temp_k', Tz_prev_deg + 273.15) - 273.15
        mhe_history['u_act'].append(float(action_final[0]))

        next_state, reward, terminated, truncated, _ = env.step(action_final)
        last_u_rl = float(action_final[0])
        done = float(terminated or truncated or (step == total_steps - 1))

        if done:
            break

        memory.add(state_ll, action_final, reward, next_state, done,
                   action_teacher, t_set_rule, t_target,
                   r_now, c_now, alpha_final, alpha_hl)

        if len(memory) >= BATCH_SIZE:
            for _ in range(UPDATES_PER_STEP):
                agent.optimize()

       
            for _ in range(HL_UPDATES_PER_STEP):
                high_level_agent.optimize(memory, BATCH_SIZE,
                                          step)  

        res = env.unwrapped.last_res
        Tz_deg = float(res.get("reaTZon_y", 293.15)) - 273.15
        Tout_deg = float(res.get("weaSta_reaWeaTDryBul_y", 283.15)) - 273.15
        Q_sol = float(res.get("weaSta_reaWeaHGloHor_y", 0.0))

    
        t_mhe_step_pred_k = teacher.get_one_step_prediction(
            T_zone_K=Tz_deg + 273.15,
            T_out_K=Tout_deg + 273.15,
            Q_sol=Q_sol,
            action=action_final[0],  # `action_final[0]` 当前的热泵动作
            R=teacher.R,
            C=teacher.C,
            Q_gain=teacher.Q_gain
        )

        t_mhe_step_pred_deg = t_mhe_step_pred_k - 273.15  # 转换回摄氏度

      
        mhe_prediction_error = Tz_deg - t_mhe_step_pred_deg  # 逐步误差

        # 保存误差
        detail_log['MHE_Prediction_Error'].append(mhe_prediction_error)  # 把误差保存到日志中



       
        rc_error = Tz_deg - t_rc_pred_deg

       
        mhe_error = Tz_deg - t_mhe_pred_deg
        if len(mhe_history['t_zon']) == mhe_window and ((step + 1) % update_every == 0):
            success, current_loss = teacher.update_model_online(
                np.array(mhe_history['t_zon'], dtype=float),
                np.array(mhe_history['t_out'], dtype=float),
                np.array(mhe_history['q_sol'], dtype=float),
                np.array(mhe_history['u_act'], dtype=float),
            )
            mhe_updated = 1 if success else 0
            if success:
                last_mhe_loss = current_loss  # 记录最新的 loss
                print(f">>> [MHE] Success! Loss: {current_loss:.6f}")
                print(f">>> [MHE] 参数更新成功! R={teacher.R:.6f}, C={teacher.C:.3e}, c_hp={teacher.c_hp}")
            else:
                print(">>> [MHE] 更新失败，保持旧参数")

        detail_log['Time_Step'].append(step)
        detail_log['Price'].append(curr_price_real)
        detail_log['T_Zone'].append(Tz_deg)
        detail_log['T_Lower'].append(curr_t_low)
        detail_log['T_Upper'].append(curr_t_high)
        detail_log['Action_MPC'].append(float(action_teacher[0]))
        detail_log['Action_Agent'].append(float(action_student[0]))
        detail_log['Action_Final'].append(float(action_final[0]))
        # detail_log['Alpha'].append(float(agent.alpha))
        detail_log['Alpha_Online'].append(float(alpha_online))
        detail_log['Risk_Gate'].append(float(risk_gate))
        detail_log['Future_Upper_Min_3h'].append(float(future_upper_min))
        detail_log['Alpha'].append(float(agent.alpha))  
        detail_log['Alpha_Final'].append(float(alpha_final))
        detail_log['Alpha_Action'].append(float(agent.alpha_action_rec))
        detail_log['Alpha_Temp'].append(float(agent.alpha_temp_rec))
        detail_log['T_Out'].append(Tout_deg)
        detail_log['Q_Sol'].append(Q_sol)
        detail_log['Temp_Rule'].append(float(t_set_rule))
        detail_log['Temp_Target'].append(float(t_target))
        Tin_next_pred_C = (float(pred_temp_student) + 1.0) * 0.5 * (30.0 - 15.0) + 15.0
        detail_log['Temp_Agent'].append(float(Tin_next_pred_C))
        detail_log['Alpha_HL'].append(float(alpha_hl))
        detail_log['Is_Away'].append(1 if is_away else 0)
        detail_log['Home_Step'].append(int(home_step))
        detail_log['Violation'].append(1 if (Tz_deg < curr_t_low or Tz_deg > curr_t_high) else 0)
        detail_log['Tset_AI'].append(float(t_set_ai_safe))
        detail_log['R_est'].append(float(teacher.R))
        detail_log['C_est'].append(float(teacher.C))
        detail_log['A_solar_est'].append(float(teacher.A_solar))
        detail_log['Q_gain_est'].append(float(teacher.Q_gain))
        detail_log['c1_est'].append(float(teacher.c_hp[0]))
        detail_log['c2_est'].append(float(teacher.c_hp[1]))
        detail_log['c3_est'].append(float(teacher.c_hp[2]))
        detail_log['MHE_Updated'].append(int(mhe_updated))  
        detail_log['T_Zone_prev'].append(Tz_prev_deg)  # x_k
        detail_log['T_Out_prev'].append(Tout_prev_deg)
        detail_log['MHE_Loss'].append(float(last_mhe_loss))
        detail_log['T_Pred_RC'].append(float(t_rc_pred_deg))  
        detail_log['T_Pred_MHE'].append(float(t_mhe_pred_deg))
        detail_log['RC_Error'].append(float(rc_error)) 
        detail_log['MHE_Error'].append(float(mhe_error)) 
        detail_log['Reward'].append(float(reward))
        hist_for_ms['t_zon'].append(Tz_deg)
        hist_for_ms['t_lower'].append(curr_t_low)
        hist_for_ms['t_upper'].append(curr_t_high)
        detail_log['T_Upper_Ctrl'].append(curr_t_high_ctrl)
        state = next_state
        if terminated or truncated:
            high_level_agent.reset_hidden() 
            break

    df = pd.DataFrame(detail_log)
    csv_path = os.path.join(LOG_DIR, f'{scenario_name}_Details.csv')
    df.to_csv(csv_path, index=False)
    plot_time_series(csv_path, scenario_name)
    plot_diagnostic_analysis(csv_path, scenario_name)
    kpis = requests.get(f"{BOPTEST_URL}/kpi/{env.unwrapped.testid}").json()['payload']
    ms_val = compute_safety_metric(hist_for_ms)
    import json
    with open(os.path.join(LOG_DIR, f"{scenario_name}_kpi.json"), "w", encoding="utf-8") as f:
        json.dump(kpis, f, indent=2)


    df_tmp = pd.DataFrame(detail_log)
    away_mask = (df_tmp['Is_Away'] == 1)

    if away_mask.any():
        away_ai = df_tmp.loc[away_mask, 'Tset_AI'].mean()
        away_rule = df_tmp.loc[away_mask, 'Temp_Rule'].mean()
        away_target = df_tmp.loc[away_mask, 'Temp_Target'].mean()
        print(f"   [Away Avg] Tset_AI={away_ai:.2f} | Rule={away_rule:.2f} | Target={away_target:.2f}")
    else:
        print("   [Away Avg] No away period found.")
    # =============================================

    print(f"   -> [Result] {scenario_name} | Total Cost: {kpis['cost_tot']:.2f}, Safety Metric (MS): {ms_val:.4f}")
    return kpis, ms_val


if __name__ == "__main__":
    print(f">>> [System] 启动 BOPTEST 直接场景测试...")

    env = make_env()
    print("env obs dim =", env.observation_space.shape[0])
    s, _ = env.reset()
    print("state dim   =", np.asarray(s).shape[0])
    print("last2(norm) =", s[-2:])

    env.unwrapped.max_episode_length = 20 * 24 * 3600


    S_DIM = env.observation_space.shape[0]
    A_DIM = env.action_space.shape[0]
    A_MAX = float(env.action_space.high[0])

    print("\n------------------------------------------------------------")
    print(">>> 正在加载 [Peak] 冬季模型参数...")
    p_cfg = PARAMS_DICT['Peak']
    teacher_peak = MPCTeacher(
        R=p_cfg['R'], C=p_cfg['C'], A_solar=p_cfg['A_solar'],
        Q_gain=p_cfg['Q_gain'], dt=p_cfg['dt'], target_mode='middle'
    )

    memory = MemoryBuffer(MEMORY_SIZE)
    agent = DDPGTAgent(S_DIM, A_DIM, A_MAX, memory, teacher_peak)
    high_level_agent = HighLevelAgent(state_dim=S_DIM, physical_dim=2)

    kpi_peak, ms_peak = run_scenario_test(
        env, agent, teacher_peak, high_level_agent,
        "Peak", TEST_SCENARIOS["Peak"], memory
    )

    print("\n>>> [System] 检测到季节切换，正在重置记忆与高层网络...")
    memory.reset()

    high_level_agent = HighLevelAgent(state_dim=S_DIM, physical_dim=2)

    print("   -> 高层策略网络已重置为初始状态。")

    print("\n------------------------------------------------------------")
    print(">>> 正在切换到 [Typical] 春季模型参数...")
    t_cfg = PARAMS_DICT['Typical']
    teacher_typ = MPCTeacher(
        R=t_cfg['R'], C=t_cfg['C'], A_solar=t_cfg['A_solar'],
        Q_gain=t_cfg['Q_gain'], dt=t_cfg['dt'], target_mode='lower'
    )

    agent = DDPGTAgent(S_DIM, A_DIM, A_MAX, memory, teacher_typ)

    kpi_typ, ms_typ = run_scenario_test(
        env, agent, teacher_typ, high_level_agent,
        "Typical", TEST_SCENARIOS["Typical"], memory
    )

    plot_kpis_comparison(
        kpi_peak, ms_peak, kpi_typ, ms_typ,
        os.path.join(LOG_DIR, 'Comparison_BarChart.png')
    )

    print(f"\n>>> 完成。结果保存在 {LOG_DIR}")
