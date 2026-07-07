import torch
import torch.nn.functional as F
import numpy as np
from collections import deque

from . import utils, models

# =========================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================================================================

BATCH_SIZE = 64
LEARNING_RATE = 0.0005
GAMMA = 0.99
TAU = 0.001

# TCN 需要固定历史窗口长度（建议 12~24）
SEQ_LEN = 12


class DDPGTAgent:
    def __init__(self, state_dim, action_dim, action_lim, ram, teacher):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.ram = ram
        self.teacher = teacher

        self.lambda_param = 10.0
        self.tau_param = 0.15
        self.tau_temp = 0.5

        self.alpha_action_rec = 1.0
        self.alpha_temp_rec = 1.0
        self.alpha = 1.0
        self.TIN_LOW = 15.0
        self.TIN_HIGH = 30.0

        # --- 1) 用 TCN 版本网络 ---
        self.actor = models.ActorTCN(state_dim, action_dim, action_lim).to(DEVICE)
        self.target_actor = models.ActorTCN(state_dim, action_dim, action_lim).to(DEVICE)
        self.critic = models.CriticTCN(state_dim, action_dim).to(DEVICE)
        self.target_critic = models.CriticTCN(state_dim, action_dim).to(DEVICE)

        # --- 2) 在线推理用历史窗口（替代 LSTM hidden） ---
        self.state_hist = deque(maxlen=SEQ_LEN)

        # 噪声
        self.noise = utils.OUNoise(action_dim)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LEARNING_RATE)

        utils.hard_update(self.target_actor, self.actor)
        utils.hard_update(self.target_critic, self.critic)

    def reset_hidden(self):
        """每个 Episode 开始 / 场景切换必须清空历史窗口"""
        self.state_hist.clear()


    def get_action_and_temp_combined(self, state, use_noise=True):
        # TCN 需要 (B,T,Ds)
        self.state_hist.append(np.asarray(state, dtype=np.float32))

        if len(self.state_hist) < SEQ_LEN:
            pad = [self.state_hist[0]] * (SEQ_LEN - len(self.state_hist))
            seq = pad + list(self.state_hist)
        else:
            seq = list(self.state_hist)

        state_seq = np.stack(seq, axis=0)[None, :, :]  # (1,T,Ds)
        state_t = torch.from_numpy(state_seq).float().to(DEVICE)

        with torch.no_grad():
            action, temp, _ = self.actor(state_t, hidden=None)

        action_np = action.detach().cpu().numpy().flatten()

        if use_noise:
            noise = self.noise.noise()
            action_np = np.clip(action_np + noise, 0.0, self.action_lim)

        # temp 是 [-1,1]，这里输出成 °C 更直观
        temp_norm = temp.detach().cpu()
        temp_C = (temp_norm + 1.0) * 0.5 * (30.0 - 15.0) + 15.0  
        temp_val = float(temp_C.numpy())
        return action_np, temp_val

    def get_action_no_noise(self, state):
        action, _ = self.get_action_and_temp_combined(state, use_noise=False)
        return action

    def save_models(self, path):
        import os
        directory = os.path.dirname(path)
        if directory and (not os.path.exists(directory)):
            os.makedirs(directory)
        torch.save(self.actor.state_dict(), path + "_actor.pt")
        torch.save(self.critic.state_dict(), path + "_critic.pt")
        torch.save(self.target_actor.state_dict(), path + "_target_actor.pt")
        torch.save(self.target_critic.state_dict(), path + "_target_critic.pt")
        print(f"   >>> Models saved to {path}...")

    def load_models(self, path):
        self.actor.load_state_dict(torch.load(path + "_actor.pt", map_location=DEVICE))
        self.critic.load_state_dict(torch.load(path + "_critic.pt", map_location=DEVICE))
        self.target_actor.load_state_dict(torch.load(path + "_target_actor.pt", map_location=DEVICE))
        self.target_critic.load_state_dict(torch.load(path + "_target_critic.pt", map_location=DEVICE))
        print(f"   >>> Models loaded from {path}...")


    def optimize(self):
        TIN_LOW, TIN_HIGH = 15.0, 30.0 
        TSET_LOW, TSET_HIGH = 15.0, 30.0  
        ET_LOW, ET_HIGH = -15.0, 15.0
        if len(self.ram) < BATCH_SIZE + SEQ_LEN:
            return

        samples = self.ram.sample_sequence(
            BATCH_SIZE, SEQ_LEN,
            with_t_target=True,
            with_next_t_target=True,  
            with_fusion=True
        )

        if samples is None:
            return

        (s1_np, a1_np, r1_np, s2_np, done_np,
         t_a_np, t_rule_np, t_target_np, t_target_next_np, 
         phys_r_np, phys_c_np,
         t_a_next_np, alpha_np, alpha_next_np) = samples

        s1 = torch.from_numpy(s1_np).to(DEVICE)
        a1_seq = torch.from_numpy(a1_np).to(DEVICE)
        r1_seq = torch.from_numpy(r1_np).to(DEVICE)
        s2 = torch.from_numpy(s2_np).to(DEVICE)
        done_seq = torch.from_numpy(done_np).to(DEVICE)
        t_target_seq = torch.from_numpy(t_target_np).to(DEVICE)

        teacher_actions_seq = torch.from_numpy(t_a_np).to(DEVICE)
        rule_t_set_seq = torch.from_numpy(t_rule_np).to(DEVICE) 

        teacher_next_seq = torch.from_numpy(t_a_next_np).to(DEVICE)  # (B,T,Adim)
        alpha_seq = torch.from_numpy(alpha_np).to(DEVICE)  # (B,T,1)
        alpha_next_seq = torch.from_numpy(alpha_next_np).to(DEVICE)  # (B,T,1)
        curr_alpha = alpha_seq[:, -1, :]  # (B,1)
        next_alpha = alpha_next_seq[:, -1, :]  # (B,1)

        next_teacher_a = teacher_next_seq[:, -1, :]  # (B,Adim)

        curr_a = a1_seq[:, -1, :]
        curr_r = r1_seq[:, -1, :]
        curr_done = done_seq[:, -1, :].view(-1)
        curr_t_target = t_target_seq[:, -1, :]
        curr_teacher_a = teacher_actions_seq[:, -1, :]
        curr_rule_t_set = rule_t_set_seq[:, -1, :]  # (B,1)
        t_target_next_seq = torch.from_numpy(t_target_next_np).to(DEVICE)  # (B,T,1)

        def norm_scalar_torch(x, low, high):
            return (x - low) / (high - low) * 2.0 - 1.0

        def denorm_scalar_torch(x_norm, low, high):
            return (x_norm + 1.0) * 0.5 * (high - low) + low

        # s2: (B,T,dim)
        Tin_next_norm = s2[:, :, 1:2]  # (B,T,1) in [-1,1]
        Tin_next_C = denorm_scalar_torch(Tin_next_norm, TIN_LOW, TIN_HIGH)

        tset_next_norm = norm_scalar_torch(t_target_next_seq, TSET_LOW, TSET_HIGH)
        eT_next_norm = norm_scalar_torch(t_target_next_seq - Tin_next_C, ET_LOW, ET_HIGH)

        s2 = s2.clone()
        s2[:, :, 0:1] = tset_next_norm
        s2[:, :, 2:3] = eT_next_norm

        # ---------------- Critic ----------------
        with torch.no_grad():
            next_student, _, _ = self.target_actor(s2)  # (B,Adim)

           
            next_fused = next_alpha * next_teacher_a + (1.0 - next_alpha) * next_student

            next_val, _ = self.target_critic(s2, next_fused)
            next_val = next_val.view(-1)

            y_expected = curr_r.view(-1) + GAMMA * (1.0 - curr_done) * next_val

        y_predicted, _ = self.critic(s1, curr_a)
        y_predicted = y_predicted.view(-1)
        loss_critic = F.mse_loss(y_predicted, y_expected)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()

        # ---------------- Actor ----------------
        pred_u, pred_t, _ = self.actor(s1)


        alpha_det = curr_alpha.detach()  # (B,1)
        teacher_det = curr_teacher_a.detach()  # (B,Adim)

        fused_pred = alpha_det * teacher_det + (1.0 - alpha_det) * pred_u

        q_val, _ = self.critic(s1, fused_pred)
        loss_ddpg = -torch.mean(q_val)

        w = curr_alpha.detach()
        loss_guide_action = torch.mean(w * (pred_u - curr_teacher_a).pow(2))

        with torch.no_grad():
            D_action_t = torch.mean(torch.norm(pred_u - curr_teacher_a, p=2, dim=1))
            alpha_action_t = torch.sigmoid((D_action_t - self.tau_param) * self.lambda_param)
            alpha_action_t = torch.clamp(alpha_action_t, 0.05, 0.95)
        alpha_action = float(alpha_action_t.item())
        self.alpha_action_rec = alpha_action

      
        Tin_next_C_last = Tin_next_C[:, -1, :].detach()  # (B,1) 摄氏度标签

     
        pred_t_C = denorm_scalar_torch(pred_t, TIN_LOW, TIN_HIGH)  # (B,1) °C

        loss_guide_temp = F.mse_loss(pred_t_C, Tin_next_C_last)

        with torch.no_grad():
            D_temp_t = torch.mean(torch.abs(pred_t_C - Tin_next_C_last))
            alpha_temp_t = torch.sigmoid((D_temp_t - self.tau_temp) * self.lambda_param)
            alpha_temp_t = torch.clamp(alpha_temp_t, 0.05, 0.95)

        alpha_temp = float(alpha_temp_t.item())
        self.alpha_temp_rec = alpha_temp

        Tin_norm = s1[:, -1, 1].view(-1, 1)  # [-1,1]
        Tin_now_C = denorm_scalar_torch(Tin_norm, TIN_LOW, TIN_HIGH)

        T_now_C = Tin_now_C

        temp_error = torch.abs(curr_t_target - T_now_C)

        phys_weight = torch.clamp((temp_error - 0.2) / (0.5 - 0.2), 0.0, 1.0)
        k_phys_base = 20.0
        dynamic_k = k_phys_base * phys_weight
        loss_phys_guidance = -torch.mean(dynamic_k * (curr_t_target - T_now_C) * pred_u)

        loss_actor = (
            loss_ddpg
            + (alpha_action * loss_guide_action * 100.0)
            + (alpha_temp * loss_guide_temp * 10.0)
            + loss_phys_guidance
        )

        self.actor_optimizer.zero_grad()
        loss_actor.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()

        # ---------------- Soft Update ----------------
        utils.soft_update(self.target_actor, self.actor, TAU)
        utils.soft_update(self.target_critic, self.critic, TAU)
