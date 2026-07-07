import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque

from .models import HighLevelCriticTCN, SetpointActorTCN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import torch

def _rc_to_feat_np(r: float, c: float) -> np.ndarray:
    """
    把 R,C 转成对网络友好的尺度：
      r_feat = log10(R) + 3   （假设 R ~ 1e-4..1e-2 -> feat ~ -1..1）
      c_feat = log10(C) - 7   （假设 C ~ 1e6..1e8 -> feat ~ -1..1）
    再做 clip 防止极端值。
    """
    eps = 1e-12
    r = max(float(r), eps)
    c = max(float(c), eps)
    r_feat = np.log10(r) + 3.0
    c_feat = np.log10(c) - 7.0
    r_feat = float(np.clip(r_feat, -3.0, 3.0))
    c_feat = float(np.clip(c_feat, -3.0, 3.0))
    return np.asarray([r_feat, c_feat], dtype=np.float32)

def _rc_to_feat_torch(r_t: torch.Tensor, c_t: torch.Tensor):
    eps = 1e-12
    r_log = torch.log10(torch.clamp(r_t, min=eps))
    c_log = torch.log10(torch.clamp(c_t, min=eps))
    r_feat = torch.clamp(r_log + 3.0, -3.0, 3.0)
    c_feat = torch.clamp(c_log - 7.0, -3.0, 3.0)
    return r_feat, c_feat

class HighLevelAgent:
    def __init__(self, state_dim, physical_dim=0, lr=0.0001, gamma=0.99, tau=0.001):
        self.gamma = gamma
        self.tau = tau

        
        self.actor = SetpointActorTCN(state_dim, physical_dim).to(DEVICE)
        self.target_actor = SetpointActorTCN(state_dim, physical_dim).to(DEVICE)
        self.critic = HighLevelCriticTCN(state_dim, physical_dim).to(DEVICE)
        self.target_critic = HighLevelCriticTCN(state_dim, physical_dim).to(DEVICE)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # TCN 历史窗口
        self.seq_len = 12
        self.state_hist = deque(maxlen=self.seq_len)
        self.physics_hist = deque(maxlen=self.seq_len)

    def reset_hidden(self):
        self.state_hist.clear()
        self.physics_hist.clear()

   
    def get_t_set(self, state, r=None, c=None, q_max=None, t_max=None):
        self.state_hist.append(np.asarray(state, dtype=np.float32))

       
        if self.actor.physical_dim > 0:
            if r is None or c is None:
                r, c = 1e-3, 1e7
            self.physics_hist.append(_rc_to_feat_np(r, c))

        
        if len(self.state_hist) < self.seq_len:
            pad_s = [self.state_hist[0]] * (self.seq_len - len(self.state_hist))
            s_seq = pad_s + list(self.state_hist)

            p_seq = None
            if self.actor.physical_dim > 0:
                pad_p = [self.physics_hist[0]] * (self.seq_len - len(self.physics_hist))
                p_seq = pad_p + list(self.physics_hist)
        else:
            s_seq = list(self.state_hist)
            p_seq = list(self.physics_hist) if self.actor.physical_dim > 0 else None

        s_t = torch.from_numpy(np.stack(s_seq, axis=0)[None, :, :]).float().to(DEVICE)

        physics_t = None
        if self.actor.physical_dim > 0:
            physics_t = torch.from_numpy(np.stack(p_seq, axis=0)[None, :, :]).float().to(DEVICE)

        with torch.no_grad():
            t_set, _ = self.actor(s_t, physics=physics_t, hidden=None)

        return float(t_set.item())

    def optimize(self, memory, batch_size, step, im_warmup=300, im_min=0.1):
        samples = memory.sample_sequence(
            batch_size, self.seq_len,
            with_t_target=True,
            with_next_t_target=True,
            with_hl_alpha=True,
            with_next_physics=True
        )

        if samples is None:
            return 0.0

        (s_seq, a_seq, r_seq, s1_seq, done_seq,
         t_a_seq, t_rule_seq, t_target_seq, t_target_next_seq, alpha_hl_seq,
         phys_r_seq, phys_c_seq,
         phys_r_next_seq, phys_c_next_seq) = samples
        t_target_next_t = torch.FloatTensor(t_target_next_seq).to(DEVICE)  # (B,T,1)
        next_t_target = t_target_next_t[:, -1, :]  # (B,1)
        alpha_hl_t = torch.FloatTensor(alpha_hl_seq).to(DEVICE)
        curr_alpha_hl = alpha_hl_t[:, -1, :]

        s_t = torch.FloatTensor(s_seq).to(DEVICE)       # (B,T,Ds)
        s1_t = torch.FloatTensor(s1_seq).to(DEVICE)     # (B,T,Ds)
        r_t = torch.FloatTensor(r_seq).to(DEVICE)       # (B,T,1)
        done_t = torch.FloatTensor(done_seq).to(DEVICE) # (B,T,1)

        t_rule_t = torch.FloatTensor(t_rule_seq).to(DEVICE)     # (B,T,1)
        t_target_t = torch.FloatTensor(t_target_seq).to(DEVICE) # (B,T,1)

        # physics: (B,T,2)
        phys_r_t = torch.FloatTensor(phys_r_seq).to(DEVICE)  # (B,T,1) raw R
        phys_c_t = torch.FloatTensor(phys_c_seq).to(DEVICE)  # (B,T,1) raw C

       
        phys_r_feat, phys_c_feat = _rc_to_feat_torch(phys_r_t, phys_c_t)
        physics_t = torch.cat([phys_r_feat, phys_c_feat], dim=-1)  # (B,T,2)

        # next physics: (B,T,1) raw R/C at t+1 aligned with s1_t
        phys_r1_t = torch.FloatTensor(phys_r_next_seq).to(DEVICE)
        phys_c1_t = torch.FloatTensor(phys_c_next_seq).to(DEVICE)

        phys_r1_feat, phys_c1_feat = _rc_to_feat_torch(phys_r1_t, phys_c1_t)
        physics1_t = torch.cat([phys_r1_feat, phys_c1_feat], dim=-1)  # (B,T,2)

       
        curr_r = r_t[:, -1, :]
        curr_done = done_t[:, -1, :]
        curr_t_target = t_target_t[:, -1, :]
        curr_t_rule = t_rule_t[:, -1, :]

        # ---------------- Critic ----------------
        with torch.no_grad():
         
            q_next, _ = self.target_critic(s1_t, physics=physics1_t, t_set=next_t_target, hidden=None)
            target_q = curr_r + self.gamma * (1.0 - curr_done) * q_next

        current_q, _ = self.critic(s_t, physics=physics_t, t_set=curr_t_target, hidden=None)
        loss_critic = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()

        # ---------------- Actor ----------------
        pred_t_set, _ = self.actor(s_t, physics=physics_t, hidden=None)
        pred_t_set = torch.clamp(pred_t_set, 15.0, 30.0)
       
        loss_imitation = F.mse_loss(pred_t_set, curr_t_rule)

      
        alpha_det = curr_alpha_hl.detach()
        t_rule_det = curr_t_rule.detach()

        pred_t_fused = alpha_det * t_rule_det + (1.0 - alpha_det) * pred_t_set
        pred_t_fused = torch.clamp(pred_t_fused, 15.0, 30.0)

        q_pi, _ = self.critic(s_t, physics=physics_t, t_set=pred_t_fused, hidden=None)
        loss_rl = -torch.mean(q_pi)

        lambda_im = max(im_min, 1.0 - step / float(im_warmup))
        total_loss = lambda_im * loss_imitation + loss_rl

        self.actor_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()

        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

        return float(total_loss.item())

    def _soft_update(self, target, source):
        tau = self.tau
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
