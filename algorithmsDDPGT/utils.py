
import numpy as np
import torch
# -------------------------
# Target 网络更新
# -------------------------
def hard_update(target, source):
    """Copy network parameters from source to target."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, tau: float):
    """Soft update target network parameters."""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


# -------------------------
# OU Noise (DDPG探索噪声)
# -------------------------
class OUNoise:
    """
    Ornstein-Uhlenbeck process noise.
    Common for DDPG in continuous control.

    action_dimension: int
    mu: mean
    theta: reversion speed
    sigma: volatility
    dt: timestep
    """

    def __init__(self, action_dimension, mu=0.0, theta=0.15, sigma=0.05, dt=1.0, x0=None):
        self.action_dimension = int(action_dimension)
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        if self.x0 is None:
            self.state = np.ones(self.action_dimension, dtype=np.float32) * self.mu
        else:
            self.state = np.array(self.x0, dtype=np.float32).copy()

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dimension)
        self.state = (x + dx).astype(np.float32)
        return self.state


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.05):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X
