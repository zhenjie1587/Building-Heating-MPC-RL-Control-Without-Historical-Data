import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

EPS = 0.003


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


# ==========================================================
# TCN building blocks (causal dilated 1D conv)
# ==========================================================

class Chomp1d(nn.Module):
    """移除右侧 padding，保证因果性（只看过去，不偷看未来）"""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        # x: (B, C, T)
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.out_relu = nn.ReLU()

    def forward(self, x):
        # x: (B, C, T)
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.out_relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_inputs: int,
        num_channels,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        in_ch = num_inputs
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C, T)
        return self.network(x)



class SetpointActorTCN(nn.Module):
    """输入历史 (B,T,Ds)+physics -> 输出 t_set（取最后一步）"""
    def __init__(
        self,
        state_dim: int,
        physical_dim: int = 2,
        channels=(128, 128, 128),
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.physical_dim = int(physical_dim)
        in_dim = int(state_dim) + self.physical_dim

        self.tcn = TemporalConvNet(
            num_inputs=in_dim,
            num_channels=list(channels),
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(channels[-1], 1)

    def forward(self, state, physics=None, hidden=None):
        if state.dim() == 2:
            state = state.unsqueeze(1)

        if self.physical_dim > 0:
            if physics is None:
                raise ValueError("physics is required when physical_dim > 0")
            if physics.dim() == 2:
                physics = physics.unsqueeze(1)
            if physics.size(1) == 1 and state.size(1) > 1:
                physics = physics.repeat(1, state.size(1), 1)
            x = torch.cat([state, physics], dim=-1)  # (B,T,D)
        else:
            x = state

        x = x.transpose(1, 2)  # (B,D,T)
        h = self.tcn(x)        # (B,Ch,T)
        h_last = h[:, :, -1]   # (B,Ch)
        t_raw = torch.sigmoid(self.fc_out(h_last))  # 约束到 [0, 1]
        t_set = t_raw * (30.0 - 15.0) + 15.0  # 映射到物理区间
        return t_set, None


class HighLevelCriticTCN(nn.Module):
    """TCN 编码 state(+physics) 序列，再与 t_set 合并输出 Q"""
    def __init__(
        self,
        state_dim: int,
        physical_dim: int = 2,
        channels=(128, 128, 128),
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.physical_dim = int(physical_dim)
        in_dim = int(state_dim) + self.physical_dim

        self.tcn = TemporalConvNet(
            num_inputs=in_dim,
            num_channels=list(channels),
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.fc_t = nn.Linear(1, 128)
        self.fc2 = nn.Linear(channels[-1] + 128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, physics=None, t_set=None, hidden=None):
        if t_set is None:
            raise ValueError("t_set is required")

        if state.dim() == 2:
            state = state.unsqueeze(1)

        if self.physical_dim > 0:
            if physics is None:
                raise ValueError("physics is required when physical_dim > 0")
            if physics.dim() == 2:
                physics = physics.unsqueeze(1)
            if physics.size(1) == 1 and state.size(1) > 1:
                physics = physics.repeat(1, state.size(1), 1)
            x = torch.cat([state, physics], dim=-1)
        else:
            x = state

        x = x.transpose(1, 2)  # (B,D,T)
        h = self.tcn(x)
        h_last = h[:, :, -1]   # (B,Ch)

        if t_set.dim() == 3:
            t_set = t_set[:, -1, :]
        t_embed = F.relu(self.fc_t(t_set))

        z = torch.cat([h_last, t_embed], dim=-1)
        z = F.relu(self.fc2(z))
        q = self.fc3(z)
        return q, None




class ActorTCN(nn.Module):
    """输入历史 (B,T,Ds) -> 输出 action/temp（取最后一步）"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_lim: float,
        channels=(256, 256, 256),
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_lim = action_lim

        self.tcn = TemporalConvNet(
            num_inputs=int(state_dim),
            num_channels=list(channels),
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.fc2 = nn.Linear(channels[-1], 128)
        self.out_action = nn.Linear(128, action_dim)
        self.out_temp = nn.Linear(128 + action_dim, 1)


        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.out_action.weight.data.uniform_(-EPS, EPS)
        self.out_temp.weight.data.uniform_(-EPS, EPS)

    def forward(self, state, hidden=None):
        if state.dim() == 2:
            state = state.unsqueeze(1)

        x = state.transpose(1, 2)  # (B,Ds,T)
        h = self.tcn(x)            # (B,Ch,T)
        h_last = h[:, :, -1]       # (B,Ch)

        y = F.relu(self.fc2(h_last))

        # 1) 动作仍然是 0~action_lim
        action = torch.sigmoid(self.out_action(y)) * self.action_lim  # (B, action_dim)

        # 2) temp head 预测 next Tin 的归一化值 [-1, 1]
        temp_in = torch.cat([y, action], dim=-1)  # (B, 128+action_dim)
        tin_next_norm = torch.tanh(self.out_temp(temp_in))  # (B, 1) in [-1,1]

        return action, tin_next_norm, None


class CriticTCN(nn.Module):
    """TCN 编码 state 序列，再和当前 action 合并输出 Q"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        channels=(256, 256, 256),
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.tcn = TemporalConvNet(
            num_inputs=int(state_dim),
            num_channels=list(channels),
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.fca1 = nn.Linear(action_dim, 128)
        self.fc2 = nn.Linear(channels[-1] + 128, 128)
        self.fc3 = nn.Linear(128, 1)

        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

    def forward(self, state, action, hidden=None):
        if state.dim() == 2:
            state = state.unsqueeze(1)

        s = state.transpose(1, 2)  # (B,Ds,T)
        h = self.tcn(s)            # (B,Ch,T)
        h_last = h[:, :, -1]       # (B,Ch)

        a = F.relu(self.fca1(action))
        x = torch.cat([h_last, a], dim=-1)
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q, None
