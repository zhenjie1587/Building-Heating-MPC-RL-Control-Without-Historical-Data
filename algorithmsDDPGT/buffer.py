import numpy as np
import random


class MemoryBuffer:
    def __init__(self, size: int):
        self.buffer = []
        self.maxSize = int(size)
        self.current_size = 0
        self._ptr = 0  # 指向“最旧元素”（也是下一次覆盖写入的位置）

    def reset(self):
        self.buffer.clear()
        self.current_size = 0
        self._ptr = 0

    def __len__(self):
        return self.current_size

    # 逻辑序号(0最旧) -> 物理下标
    def _phys_index(self, logical_idx: int) -> int:
        if self.current_size < self.maxSize:
            return int(logical_idx)
        return int((self._ptr + logical_idx) % self.maxSize)

    def _get(self, logical_idx: int):
        return self.buffer[self._phys_index(logical_idx)]

    def add(self, s, a, r, s1, done, t_a, t_rule, t_target, phys_r, phys_c, alpha_ll, alpha_hl=1.0):
        # b[10] = alpha_ll（低层动作融合权重 alpha_final）
        # b[11] = alpha_hl（高层目标温度融合权重 alpha_hl）
        transition = (s, a, r, s1, done, t_a, t_rule, t_target, phys_r, phys_c, alpha_ll, alpha_hl)

        if self.current_size < self.maxSize:
            self.buffer.append(transition)
            self.current_size += 1
            if self.current_size == self.maxSize:
                self._ptr = 0
        else:

            self.buffer[self._ptr] = transition
            self._ptr = (self._ptr + 1) % self.maxSize

    def sample(self, count: int, with_t_target: bool = False):
        count = int(min(count, self.current_size))
        batch = random.sample(self.buffer, count)  # 这里随机采样没问题：buffer里每个格子都是有效transition
        return self._format_batch(batch, with_t_target)

    def sample_sequence(self, batch_size, seq_len,
                        with_t_target=False,
                        with_next_t_target=False,  
                        with_fusion=False,
                        with_hl_alpha=False,
                        with_next_physics=False):

        if self.current_size < seq_len + 1:
            return None

        batch_sequences = []
        max_start = self.current_size - (seq_len + 1)

        for _ in range(batch_size):
            found = False
            for _try in range(100):
                start_idx = random.randint(0, max_start)  # logical start
                seq_all = [self._get(start_idx + j) for j in range(seq_len + 1)]
                seq = seq_all[:-1]

                # 前 seq_len-1 步不能 done
                if all(float(tr[4]) < 0.5 for tr in seq[:-1]):
                    batch_sequences.append(seq_all)
                    found = True
                    break

            if not found:
                print(f"[WARN] sample_sequence failed: seq_len={seq_len}, buffer_size={self.current_size}")
                break

        actual_bs = len(batch_sequences)
        if actual_bs == 0:
            return None

        # 当前序列 (t)
        s_seq = np.float32([[b[0] for b in seq_all[:-1]] for seq_all in batch_sequences])
        a_seq = np.float32([[b[1] for b in seq_all[:-1]] for seq_all in batch_sequences])
        r_seq = np.float32([[b[2] for b in seq_all[:-1]] for seq_all in batch_sequences]).reshape(actual_bs, seq_len, 1)
        s1_seq = np.float32([[b[3] for b in seq_all[:-1]] for seq_all in batch_sequences])
        done_seq = np.float32([[b[4] for b in seq_all[:-1]] for seq_all in batch_sequences]).reshape(actual_bs, seq_len, 1)

        t_a_seq = np.float32([[b[5] for b in seq_all[:-1]] for seq_all in batch_sequences])
        t_rule_seq = np.float32([[b[6] for b in seq_all[:-1]] for seq_all in batch_sequences]).reshape(actual_bs, seq_len, 1)

        phys_r_seq = np.float32([[b[8] for b in seq_all[:-1]] for seq_all in batch_sequences]).reshape(actual_bs, seq_len, 1)
        phys_c_seq = np.float32([[b[9] for b in seq_all[:-1]] for seq_all in batch_sequences]).reshape(actual_bs, seq_len, 1)

        phys_r_next_seq = phys_c_next_seq = None
        if with_next_physics:
            phys_r_next_seq = np.float32([[b[8] for b in seq_all[1:]] for seq_all in batch_sequences]).reshape(actual_bs, seq_len, 1)
            phys_c_next_seq = np.float32([[b[9] for b in seq_all[1:]] for seq_all in batch_sequences]).reshape(actual_bs, seq_len, 1)

        if with_t_target:
            t_target_seq = np.float32([[b[7] for b in seq_all[:-1]] for seq_all in batch_sequences]).reshape(actual_bs, seq_len, 1)
            alpha_hl_seq = None
            if with_hl_alpha:
                # b[11] = alpha_hl
                alpha_hl_seq = np.float32([[b[11] for b in seq_all[:-1]] for seq_all in batch_sequences]).reshape(
                    actual_bs,
                    seq_len,
                    1)

            t_target_next_seq = None
            if with_next_t_target:
                # seq_all[1:] 和 s1_seq 对齐（每一步对应下一步）
                t_target_next_seq = np.float32([[b[7] for b in seq_all[1:]] for seq_all in batch_sequences]).reshape(
                    actual_bs, seq_len, 1)

            if with_fusion:
                t_a_next_seq = np.float32([[b[5] for b in seq_all[1:]] for seq_all in batch_sequences])
                alpha_seq = np.float32([[b[10] for b in seq_all[:-1]] for seq_all in batch_sequences]).reshape(actual_bs, seq_len, 1)
                alpha_next_seq = np.float32([[b[10] for b in seq_all[1:]] for seq_all in batch_sequences]).reshape(actual_bs, seq_len, 1)
                if with_next_t_target:
                    return (s_seq, a_seq, r_seq, s1_seq, done_seq,
                            t_a_seq, t_rule_seq, t_target_seq, t_target_next_seq,
                            phys_r_seq, phys_c_seq,
                            t_a_next_seq, alpha_seq, alpha_next_seq)

                return (s_seq, a_seq, r_seq, s1_seq, done_seq,
                        t_a_seq, t_rule_seq, t_target_seq,
                        phys_r_seq, phys_c_seq,
                        t_a_next_seq, alpha_seq, alpha_next_seq)

            if with_next_physics:
                if with_next_t_target:
                    return (s_seq, a_seq, r_seq, s1_seq, done_seq,
                            t_a_seq, t_rule_seq, t_target_seq, t_target_next_seq, alpha_hl_seq,
                            phys_r_seq, phys_c_seq,
                            phys_r_next_seq, phys_c_next_seq)

                return (s_seq, a_seq, r_seq, s1_seq, done_seq,
                        t_a_seq, t_rule_seq, t_target_seq, phys_r_seq, phys_c_seq,
                        phys_r_next_seq, phys_c_next_seq)

            return (s_seq, a_seq, r_seq, s1_seq, done_seq,
                    t_a_seq, t_rule_seq, t_target_seq, phys_r_seq, phys_c_seq)

        return (s_seq, a_seq, r_seq, s1_seq, done_seq,
                t_a_seq, t_rule_seq, phys_r_seq, phys_c_seq)

    def _format_batch(self, batch, with_t_target):
        s_arr = np.float32([b[0] for b in batch])
        a_arr = np.float32([b[1] for b in batch])
        r_arr = np.float32([b[2] for b in batch]).reshape(-1, 1)
        s1_arr = np.float32([b[3] for b in batch])
        done_arr = np.float32([b[4] for b in batch]).reshape(-1, 1)
        t_a_arr = np.float32([b[5] for b in batch])
        t_rule_arr = np.float32([b[6] for b in batch]).reshape(-1, 1)
        phys_r_arr = np.float32([b[8] for b in batch]).reshape(-1, 1)
        phys_c_arr = np.float32([b[9] for b in batch]).reshape(-1, 1)

        if with_t_target:
            t_target_arr = np.float32([b[7] for b in batch]).reshape(-1, 1)
            return (s_arr, a_arr, r_arr, s1_arr, done_arr,
                    t_a_arr, t_rule_arr, t_target_arr, phys_r_arr, phys_c_arr)

        return (s_arr, a_arr, r_arr, s1_arr, done_arr,
                t_a_arr, t_rule_arr, phys_r_arr, phys_c_arr)
