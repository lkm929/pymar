# In modules/agents/feudal_manager.py
# manager_agent架構完成
import torch
import torch.nn as nn
import torch.nn.functional as F
# 12/13
class FeUdalManager(nn.Module):
    def __init__(self, input_shape, args, dilation_rate=3):
        super(FeUdalManager, self).__init__()
        self.args = args
        self.n_agents = args.n_agents  # 智能體數量
        self.lstm = nn.LSTM(input_shape, args.manager_hidden_dim, batch_first=True)  # LSTM層
        self.dilation_rate = dilation_rate  # 稀疏更新

        # 添加屬性來保存上一次的 goal 和 hidden_states
        self.last_goal = None
        self.last_hidden = None

    def forward(self, state, hidden_states_manager, time_step):
        """
        Args:
            state (torch.Tensor): 當前時間步的輸入 (batch_size, input_dim) ，即state。
            hidden_states_manager (tuple): 隱藏狀態 (h_t, c_t) 。
            time_step (int): 當前時間步，用於稀疏更新。
        Returns:
            goals (torch.Tensor): 每個智能體的目標 (batch_size, n_agents, hidden_dim)。
            new_hidden (tuple): 更新後的隱藏狀態。
        """
        hidden_states_manager = tuple([h.contiguous() for h in hidden_states_manager]) # ?
        # print("----time_step----",time_step)
        if time_step % self.dilation_rate == 0:
            state = state.unsqueeze(1)  # 添加時間步維度，形狀變為 (batch_size, seq_len=1, input_dim)
            goal, new_hidden = self.lstm(state, hidden_states_manager)  # LSTM輸出
            goal = goal.repeat(1, self.n_agents, 1)  # 每個智能體共享目標

            # 保存當前的 goal 和 hidden states
            self.last_goal = goal
            self.last_hidden = new_hidden
        else:
            # 返回上一次的 goal 和 hidden states
            goal = self.last_goal
            new_hidden = self.last_hidden
        return goal, new_hidden

    def init_hidden(self, device='cpu'):
        return (torch.zeros(1, 1, self.args.manager_hidden_dim, device=device),
                torch.zeros(1, 1, self.args.manager_hidden_dim, device=device))
