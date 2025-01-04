import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeUdalWorker(nn.Module):
    def __init__(self, input_shape, args):
        super(FeUdalWorker, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.batch_size = args.batch_size
        self.n_actions = args.n_actions
        self.fc1 = nn.Linear(args.manager_hidden_dim, args.worker_hidden_dim, bias=False)  # 線性變換目標
        self.lstm = nn.LSTM(input_shape, args.worker_hidden_dim, batch_first=True)  # LSTM層
        self.fc2 = nn.Linear(args.n_agents, args.n_actions)  # 動作 logits
        self.fc3 = nn.Linear(args.n_agents*args.batch_size, args.n_actions)  # 動作 logits
    def forward(self, obs, hidden_states_worker, goals):
        batch_size = obs.shape[0]  # 從 obs 提取當前 batch_size
        n_agents = goals.shape[1]  # 提取智能體數量
        
        u_t, new_hidden = self.lstm(obs, hidden_states_worker)  # LSTM 前向傳遞
        # print("----u_t----",u_t.shape)
        reshaped_ut = u_t.reshape(-1, 256)

        # print("----new_hidden----",new_hidden[0].shape)
        # print("----goals----",goals.shape)
        # 處理 goals # goals.reshape(-1, goals.shape[-1])
        w_t = self.fc1(goals)  # 轉換 goals
        reshaped_wt = w_t.reshape(-1, 256)
        # print("----w_t----",w_t.shape) # u_t.squeeze(1), w_t.T
        logits = torch.matmul(reshaped_ut, reshaped_wt.T)  # 計算 logits
        # print("----logits----",logits.shape)
        if logits.shape[1] == self.n_agents*self.batch_size:
            # pi_t = torch.softmax(self.fc3(logits), dim=-1)
            pi_t = self.fc3(logits)
        else:
            # pi_t = torch.softmax(self.fc2(logits), dim=-1)  # 計算動作概率分佈
            pi_t = self.fc2(logits)
        # print("----pi_t----",pi_t.view(batch_size, n_agents, -1))
        return pi_t.view(batch_size, n_agents, -1), new_hidden
        # return logits.view(batch_size, n_agents, -1), new_hidden
       


    def init_hidden(self, batch_size ,device='cpu'):
        return (torch.zeros(1, batch_size, self.args.worker_hidden_dim, device=device),
                torch.zeros(1, batch_size, self.args.worker_hidden_dim, device=device))
