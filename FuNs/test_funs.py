import torch
import torch.nn as nn
import torch.nn.functional as F

class PerceptualModule(nn.Module):
    """
    感知模組: 將每個代理的觀察向量轉換為高層特徵表示。
    """
    def __init__(self, obs_dim, feature_dim):
        super(PerceptualModule, self).__init__()
        self.fc = nn.Linear(obs_dim, feature_dim)
    
    def forward(self, obs):
        """
        obs: [batch_size, n_agents, obs_dim]
        """
        z = F.relu(self.fc(obs))  # 特徵表示
        return z  # [batch_size, n_agents, feature_dim]

class Manager(nn.Module):
    """
    管理者: 為每個代理生成方向性目標。
    """
    def __init__(self, feature_dim, goal_dim, hidden_dim):
        super(Manager, self).__init__()
        self.rnn = nn.GRU(feature_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, goal_dim)
    
    def forward(self, features, hidden_state):
        """
        features: [batch_size, n_agents, feature_dim]
        hidden_state: 管理者的隱藏狀態
        """
        batch_size, n_agents, _ = features.shape
        features = features.view(batch_size * n_agents, -1, features.size(-1))  # 合併代理和序列
        rnn_out, hidden_state = self.rnn(features, hidden_state)
        g_raw = self.fc(rnn_out)  # 未歸一化的目標
        g = F.normalize(g_raw, p=2, dim=-1)  # 歸一化目標方向
        g = g.view(batch_size, n_agents, -1)  # 恢復到每個代理的形狀
        return g, hidden_state

class Worker(nn.Module):
    """
    工作者: 基於目標生成具體行動。
    """
    def __init__(self, feature_dim, goal_dim, action_dim, hidden_dim):
        super(Worker, self).__init__()
        self.rnn = nn.GRU(feature_dim + goal_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, features, goals, hidden_state):
        """
        features: [batch_size, n_agents, feature_dim]
        goals: [batch_size, n_agents, goal_dim]
        hidden_state: 工作者的隱藏狀態
        """
        x = torch.cat([features, goals], dim=-1)  # 合併特徵與目標
        batch_size, n_agents, _ = x.shape
        x = x.view(batch_size * n_agents, -1, x.size(-1))
        rnn_out, hidden_state = self.rnn(x, hidden_state)
        action_logits = self.fc(rnn_out)
        action_logits = action_logits.view(batch_size, n_agents, -1)  # 恢復到每個代理的形狀
        return action_logits, hidden_state

class FeUdalNetwork(nn.Module):
    """
    FeUdal Network: 適用於SC2的多智能體架構。
    """
    def __init__(self, obs_dim, feature_dim, goal_dim, action_dim, manager_hidden_dim, worker_hidden_dim, n_agents):
        super(FeUdalNetwork, self).__init__()
        self.n_agents = n_agents
        self.perceptual_module = PerceptualModule(obs_dim, feature_dim)
        self.manager = Manager(feature_dim, goal_dim, manager_hidden_dim)
        self.worker = Worker(feature_dim, goal_dim, action_dim, worker_hidden_dim)
    
    def forward(self, obs, manager_hidden, worker_hidden):
        """
        obs: [batch_size, n_agents, obs_dim]
        manager_hidden: 管理者的隱藏狀態
        worker_hidden: 工作者的隱藏狀態
        """
        # 提取特徵
        features = self.perceptual_module(obs)
        
        # 管理者生成目標
        goals, manager_hidden = self.manager(features, manager_hidden)
        
        # 工作者生成行動
        action_logits, worker_hidden = self.worker(features, goals, worker_hidden)
        
        return action_logits, goals, manager_hidden, worker_hidden

# 測試架構
if __name__ == "__main__":
    # 假設輸入: 多智能體環境
    batch_size = 32
    n_agents = 3
    obs_dim = 32  # 每個代理的觀察維度
    feature_dim = 64
    goal_dim = 16
    action_dim = 8
    manager_hidden_dim = 128
    worker_hidden_dim = 128

    obs = torch.randn(batch_size, n_agents, obs_dim)

    # 初始化模型
    model = FeUdalNetwork(obs_dim, feature_dim, goal_dim, action_dim,
                          manager_hidden_dim, worker_hidden_dim, n_agents)

    # 初始化隱藏狀態
    manager_hidden = torch.zeros(1, batch_size * n_agents, manager_hidden_dim)  # GRU 隱藏狀態
    worker_hidden = torch.zeros(1, batch_size * n_agents, worker_hidden_dim)  # GRU 隱藏狀態

    # 前向傳播
    action_logits, goals, manager_hidden, worker_hidden = model(obs, manager_hidden, worker_hidden)

    print("行動 logits:", action_logits.shape)  # [batch_size, n_agents, action_dim]
    print("目標方向:", goals.shape)  # [batch_size, n_agents, goal_dim]
