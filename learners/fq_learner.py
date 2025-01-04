import copy
import torch
from torch.optim import RMSprop
import numpy as np
from components.episode_buffer import EpisodeBatch


class FeUdalQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac  # 包括 Manager 和 Worker
        self.logger = logger
        self.params = list(mac.manager.parameters()) + list(mac.worker.parameters())

        self.last_target_update_episode = 0

        # Mixer (optional, e.g., VDN or QMIX)
        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                from modules.mixers.vdn import VDNMixer
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                from modules.mixers.qmix import QMixer
                self.mixer = QMixer(args)
            else:
                raise ValueError(f"Mixer {args.mixer} not recognised.")
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        # Optimizer
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # Target networks
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 取得資料
        # print("----batch----",batch)
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # 初始化隱藏狀態
        self.mac.init_hidden(batch.batch_size)

        # Step 1: Manager生成目標
        manager_goals = []
        for t in range(batch.max_seq_length - 1):
            state = batch["state"][:, t]
            # print("----state----",state.shape,state)
            goal, _ = self.mac.manager(state, self.mac.hidden_states_manager, t)  # Manager生成目標
            
            # print("----goal.shape----",goal.shape)
            # print("----goal----",goal)
            manager_goals.append(goal)
        manager_goals = torch.stack(manager_goals, dim=1)  # shape: [batch_size, seq_length, goal_dim]
        # print("----manager_goals----",manager_goals.shape,manager_goals)
        
        
        # for t in range(batch.max_seq_length - 1):
        #     obs = batch["obs"][:, t]
        #     goal = manager_goals[:, t]
        #     q_values_t = []
        #     # print("----goal----",goal.shape,goal)
        #     # print("----obs----",obs.shape,obs)
        #     for i in range(obs.shape[0]):
        #         single_obs = obs[i].unsqueeze(0)
        #         single_goal = goal[i].unsqueeze(0)
        #         # print("----hidden_states_worker----",hidden_states_worker)
        #         single_hidden_states_worker = (self.mac.hidden_states_worker[0][:, i, :].unsqueeze(0), self.mac.hidden_states_worker[1][:, i, :].unsqueeze(0)) 
        #         q_values, _ = self.mac.worker(single_obs, single_hidden_states_worker, single_goal)  # Worker使用目標計算Q值 obs, hidden_states_worker, goals
        #         # q_values, _ = self.mac.worker_forward(single_obs, single_hidden_states_worker, single_goal)
        #         # print("----q_values----",q_values.shape,q_values)
        #         q_values_t.append(q_values)
        #     q_values_t = torch.cat(q_values_t, dim=0)
        #     worker_q_values.append(q_values_t)
        # worker_q_values = torch.stack(worker_q_values, dim=1)  # shape: [batch_size, seq_length, n_agents, n_actions]
        # # print("----worker_q_values----",worker_q_values.shape,worker_q_values)
        # # print("----batch.max_seq_length----",batch.max_seq_length)
        # # 挑選動作的Q值
        # chosen_action_qvals = torch.gather(worker_q_values, dim=3, index=actions).squeeze(3)
        # # print("----chosen_action_qvals----",chosen_action_qvals.shape,chosen_action_qvals)
        # Step 2: Worker計算動作Q值
        worker_q_values = []
        for t in range(batch.max_seq_length - 1):
            obs = batch["obs"][:, t]
            goal = manager_goals[:, t]

            # print("----obs----",obs.shape,obs)
            # print("----goal----",goal.shape,goal)
            # print("----hidden_states_worker----",self.mac.hidden_states_worker[0].shape,self.mac.hidden_states_worker[1].shape)
            # single_hidden_states_worker = (self.mac.hidden_states_worker[0][:, i, :].unsqueeze(0), self.mac.hidden_states_worker[1][:, i, :].unsqueeze(0)) 
            q_values, _ = self.mac.worker(obs, self.mac.hidden_states_worker, goal)  # Worker使用目標計算Q值 obs, hidden_states_worker, goals

            worker_q_values.append(q_values)
        worker_q_values = torch.stack(worker_q_values, dim=1)
        # print("----worker_q_values----",worker_q_values.shape,worker_q_values)  # shape: [batch_size, seq_length, n_agents, n_actions]
        chosen_action_qvals = torch.gather(worker_q_values, dim=3, index=actions).squeeze(3)
        # print("----chosen_action_qvals----",chosen_action_qvals.shape,chosen_action_qvals)
        # Step 3: 目标网络的Q值(manager)###################
        target_worker_q_values = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            obs = batch["obs"][:, t]
            goal = manager_goals[:, t]

            
            q_values, _ = self.mac.worker(obs, self.mac.hidden_states_worker, goal) 
            # for i in range(obs.shape[0]):
            #     single_obs = obs[i].unsqueeze(0)
            #     single_goal = goal[i].unsqueeze(0)
            #     # print("----hidden_states_worker----",hidden_states_worker)
            #     single_hidden_states_worker = (self.mac.hidden_states_worker[0][:, i, :].unsqueeze(0), self.mac.hidden_states_worker[1][:, i, :].unsqueeze(0)) 
            #     q_values, _ = self.target_mac.worker(single_obs, single_hidden_states_worker, single_goal)  # Worker使用目標計算Q值 obs, hidden_states_worker, goals
            #     # print("----q_values----",q_values.shape,q_values)
            #     q_values_t.append(q_values)
            # q_values_t = torch.cat(q_values_t, dim=0)
            target_worker_q_values.append(q_values)
        target_worker_q_values = torch.stack(target_worker_q_values, dim=1)  # shape: [batch_size, seq_length, n_agents, n_actions]
        # print("----target_worker_q_values----",target_worker_q_values.shape,target_worker_q_values)  # shape: [batch_size, seq_length, n_agents, n_actions]
        # 计算目标Q值
        if self.args.double_q:
            mac_out_detach = worker_q_values.clone().detach()
            mac_out_detach[avail_actions[:, :-1] == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(target_worker_q_values, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_worker_q_values.max(dim=3)[0]

        # print("----chosen_action_qvals----",chosen_action_qvals.shape,chosen_action_qvals)  # shape: [batch_size, seq_length, n_agents]
        # print("----target_max_qvals----",target_max_qvals.shape,target_max_qvals)  # shape: [batch_size, seq_length, n_agents]
        # Mixer（可选）
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])
        # print("----chosen_action_qvals----",chosen_action_qvals.shape,chosen_action_qvals)  # shape: [batch_size, seq_length, n_agents]
        # print("----target_max_qvals----",target_max_qvals.shape,target_max_qvals)  # shape: [batch_size, seq_length, n_agents]

        # TD目标
        # print("----self.logger----",self.logger.ep_length_mean)
        # if self.logger.ep_length_mean > 50:
        #     rewards -= 1
        # print("----rewards----",rewards.shape,rewards[1])
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        # print("----targets----",targets.shape,targets[1])
        # TD误差和损失
        td_error = (chosen_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # 优化
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        # 更新目标网络
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        # 日志记录
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), f"{path}/mixer.torch")
        torch.save(self.optimiser.state_dict(), f"{path}/opt.torch")

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(torch.load(f"{path}/mixer.torch", map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(torch.load(f"{path}/opt.torch", map_location=lambda storage, loc: storage))
