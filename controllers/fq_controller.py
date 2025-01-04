
import torch
import torch.nn.functional as F
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch.nn as nn


class FeUdalMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.device = args.device
        self.manager_input_shape = self._get_manager_input_shape(scheme)
        self.worker_input_shape = self._get_worker_input_shape(scheme)
        
        self._build_manager(self.manager_input_shape)
        self._build_worker(self.worker_input_shape)
        
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states_manager = None
        self.hidden_states_worker = None

    # def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
    #     # print("----ep_batch----",ep_batch)
    #     # 取得所有 agent 的可用動作
    #     avail_actions = ep_batch["avail_actions"][:, t_ep]
    #     # print("----可用動作----:",avail_actions)
    #     # Manager 根據狀態state產生目標goals
    #     state = ep_batch["state"][:, t_ep] # manager input
    #     # print("----狀態----:",state.shape)
    #     goals,self.hidden_states_manager = self.manager(state, self.hidden_states_manager, t_ep)
    #     # print("----目標----:",goals.shape)

    #     # Worker 根據Manager目標和觀察值obs選擇動作
    #     agent_outputs = self.worker_forward(ep_batch, t_ep, goals, test_mode=test_mode)
    #     print("----worker輸出----:",agent_outputs)
    #     chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
    #     return chosen_actions
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # 獲取可用動作和觀察值
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        state = ep_batch["state"][:, t_ep]  # Manager 的輸入
        obs = ep_batch["obs"][:, t_ep]  # Worker 的輸入

        # Manager 輸出目標
        goals, self.hidden_states_manager = self.manager(state, self.hidden_states_manager, t_ep)

        # Worker 輸出動作概率分佈
        agent_outputs, self.hidden_states_worker = self.worker(obs, self.hidden_states_worker, goals)

        # 動作選擇
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def worker_forward(self, ep_batch, t_ep, goals, test_mode=False):
        obs = ep_batch["obs"][:, t_ep] # worker input
        # agent_inputs = self._build_worker_inputs(ep_batch, t_ep, goals)
        print("----worker_agent_inputs----:",obs.shape)
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        print("----worker_avail_actions----:",avail_actions)
        print("----hidden_states_worker----",self.hidden_states_worker)
        print("----hidden_states_worker----",self.hidden_states_worker[0].shape)
        agent_outs, self.hidden_states_worker = self.worker(obs, self.hidden_states_worker, goals)

        # Softmax the agent outputs if they're policy logits
        if self.args.worker_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = torch.nn.functional.softmax(agent_outs, dim=-1)
            if not test_mode:
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                              + torch.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

                if getattr(self.args, "mask_before_softmax", True):
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        # 調整隱藏狀態的擴展
        # 取得 worker 的初始隱藏狀態
        h_0, c_0 = self.worker.init_hidden(batch_size,device=self.device)
        # print("----h_0----",h_0.shape
        #       ,"----c_0----",c_0.shape)
        # print((h_0,c_0))
        self.hidden_states_worker = (h_0,c_0)   # Adjust expansion
        # 取得 manager 的初始隱藏狀態
        h_1, c_1 = self.manager.init_hidden(device=self.device)  # Unpack tuple
        # print("----h_1----",h_1.shape
        #       ,"----c_1----",c_1.shape)
        # print("----h_1----",h_1
        #       ,"----c_1----",c_1)
        # print((h_1.expand(1,batch_size, self.args.manager_hidden_dim),
        #                             c_1.expand(1,batch_size, self.args.manager_hidden_dim)))
        # Expand hidden states to match batch size (1,1,256)
        self.hidden_states_manager = (h_1.expand(1,batch_size, self.args.manager_hidden_dim),
                                    c_1.expand(1,batch_size, self.args.manager_hidden_dim))
        

    def parameters(self):
        return list(self.manager.parameters()) + list(self.worker.parameters())

    def load_state(self, other_mac):
        self.manager.load_state_dict(other_mac.manager.state_dict())
        self.worker.load_state_dict(other_mac.worker.state_dict())

    def cuda(self):
        self.manager.cuda()
        self.worker.cuda()

    def save_models(self, path):
        torch.save(self.manager.state_dict(), "{}/manager.torch".format(path))
        torch.save(self.worker.state_dict(), "{}/worker.torch".format(path))

    def load_models(self, path):
        self.manager.load_state_dict(torch.load("{}/manager.torch".format(path), map_location=lambda storage, loc: storage))
        self.worker.load_state_dict(torch.load("{}/worker.torch".format(path), map_location=lambda storage, loc: storage))

    def _build_manager(self, input_shape): # 建立 manager
        self.manager = agent_REGISTRY["feudal_manager"](input_shape, self.args)

    def _build_worker(self, input_shape): # 建立 worker
        self.worker = agent_REGISTRY["feudal_worker"](input_shape, self.args)

    # def _build_worker_inputs(self, ep_batch, t_ep, goals): # 建立 worker 輸入
    #     # Worker 輸入包括觀察值obs和目標goal
    #     bs = ep_batch.batch_size
    #     obs = [ep_batch["obs"][:, t_ep]]  # 觀察值 input

    #     # print("----inputs(obs)----:",inputs)
    #     # inputs.append(goals.expand(-1, self.n_agents, -1))  # 目標
    #     # print("----inputs(obs+goal)----:",inputs)

    #     if self.args.obs_last_action:
    #         if t_ep == 0:
    #             obs.append(torch.zeros_like(ep_batch["actions_onehot"][:, t]))
    #         else:
    #             obs.append(ep_batch["actions_onehot"][:, t_ep-1])
    #     if self.args.obs_agent_id:
    #         obs.append(torch.eye(self.n_agents, device=ep_batch.device).unsqueeze(0).expand(bs, -1, -1))

    #     inputs = torch.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
    #     print("----worker輸入----:",inputs)
    #     return inputs

    def _get_manager_input_shape(self, scheme): # state_shape=48
        return scheme["state"]["vshape"]

    def _get_worker_input_shape(self, scheme): # obs_shape
        return scheme["obs"]["vshape"]
        # input_shape = scheme["obs"]["vshape"] + self.args.goal_dim
        # if self.args.obs_last_action:
        #     input_shape += scheme["actions_onehot"]["vshape"][0]
        # if self.args.obs_agent_id:
        #     input_shape += self.n_agents
        # return input_shape
