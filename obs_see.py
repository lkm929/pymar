#動作 0: 無操作（no-op，當代理已死亡時使用）
#動作 1: 停止或保持位置
#動作 2-5: 朝四個方向移動（北、南、東、西）
#動作 6-8: 攻擊 1 至 3 個敵方單位
#3m地圖的n_actions: 9
import torch
import numpy as np
from smac.env import StarCraft2Env  

def test_obs_format():
    # 初始化環境
    env = StarCraft2Env(map_name="3m")  # 替換為所用的地圖
    env_info = env.get_env_info()
    print(f"Environment info: {env_info}")
    
    # 獲取觀察值
    env.reset()
    obs = env.get_obs()
    print(f"Original obs (list format): {obs}")

    # 檢查是否為列表，然後轉換為張量
    if isinstance(obs, list):
        try:
            obs_tensor = torch.tensor(np.array(obs), dtype=torch.float32)
            print(f"Converted obs to tensor with shape: {obs_tensor.shape}")
        except Exception as e:

            print(f"Failed to convert obs to tensor: {e}")

if __name__ == "__main__":
    test_obs_format()