import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.bandit_env import BanditEnv
env = BanditEnv(n_arms=5,variance=0.1)
obs , info = env.reset()
print('hidde true mean',env.true_mean)
for i in range(10):
    obs, reward, term, trunc, info = env.step(action=2)
    print(f"Pull {i+1} | Action: 2 | Reward Received: {reward:.2f}")
