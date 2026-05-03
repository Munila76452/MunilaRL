import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.grid_world import SimpleGridWorld
from agents.Q_learning import QLearning

env = SimpleGridWorld(grid_size=4)
agents = QLearning(
    num_state=env.num_state,
    num_action=env.num_action,
    alpha=0.1,
    epsilon=1.0,
    epsilon_decay=0.99,
    epsilon_min=0.1,
    gamma=0.9
)

total_episode = 500
print('started training')

for episode in range(total_episode):
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = agents.choose_action(state)
        next_state,reward,done = env.step(action)
        agents.update(state,action,reward,next_state)
        episode_reward += reward
        state = next_state
    agents.decay_epsilon()
    if (episode + 1) % 100 == 0:
        print(f"Episode: {episode + 1} | Last Reward: {episode_reward} | Epsilon: {agents.epsilon:.3f}")

print('training completed')

# evaluation 
print("\nTesting the trained agent (Greedy Policy):")
agents.epsilon = 0.0 
state = env.reset()
done = False
steps_taken = 0
path = [env.agent_pos.copy()]

while not done and steps_taken < 20: 
    action = agents.choose_action(state) 
    state, reward, done = env.step(action)
    path.append(env.agent_pos.copy())
    steps_taken += 1

print(f"Path taken by agent: {path}")
if path[-1] == list(env.goal_pos):
    print("SUCCESS: Agent reached the goal!")
    
    print(f"Total Steps: {steps_taken}") 
else:
    print("FAILED: Agent did not reach the goal.")
