from envs.bandit_env import BanditEnv
from utils.plotting import plot_learning_curve
from agents.epsilon_greedy import EpsilonGreedyAgent
import numpy as np
def main():
    # 1. Set Hyperparameters
    N_ARMS = 10
    VARIANCE = 1.0
    EPSILON = 0.1
    TOTAL_STEPS = 1000
    WINDOW_SIZE = 50  # For smoothing the plot
    
    # 2. Initialize Environment and Agent
    print(f"Initializing {N_ARMS}-Armed Bandit Environment...")
    env = BanditEnv(n_arms=N_ARMS, variance=VARIANCE)
    agent = EpsilonGreedyAgent(n_arms=N_ARMS, epsilon=EPSILON)
    
    # 3. Setup Tracking Variables
    obs, info = env.reset()
    reward_history = []
    total_reward = 0
    
    print(f"Starting Training for {TOTAL_STEPS} steps...")
    
    # 4. The Main Training Loop
    for step in range(TOTAL_STEPS):
        # A. Agent selects an action based on its current knowledge
        action = agent.select_action()
        
        # B. Environment processes the action and returns a noisy reward
        obs, reward, terminated, truncated, info = env.step(action)
        
        # C. Agent updates its internal estimates using the reward
        agent.update(action, reward)
        
        # D. Log the reward for plotting later
        reward_history.append(reward)
        total_reward += reward

    # 5. Print Final Statistics
    print("\n=== Training Complete ===")
    print(f"Total Reward Gathered: {total_reward:.2f}")
    
    print("\n--- Final Estimations vs Reality ---")
    for i in range(N_ARMS):
        true_val = env.true_mean[i]
        est_val = agent.q_value[i]
        pulls = int(agent.action_counts[i])
        
        # Format the output so the columns align nicely
        print(f"Arm {i:2}: True = {true_val:>5.2f} | Est = {est_val:>5.2f} | Pulls = {pulls:>4}")

    # Determine if the agent actually found the best arm
    best_actual_arm = np.argmax(env.true_mean)
    best_guessed_arm = np.argmax(agent.q_value)
    
    print(f"\nBest Actual Arm : {best_actual_arm} (Value: {env.true_mean[best_actual_arm]:.2f})")
    print(f"Agent's Top Pick: {best_guessed_arm} (Guessed Value: {agent.q_value[best_guessed_arm]:.2f})")
    
    if best_actual_arm == best_guessed_arm:
        print("Success! The agent successfully identified the optimal machine.")
    else:
        print("Failure. The agent got stuck exploiting the wrong machine. Try a higher epsilon or more steps!")

    # 6. Visualize the Learning Curve
    print("\nGenerating Learning Curve Plot...")
    plot_learning_curve(
        reward_history, 
        window_size=WINDOW_SIZE, 
        title=f"Epsilon-Greedy (ε={EPSILON}) Learning Curve"
    )

if __name__ == "__main__":
    main()