import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(reward_history, window_size=50, title="Agent Learning Curve"):
    rewards = np.array(reward_history)
    weights = np.ones(window_size) / window_size
    moving_avg = np.convolve(rewards, weights, mode='valid')
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.2, color='gray', label='Raw Reward')
    plt.plot(np.arange(window_size - 1, len(rewards)), moving_avg, 
             color='blue', linewidth=2, label=f'Moving Average (Window={window_size})')

    plt.title(title, fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
    
    