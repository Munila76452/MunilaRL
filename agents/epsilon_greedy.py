import numpy as np
class EpsilonGreedyAgent:
    def __init__(self,n_arms,epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_value = np.zeros(n_arms)
        self.action_counts = np.zeros(n_arms)
        
    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.q_value)

    def update(self,action,reward):
        self.action_counts[action] += 1
        step_size = 1/self.action_counts[action]
        current_estimate = self.q_value[action]
        self.q_value[action] = current_estimate + step_size * (reward - current_estimate)
    