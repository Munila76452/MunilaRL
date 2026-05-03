import numpy as np
class QLearning:
    def __init__(self,num_state,num_action,alpha=0.1,epsilon=1,epsilon_decay=0.995,epsilon_min=0.01,gamma=0.99):
        self.num_state = num_state
        self.num_action = num_action
        self.gamma = gamma
        self.alpha = alpha
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsion_min = epsilon_min
        
        self.q_table = np.zeros((num_state,num_action))
        
    def choose_action(self,state):
        if np.random.uniform(0,1) < self.epsilon:
            # fully random
            return np.random.choice(self.num_action)
        else:
            return np.argmax(self.q_table[state,:])
    
    def update(self,state,action,reward,next_state):
        current_q = self.q_table[state,action]
        max_next_q = np.max(self.q_table[next_state,:])
        target = reward + self.gamma * max_next_q
        self.q_table[state,action] = current_q + self.alpha * (target - current_q)
        
    def decay_epsilon(self):
        self.epsilon = max(self.epsion_min,self.epsilon*self.epsilon_decay)
        
    def reset_table(self):
        self.q_table = np.zeros((self.num_state,self.num_action))