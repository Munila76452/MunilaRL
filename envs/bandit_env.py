import numpy as np
class BanditEnv:
    def __init__ (self,n_arms=10,variance=1.0):
        self.n_arms = n_arms
        self.vaiance = variance
        self.true_mean = np.random.normal(loc=0.0,scale=1,size=n_arms)
        
    def reset(self):
        '''
        in this there is no state , so on reset 
        '''
        return 0,{}
    
    def step(self,action):
        actual_pay_out = self.true_mean[action]
        reward = np.random.normal(loc=actual_pay_out,scale=self.vaiance)
        observation = 0
        terminated = False
        truncated = False
        info = {'true_mean':actual_pay_out}
        return observation, reward, terminated, truncated, info