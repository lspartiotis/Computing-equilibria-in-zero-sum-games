import numpy as np
import matplotlib.pyplot as plt

class BattleOfTheSexesEnv:
    def __init__(self):
        self.actions = [0, 1]  # 0: Opera, 1: Football
        # Rewards based on the action combinations: [Opera, Opera], [Opera, Football], [Football, Opera], [Football, Football]
        self.rewards = np.array([[3, 1], [0, 2], [0, 2], [1, 3]])
        
    def step(self, action1, action2):
        # Map actions to the rewards matrix index
        index = action1 * 2 + action2
        return self.rewards[index]

class MaxMinQLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, num_actions=2):
        self.Q = np.zeros((1, num_actions))  # Only one state
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            # Max-Min strategy: Choose action maximizing the minimum value
            min_q_values = np.min(self.Q, axis=0)
            return np.argmax(min_q_values)
    
    def update(self, action, reward):
        # Max-Min Q-Learning update
        q_predict = self.Q[0, action]
        q_target = reward  # Simplified, as we don't have next state
        self.Q[0, action] += self.alpha * (q_target - q_predict)

class SARSAAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, num_actions=2):
        self.Q = np.zeros((1, num_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = num_actions

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.Q[0, :])
    
    def update(self, action, reward, next_action):
        q_predict = self.Q[0, action]
        q_target = reward + self.gamma * self.Q[0, next_action]
        self.Q[0, action] += self.alpha * (q_target - q_predict)

def simulate_battle_of_the_sexes(episodes=1000):
    env = BattleOfTheSexesEnv()
    agent_maxmin = MaxMinQLearningAgent()
    agent_sarsa = SARSAAgent()
    
    rewards_maxmin = np.zeros(episodes)
    rewards_sarsa = np.zeros(episodes)
    
    for e in range(episodes):
        action_maxmin = agent_maxmin.choose_action()
        action_sarsa = agent_sarsa.choose_action()
        
        rewards = env.step(action_maxmin, action_sarsa)
        reward_maxmin, reward_sarsa = rewards
        
        next_action_maxmin = agent_maxmin.choose_action()  # For MaxMin, we don't really use this
        agent_maxmin.update(action_maxmin, reward_maxmin)
        
        next_action_sarsa = agent_sarsa.choose_action()
        agent_sarsa.update(action_sarsa, reward_sarsa, next_action_sarsa)
        
        rewards_maxmin[e] = reward_maxmin
        rewards_sarsa[e] = reward_sarsa
    
    return rewards_maxmin.cumsum(), rewards_sarsa.cumsum()


episodes = 1000
cum_rewards_maxmin, cum_rewards_sarsa = simulate_battle_of_the_sexes(episodes=episodes)


plt.figure(figsize=(10, 6))
plt.plot(cum_rewards_maxmin, label='Cumulative Reward: Max-Min Q-Learning')
plt.plot(cum_rewards_sarsa, label='Cumulative Reward: SARSA Q-Learning')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Rewards of Max-Min Q-Learning vs SARSA Q-Learning')
plt.legend()
plt.grid()
plt.show()
