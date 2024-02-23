


import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

class BattleOfTheSexesEnv:
    def __init__(self):
        self.action_space = [0, 1]  # 0 for Opera, 1 for Football
        # Payoffs: (Husband, Wife)
        self.payoffs = {('0', '0'): (2, 1), ('0', '1'): (0, 0),
                        ('1', '0'): (0, 0), ('1', '1'): (1, 2)}

    def step(self, action_husband, action_wife):
        key = (str(action_husband), str(action_wife))
        return self.payoffs[key]

class MinMaxQLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.actions = actions
        self.alpha = learning_rate  # Learning rate
        self.gamma = discount_factor  # Discount factor
        self.epsilon = exploration_rate  # Exploration rate
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            # Choosing action with the minimum Q-value as a simplistic adversarial strategy
            return np.argmin(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        # Assuming an adversarial environment, consider the minimum future Q-value
        next_min_q = np.min(self.q_table[next_state])
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_min_q)
        self.q_table[state][action] = new_q



class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_max_q)
        self.q_table[state][action] = new_q

def simulate_episodes(env, num_episodes=1000):
    husband_agent = QLearningAgent(env.action_space)
    wife_agent = MinMaxQLearningAgent(env.action_space)

    husband_rewards = []
    wife_rewards = []

    for episode in range(num_episodes):
        done = False
        husband_total_reward = 0
        wife_total_reward = 0
        while not done:
            # In this simple environment, 'state' can be disregarded as it doesn't change
            state = None
            action_husband = husband_agent.choose_action(state)
            action_wife = wife_agent.choose_action(state)
            rewards = env.step(action_husband, action_wife)
            
            husband_reward, wife_reward = rewards
            husband_total_reward += husband_reward
            wife_total_reward += wife_reward

            # Assume next_state is the same since the environment doesn't change
            next_state = state
            
            husband_agent.learn(state, action_husband, husband_reward, next_state)
            wife_agent.learn(state, action_wife, wife_reward, next_state)
            
            done = True  # End after one iteration for simplicity
        
        husband_rewards.append(husband_total_reward)
        wife_rewards.append(wife_total_reward)

    return husband_rewards, wife_rewards

# Plotting the results
def plot_results(husband_rewards, wife_rewards):
    plt.plot(husband_rewards, label='Husband Rewards')
    plt.plot(wife_rewards, label='Wife Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.title('Battle of the Sexes Rewards')
    plt.show()

if __name__ == "__main__":
    env = BattleOfTheSexesEnv()
    num_episodes = 1000
    husband_rewards, wife_rewards = simulate_episodes(env, num_episodes)
    plot_results(husband_rewards, wife_rewards)
