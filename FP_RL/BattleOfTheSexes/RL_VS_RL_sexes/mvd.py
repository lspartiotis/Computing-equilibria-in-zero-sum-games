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

class MinMaxQLearningAgent(QLearningAgent):
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            return np.argmin(self.q_table[state])  # Min for adversarial approach

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        next_min_q = np.min(self.q_table[next_state])  # Assume adversarial environment
        new_q = (1 - self.alpha) * current_q + self.alpha * (reward + self.gamma * next_min_q)
        self.q_table[state][action] = new_q

def simulate_episodes(env, num_episodes=1000):
    husband_agent = QLearningAgent(env.action_space)
    wife_agent = MinMaxQLearningAgent(env.action_space)

    husband_cumulative_rewards = np.zeros(num_episodes)
    wife_cumulative_rewards = np.zeros(num_episodes)

    for episode in range(num_episodes):
        state = None  # Simplified as there's no state change
        action_husband = husband_agent.choose_action(state)
        action_wife = wife_agent.choose_action(state)
        rewards = env.step(action_husband, action_wife)
        
        husband_reward, wife_reward = rewards
        if episode > 0:
            husband_cumulative_rewards[episode] = husband_cumulative_rewards[episode-1] + husband_reward
            wife_cumulative_rewards[episode] = wife_cumulative_rewards[episode-1] + wife_reward
        else:
            husband_cumulative_rewards[episode] = husband_reward
            wife_cumulative_rewards[episode] = wife_reward

        husband_agent.learn(state, action_husband, husband_reward, state)
        wife_agent.learn(state, action_wife, wife_reward, state)

    return husband_cumulative_rewards, wife_cumulative_rewards

def plot_cumulative_rewards(husband_cumulative_rewards, wife_cumulative_rewards):
    plt.plot(husband_cumulative_rewards, label='Husband (Q-Learning)')
    plt.plot(wife_cumulative_rewards, label='Wife (Min-Max Q-Learning)')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards Over Time')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    env = BattleOfTheSexesEnv()
    num_episodes = 1000
    husband_cumulative_rewards, wife_cumulative_rewards = simulate_episodes(env, num_episodes)
    plot_cumulative_rewards(husband_cumulative_rewards, wife_cumulative_rewards)
