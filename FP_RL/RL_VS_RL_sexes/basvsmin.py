import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt

class BattleOfTheSexesEnv:
    def __init__(self):
        self.action_space = [0, 1]  # 0 for Opera, 1 for Football
        self.payoffs = {('0', '0'): (2, 1), ('0', '1'): (0, 0),
                        ('1', '0'): (0, 0), ('1', '1'): (1, 2)}

    def step(self, action_husband, action_wife):
        key = (str(action_husband), str(action_wife))
        return self.payoffs[key]

class MinMaxQLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
        self.actions = actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: choose a random action
            return random.choice(self.actions)
        else:
            # Exploitation: choose the action with the minimum Q-value (adversarial perspective)
            return np.argmin(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        # Update using the minimum Q-value for the next state (considering adversarial perspective)
        next_min_q = np.min(self.q_table[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * next_min_q - current_q)
        self.q_table[state][action] = new_q

class SARSAgent:
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

    def learn(self, state, action, reward, next_state, next_action):
        current_q = self.q_table[state][action]
        next_q = self.q_table[next_state][next_action]
        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[state][action] = new_q

def simulate_episodes(env, num_episodes=1000):
    husband_agent = MinMaxQLearningAgent(env.action_space)
    wife_agent = SARSAgent(env.action_space)

    husband_cumulative_rewards = np.zeros(num_episodes)
    wife_cumulative_rewards = np.zeros(num_episodes)

    for episode in range(num_episodes):
        state = None  # Simplified as there's no state change
        done = False
        husband_reward_total, wife_reward_total = 0, 0

        action_wife = wife_agent.choose_action(state)
        # Proceed with the episode
        while not done:
            action_husband = husband_agent.choose_action(state)
            rewards = env.step(action_husband, action_wife)
            husband_reward, wife_reward = rewards

            next_state = state  # No state change in this game
            next_action_wife = wife_agent.choose_action(next_state)

            # Min-Max Q-Learning Husband learns
            husband_agent.learn(state, action_husband, husband_reward, next_state)

            # SARSA Wife learns
            wife_agent.learn(state, action_wife, wife_reward, next_state, next_action_wife)
            action_wife = next_action_wife  # Update action for SARSA
            
            husband_reward_total += husband_reward
            wife_reward_total += wife_reward
            done = True

        husband_cumulative_rewards[episode] = husband_reward_total if episode == 0 else husband_cumulative_rewards[episode - 1] + husband_reward_total
        wife_cumulative_rewards[episode] = wife_reward_total if episode == 0 else wife_cumulative_rewards[episode - 1] + wife_reward_total

    return husband_cumulative_rewards, wife_cumulative_rewards

def plot_cumulative_rewards(husband_cumulative_rewards, wife_cumulative_rewards):
    plt.plot(husband_cumulative_rewards, label='Husband (Min-Max Q-Learning)')
    plt.plot(wife_cumulative_rewards, label='Wife (SARSA)')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('Min-Max Q-Learning vs SARSA: Cumulative Rewards Over Time')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    env = BattleOfTheSexesEnv()
    num_episodes = 1000
    husband_cumulative_rewards, wife_cumulative_rewards = simulate_episodes(env, num_episodes)
    plot_cumulative_rewards(husband_cumulative_rewards, wife_cumulative_rewards)
