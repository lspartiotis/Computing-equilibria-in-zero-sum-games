import numpy as np
import matplotlib.pyplot as plt

class Game:
    def __init__(self, payoff_matrix):
        self.payoff_matrix = payoff_matrix

    def play(self, action1, action2):
        return self.payoff_matrix[action1][action2]

class MinMaxRLAgent:
    def __init__(self, num_actions, opponent_actions):
        self.q_values = np.zeros((num_actions, opponent_actions))
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate
        self.opponent_actions = opponent_actions

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.q_values))
        else:
            estimated_opponent_strategy = np.ones(self.opponent_actions) / self.opponent_actions
            expected_rewards = self.q_values.dot(estimated_opponent_strategy)
            return np.argmax(expected_rewards)

    def update(self, action, opponent_action, reward):
        self.q_values[action, opponent_action] += self.alpha * (reward - self.q_values[action, opponent_action])

class FPAgent:
    def __init__(self, num_actions, payoff_matrix):
        self.num_actions = num_actions
        self.action_counts = np.zeros(num_actions)
        self.payoff_matrix = payoff_matrix

    def choose_action(self):
        if np.sum(self.action_counts) == 0:
            return np.random.choice(self.num_actions)
        opponent_strategy = self.action_counts / np.sum(self.action_counts)
        my_payoffs = np.dot(self.payoff_matrix, opponent_strategy)
        return np.argmax(my_payoffs)

    def update(self, action, opponent_action=None, reward=None):  # Updated to match expected signature
        self.action_counts[action] += 1

def simulate_game(game, agent1, agent2, iterations=1000):
    agent1_rewards = []
    agent2_rewards = []
    for _ in range(iterations):
        action1 = agent1.choose_action()
        action2 = agent2.choose_action()
        reward1, reward2 = game.play(action1, action2)
        agent1.update(action1, action2, reward1)
        agent2.update(action2, action1, reward2)
        agent1_rewards.append(reward1)
        agent2_rewards.append(reward2)
    return agent1_rewards, agent2_rewards

# Example game setup
payoff_matrix = np.array([[(3, 3), (0, 5)], [(5, 0), (1, 1)]])
game = Game(payoff_matrix)

# Create agents
rl_agent = MinMaxRLAgent(num_actions=2, opponent_actions=2)
fp_agent = FPAgent(num_actions=2, payoff_matrix=payoff_matrix[:, :, 1])  # Assuming FP agent is player 2

# Simulate the game
agent1_rewards, agent2_rewards = simulate_game(game, rl_agent, fp_agent, 1000)

# Plotting the results
plt.figure()
plt.title("Average Rewards Over Time")
plt.plot(np.cumsum(agent1_rewards) / np.arange(1, len(agent1_rewards) + 1), label='MinMax RL Agent')
plt.plot(np.cumsum(agent2_rewards) / np.arange(1, len(agent2_rewards) + 1), label='FP Agent')
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.legend()
plt.show()
