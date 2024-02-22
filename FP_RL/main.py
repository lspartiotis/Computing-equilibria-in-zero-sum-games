import numpy as np
import matplotlib.pyplot as plt

class Game:
    def __init__(self, payoff_matrix):
        self.payoff_matrix = payoff_matrix

    def play(self, action1, action2):
        return self.payoff_matrix[action1][action2]

# Implementing the strategies
class RLAgent:
    def __init__(self, num_actions):
        self.q_values = np.zeros(num_actions)
        self.alpha = 0.1  # learning rate
        self.gamma = 0.9  # discount factor
        self.epsilon = 0.1  # exploration rate

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.q_values))
        else:
            return np.argmax(self.q_values)

    def update(self, action, reward):
        self.q_values[action] += self.alpha * (reward - self.q_values[action])

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

    def update(self, action, reward=None):  # Adjusted to accept a 'reward' parameter, but does not use it
        self.action_counts[action] += 1


def simulate_game(game, agent1, agent2, iterations=1000):
    agent1_rewards = []
    agent2_rewards = []
    for _ in range(iterations):
        action1 = agent1.choose_action()
        action2 = agent2.choose_action()
        reward1, reward2 = game.play(action1, action2)
        agent1.update(action1, reward1)
        agent2.update(action2, reward2)
        agent1_rewards.append(reward1)
        agent2_rewards.append(reward2)
    return agent1_rewards, agent2_rewards

# Define games
prisoners_dilemma_payoffs = np.array([[(3,3), (0,5)], [(5,0), (1,1)]])
rock_paper_scissors_payoffs = np.array([[(0,0), (-1,1), (1,-1)], [(1,-1), (0,0), (-1,1)], [(-1,1), (1,-1), (0,0)]])
battle_of_the_sexes_payoffs = np.array([[(2,1), (0,0)], [(0,0), (1,2)]])

games = [
    ("Prisoner's Dilemma", Game(prisoners_dilemma_payoffs)),
    ("Rock-Paper-Scissors", Game(rock_paper_scissors_payoffs)),
    ("Battle of the Sexes", Game(battle_of_the_sexes_payoffs))
]

for game_name, game in games:
    rl_agent = RLAgent(len(game.payoff_matrix))
    fp_agent = FPAgent(len(game.payoff_matrix), game.payoff_matrix[:, :, 1])  # Assuming FP agent is player 2
    rewards = simulate_game(game, rl_agent, fp_agent)
    plt.figure()
    plt.title(f"{game_name} - Average Rewards Over Time")
    plt.plot(np.cumsum(rewards[0]) / np.arange(1, len(rewards[0]) + 1), label='RL Agent')
    plt.plot(np.cumsum(rewards[1]) / np.arange(1, len(rewards[1]) + 1), label='FP Agent')
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.show()
