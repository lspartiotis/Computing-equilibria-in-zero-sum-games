import numpy as np
import matplotlib.pyplot as plt

# Define the Rock-Paper-Scissors game environment
class RPSGame:
    actions = ['rock', 'paper', 'scissors']

    @staticmethod
    def get_reward(action1, action2):
        if action1 == action2:
            return 0  # Tie
        elif (action1 == 'rock' and action2 == 'scissors') or \
             (action1 == 'scissors' and action2 == 'paper') or \
             (action1 == 'paper' and action2 == 'rock'):
            return 1  # Win
        else:
            return -1  # Lose

# Define the RL agents
class RLAgent:
    def __init__(self, learning_algorithm):
        self.q_table = np.zeros((3, 3))  # Simplified Q-table (action x action)
        self.learning_algorithm = learning_algorithm
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Exploration rate

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(3)  # Explore
        else:
            return np.argmax(self.q_table[:, 0])  # Exploit

    def update_q_table(self, action1, action2, reward):
        if self.learning_algorithm == 'minimax':
            best_response = np.min(self.q_table[action2, :])
            self.q_table[action1, action2] += self.alpha * (reward + self.gamma * best_response - self.q_table[action1, action2])
        elif self.learning_algorithm == 'sarsa':
            next_action = self.choose_action()
            self.q_table[action1, action2] += self.alpha * (reward + self.gamma * self.q_table[action2, next_action] - self.q_table[action1, action2])

# Softmax function to convert Q-values into probabilities
def softmax(q_values):
    exp_q = np.exp(q_values - np.max(q_values))
    return exp_q / exp_q.sum()

# Modified simulation to track probabilities
def simulate_games_with_probabilities(n_episodes=1000):
    agent1 = RLAgent('minimax')
    agent2 = RLAgent('sarsa')
    probabilities1, probabilities2 = [], []
    
    for _ in range(n_episodes):
        action1 = agent1.choose_action()
        action2 = agent2.choose_action()
        
        reward = RPSGame.get_reward(RPSGame.actions[action1], RPSGame.actions[action2])
        
        agent1.update_q_table(action1, action2, reward)
        agent2.update_q_table(action2, action1, -reward)
        
        probabilities1.append(softmax(agent1.q_table[:, 0]))
        probabilities2.append(softmax(agent2.q_table[:, 0]))
        
    return np.array(probabilities1), np.array(probabilities2)

# Plotting function for the action probabilities
def plot_action_probabilities(probabilities, title):
    plt.figure(figsize=(10, 6))
    plt.plot(probabilities[:, 0], label='Rock')
    plt.plot(probabilities[:, 1], label='Paper')
    plt.plot(probabilities[:, 2], label='Scissors')
    plt.xlabel('Episode')
    plt.ylabel('Probability')
    plt.title(title)
    plt.legend()
    plt.show()

# Example usage
n_episodes = 1000
probabilities_minimax, probabilities_sarsa = simulate_games_with_probabilities(n_episodes)
plot_action_probabilities(probabilities_minimax, 'Minimax Q-Learning Action Probabilities')
plot_action_probabilities(probabilities_sarsa, 'SARSA Q-Learning Action Probabilities')
