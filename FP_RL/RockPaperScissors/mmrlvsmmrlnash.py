import numpy as np
import matplotlib.pyplot as plt

# Game Setup
ACTIONS = ['Rock', 'Paper', 'Scissors']
NUM_ACTIONS = len(ACTIONS)
PAYOFFS = np.array([
    [0, -1, 1],
    [1, 0, -1],
    [-1, 1, 0]
])

# Minimax Q-learning Agent with action probability calculation
class MinimaxQLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.1):
        self.Q = np.zeros((NUM_ACTIONS, NUM_ACTIONS))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate

    def choose_action(self, epsilon=True):
        if epsilon and np.random.rand() < self.exploration_rate:
            return np.random.randint(NUM_ACTIONS)
        else:
            return np.argmax(np.min(self.Q, axis=1))
    
    def update(self, my_action, opponent_action, reward):
        best_future_reward = np.max(np.min(self.Q, axis=1))
        self.Q[my_action, opponent_action] += self.learning_rate * (
            reward + self.discount_factor * best_future_reward - self.Q[my_action, opponent_action])

    def action_probabilities(self):
        # Calculate action probabilities using a softmax on Q-values to ensure they are between 0 and 1
        q_values_min = np.min(self.Q, axis=1)  # Use min Q-values for the minimax strategy
        exp_q = np.exp(q_values_min - np.max(q_values_min))  # Subtract max for numerical stability
        return exp_q / np.sum(exp_q)

# Simulation with corrected action probability calculation
def simulate(game_rounds=1000):
    agent1 = MinimaxQLearningAgent()
    agent2 = MinimaxQLearningAgent()
    action_probs_agent1 = np.zeros((game_rounds, NUM_ACTIONS))
    action_probs_agent2 = np.zeros((game_rounds, NUM_ACTIONS))

    for round in range(game_rounds):
        action1 = agent1.choose_action()
        action2 = agent2.choose_action()
        
        reward1 = PAYOFFS[action1, action2]
        reward2 = -reward1
        
        agent1.update(action1, action2, reward1)
        agent2.update(action2, action1, reward2)
        
        action_probs_agent1[round] = agent1.action_probabilities()
        action_probs_agent2[round] = agent2.action_probabilities()

    return action_probs_agent1, action_probs_agent2

def plot_action_probabilities(action_probs, title):
    plt.figure(figsize=(10, 6))
    for i, action in enumerate(ACTIONS):
        plt.plot(action_probs[:, i], label=f'Prob({action})')
    plt.title(title)
    plt.xlabel('Rounds')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

game_rounds = 1000
action_probs_agent1, action_probs_agent2 = simulate(game_rounds)

plot_action_probabilities(action_probs_agent1, 'Minimax Q-Learning Agent 1 Action Probabilities')
plot_action_probabilities(action_probs_agent2, 'Minimax Q-Learning Agent 2 Action Probabilities')
