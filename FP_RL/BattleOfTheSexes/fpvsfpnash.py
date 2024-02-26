import numpy as np
import matplotlib.pyplot as plt

# Game Setup for Battle of the Sexes
ACTIONS = ['Opera', 'Football']
NUM_ACTIONS = len(ACTIONS)
# Player A's payoff matrix
PAYOFFS_A = np.array([
    [3, 1],  # Payoffs when Player A chooses Opera
    [0, 2]   # Payoffs when Player A chooses Football
])
# Player B's payoff matrix
PAYOFFS_B = np.array([
    [2, 0],  # Payoffs when Player B chooses Opera
    [1, 3]   # Payoffs when Player B chooses Football
])

class FictitiousPlayAgent:
    def __init__(self, payoff_matrix):
        self.payoff_matrix = payoff_matrix
        self.opponent_action_counts = np.zeros(NUM_ACTIONS)
    
    def update_opponent_strategy(self, opponent_action):
        self.opponent_action_counts[opponent_action] += 1

    def choose_action(self):
        if np.sum(self.opponent_action_counts) == 0:
            return np.random.randint(NUM_ACTIONS)  # Random action if no history
        opponent_strategy = self.opponent_action_counts / np.sum(self.opponent_action_counts)
        expected_payoffs = self.payoff_matrix @ opponent_strategy
        return np.argmax(expected_payoffs)

def simulate(game_rounds):
    agent_a = FictitiousPlayAgent(PAYOFFS_A)
    agent_b = FictitiousPlayAgent(PAYOFFS_B)
    action_probabilities_a = np.zeros((game_rounds, NUM_ACTIONS))
    action_probabilities_b = np.zeros((game_rounds, NUM_ACTIONS))

    for round in range(game_rounds):
        action_a = agent_a.choose_action()
        action_b = agent_b.choose_action()
        
        agent_a.update_opponent_strategy(action_b)
        agent_b.update_opponent_strategy(action_a)
        
        action_probabilities_a[round] = agent_a.opponent_action_counts / np.sum(agent_a.opponent_action_counts)
        action_probabilities_b[round] = agent_b.opponent_action_counts / np.sum(agent_b.opponent_action_counts)
    
    return action_probabilities_a, action_probabilities_b

def plot_action_probabilities(action_probabilities, title):
    plt.plot(action_probabilities[:, 0], label='Opera')
    plt.plot(action_probabilities[:, 1], label='Football')
    plt.title(title)
    plt.xlabel('Rounds')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

game_rounds = 1000
action_probabilities_a, action_probabilities_b = simulate(game_rounds)

plot_action_probabilities(action_probabilities_a, 'Player A Action Probabilities Over Time')
plot_action_probabilities(action_probabilities_b, 'Player B Action Probabilities Over Time')
