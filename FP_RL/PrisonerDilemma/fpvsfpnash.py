import numpy as np
import matplotlib.pyplot as plt

# Game Setup for Prisoner's Dilemma
ACTIONS = ['Cooperate', 'Defect']
NUM_ACTIONS = len(ACTIONS)
PAYOFFS = np.array([
    [(-1, -1), (-3, 0)],  # Player 1 Cooperates
    [(0, -3), (-2, -2)]   # Player 1 Defects
])

# Fictitious Play Agent
class FictitiousPlayAgent:
    def __init__(self):
        self.opponent_action_counts = np.zeros(NUM_ACTIONS)
    
    def update(self, opponent_action):
        self.opponent_action_counts[opponent_action] += 1

    def choose_action(self):
        if np.sum(self.opponent_action_counts) == 0:
            return np.random.randint(NUM_ACTIONS)  # Random action if no history
        opponent_strategy = self.opponent_action_counts / np.sum(self.opponent_action_counts)
        # Calculate expected utility for each action
        my_expected_payoffs = [np.dot(row, opponent_strategy) for row in PAYOFFS[:, :, 0]]
        return np.argmax(my_expected_payoffs)  # Best response to opponent's strategy
    
    def get_strategy(self):
        if np.sum(self.opponent_action_counts) == 0:
            return np.array([0.5, 0.5])  # Equal probabilities if no history
        return self.opponent_action_counts / np.sum(self.opponent_action_counts)

# Simulation with Strategy Tracking
def simulate_with_strategy_tracking(game_rounds=1000):
    agent1 = FictitiousPlayAgent()
    agent2 = FictitiousPlayAgent()
    strategies1 = np.zeros((game_rounds, NUM_ACTIONS))
    strategies2 = np.zeros((game_rounds, NUM_ACTIONS))

    for i in range(game_rounds):
        action1 = agent1.choose_action()
        action2 = agent2.choose_action()
        
        agent1.update(action2)
        agent2.update(action1)
        
        strategies1[i] = agent1.get_strategy()
        strategies2[i] = agent2.get_strategy()

    return strategies1, strategies2

def plot_strategy_evolution(strategies1, strategies2, title):
    plt.figure(figsize=(14, 7))
    
    iterations = np.arange(len(strategies1))
    
    # Plotting the strategy evolution for Agent 1
    plt.plot(iterations, strategies1[:, 0], label='Agent 1: Prob(Cooperate)', color='skyblue')
    plt.plot(iterations, strategies1[:, 1], label='Agent 1: Prob(Defect)', color='blue')
    
    # Plotting the strategy evolution for Agent 2
    plt.plot(iterations, strategies2[:, 0], '--', label='Agent 2: Prob(Cooperate)', color='orange')
    plt.plot(iterations, strategies2[:, 1], '--', label='Agent 2: Prob(Defect)', color='red')
    
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()

game_rounds = 1000
strategies1, strategies2 = simulate_with_strategy_tracking(game_rounds)

plot_strategy_evolution(strategies1, strategies2, 'Strategy Evolution in Prisoner\'s Dilemma')
