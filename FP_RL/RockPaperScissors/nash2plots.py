import numpy as np
import matplotlib.pyplot as plt

# Constants
ACTIONS = ['Rock', 'Paper', 'Scissors']
NUM_ACTIONS = len(ACTIONS)

# Payoff matrix for Rock-Paper-Scissors
PAYOFF_MATRIX = np.array([
    [0, -1, 1],  # Rock
    [1, 0, -1],  # Paper
    [-1, 1, 0]   # Scissors
])

def best_response(opponent_history):
    if sum(opponent_history) == 0:
        return np.random.choice(NUM_ACTIONS)  # Random choice if no history
    opponent_strategy = opponent_history / sum(opponent_history)
    expected_payoffs = PAYOFF_MATRIX @ opponent_strategy
    return np.random.choice(np.where(expected_payoffs == np.max(expected_payoffs))[0])

def update_strategy(agent_history):
    strategy = np.zeros(NUM_ACTIONS)
    if sum(agent_history) > 0:
        strategy = agent_history / sum(agent_history)
    return strategy

def fictitious_play(num_iterations):
    history1 = np.zeros(NUM_ACTIONS)
    history2 = np.zeros(NUM_ACTIONS)

    strategy_history1 = []
    strategy_history2 = []

    for _ in range(num_iterations):
        action1 = best_response(history2)
        action2 = best_response(history1)

        history1[action1] += 1
        history2[action2] += 1

        strategy1 = update_strategy(history1)
        strategy2 = update_strategy(history2)
        strategy_history1.append(strategy1)
        strategy_history2.append(strategy2)

    return np.array(strategy_history1), np.array(strategy_history2)

def plot_strategy(strategy_history, agent_number, num_iterations):
    plt.figure(figsize=(10, 4))
    for i, action in enumerate(ACTIONS):
        plt.plot(range(num_iterations), strategy_history[:, i], label=f'{action}')

    plt.title(f'Agent {agent_number} Strategy Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Action Probability')
    plt.legend()
    plt.show()

# Main simulation
num_iterations = 1000
strategy_history1, strategy_history2 = fictitious_play(num_iterations)

# Plotting for each agent
plot_strategy(strategy_history1, 1, num_iterations)
plot_strategy(strategy_history2, 2, num_iterations)
