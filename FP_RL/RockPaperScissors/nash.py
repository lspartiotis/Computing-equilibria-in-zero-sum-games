import numpy as np
import matplotlib.pyplot as plt

# Constants
ACTIONS = ['Rock', 'Paper', 'Scissors']
NUM_ACTIONS = len(ACTIONS)

# Payoff matrix for Rock-Paper-Scissors
# Rows: Player 1's actions, Columns: Player 2's actions
# Values: Player 1's payoff
PAYOFF_MATRIX = np.array([
    [0, -1, 1],  # Rock
    [1, 0, -1],  # Paper
    [-1, 1, 0]   # Scissors
])

def best_response(opponent_history):
    """
    Compute the best response to the opponent's strategy based on their action history.
    """
    if sum(opponent_history) == 0:
        return np.random.choice(NUM_ACTIONS)  # If no history, choose randomly
    opponent_strategy = opponent_history / sum(opponent_history)
    expected_payoffs = PAYOFF_MATRIX @ opponent_strategy
    return np.random.choice(np.where(expected_payoffs == np.max(expected_payoffs))[0])

def update_strategy(agent_history):
    """
    Update an agent's strategy based on action history.
    """
    strategy = np.zeros(NUM_ACTIONS)
    if sum(agent_history) > 0:
        strategy = agent_history / sum(agent_history)
    return strategy

def fictitious_play(num_iterations):
    """
    Simulate the game for a given number of iterations using Fictitious Play.
    """
    # Initialize history
    history1 = np.zeros(NUM_ACTIONS)
    history2 = np.zeros(NUM_ACTIONS)

    strategy_history1 = []
    strategy_history2 = []

    for _ in range(num_iterations):
        # Agents choose best response based on the opponent's history
        action1 = best_response(history2)
        action2 = best_response(history1)

        # Update history
        history1[action1] += 1
        history2[action2] += 1

        # Update and record strategy
        strategy1 = update_strategy(history1)
        strategy2 = update_strategy(history2)
        strategy_history1.append(strategy1)
        strategy_history2.append(strategy2)

    return np.array(strategy_history1), np.array(strategy_history2)

def plot_strategies(strategy_history1, strategy_history2, num_iterations):
    """
    Plot the strategy evolution for both agents.
    """
    plt.figure(figsize=(12, 5))

    for i, action in enumerate(ACTIONS):
        plt.plot(range(num_iterations), strategy_history1[:, i], label=f'Agent 1 {action}')
        plt.plot(range(num_iterations), strategy_history2[:, i], '--', label=f'Agent 2 {action}')

    plt.title('Strategy Evolution in Fictitious Play')
    plt.xlabel('Iteration')
    plt.ylabel('Action Probability')
    plt.legend()
    plt.show()

# Main simulation
num_iterations = 1000
strategy_history1, strategy_history2 = fictitious_play(num_iterations)

# Plotting
plot_strategies(strategy_history1, strategy_history2, num_iterations)
