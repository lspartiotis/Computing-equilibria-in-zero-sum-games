import numpy as np
import matplotlib.pyplot as plt

# Constants
ACTIONS = ['Rock', 'Paper', 'Scissors']
NUM_ACTIONS = len(ACTIONS)
ALPHA = 0.1  # Learning rate for Q-learning agent
GAMMA = 0.9  # Discount factor for future rewards
EPSILON = 0.1  # Exploration rate

# Payoff matrix for Rock-Paper-Scissors
# Rows: FP Agent's actions, Columns: RL Agent's actions
# Values: FP Agent's payoff
PAYOFF_MATRIX = np.array([
    [0, -1, 1],  # Rock
    [1, 0, -1],  # Paper
    [-1, 1, 0]   # Scissors
])

def best_response(opponent_history):
    if np.sum(opponent_history) == 0:
        return np.random.randint(NUM_ACTIONS)
    strategy = opponent_history / np.sum(opponent_history)
    expected_utility = PAYOFF_MATRIX @ strategy
    return np.argmax(expected_utility)

def update_q(agent_action, opponent_action, reward):
    current_q = Q[agent_action, opponent_action]
    future_rewards = np.max(Q[opponent_action])  # Assuming next state is the opponent's action
    Q[agent_action, opponent_action] = current_q + ALPHA * (reward + GAMMA * future_rewards - current_q)

def fictitious_play_vs_q_learning(num_iterations):
    fp_history = np.zeros(NUM_ACTIONS)
    q_strategy_history = []
    fp_strategy_history = []

    for _ in range(num_iterations):
        fp_action = best_response(fp_history)
        if np.random.rand() < EPSILON:
            q_action = np.random.randint(NUM_ACTIONS)
        else:
            q_action = np.argmax(np.mean(Q, axis=1))

        # Update FP history
        fp_history[q_action] += 1

        # Get reward from the payoff matrix
        reward = PAYOFF_MATRIX[fp_action, q_action] - PAYOFF_MATRIX[q_action, fp_action]

        # Update Q-table for the RL agent
        update_q(q_action, fp_action, reward)

        # Record strategies
        q_strategy = np.zeros(NUM_ACTIONS)
        q_strategy[q_action] = 1
        q_strategy_history.append(q_strategy)
        fp_strategy = fp_history / np.sum(fp_history)
        fp_strategy_history.append(fp_strategy)

    return np.array(fp_strategy_history), np.array(q_strategy_history)

Q = np.random.uniform(low=-1, high=1, size=(NUM_ACTIONS, NUM_ACTIONS))

# Simulate and plot
num_iterations = 1000
fp_strategy_history, q_strategy_history = fictitious_play_vs_q_learning(num_iterations)

def plot_strategy(strategy_history, agent_name, num_iterations):
    plt.figure(figsize=(10, 4))
    for i, action in enumerate(ACTIONS):
        plt.plot(range(num_iterations), strategy_history[:, i], label=f'{action}')
    plt.title(f'{agent_name} Strategy Evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Action Probability')
    plt.legend()
    plt.show()

plot_strategy(fp_strategy_history, 'Fictitious Play Agent', num_iterations)
plot_strategy(q_strategy_history, 'Minimax Q-Learning Agent', num_iterations)
