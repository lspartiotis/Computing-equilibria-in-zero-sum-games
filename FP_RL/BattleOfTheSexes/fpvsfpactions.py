
import numpy as np
import matplotlib.pyplot as plt

def run_fictitious_play(num_iterations=100):
    # Define the payoff matrices for the Battle of the Sexes
    payoff_matrix_A = np.array([[3, 0], [1, 2]])  # Player A's payoff matrix
    payoff_matrix_B = np.array([[2, 1], [0, 3]])  # Player B's payoff matrix

    # Initialize histories of actions
    history_A = np.zeros((2, num_iterations+1))  # +1 for initial belief
    history_B = np.zeros((2, num_iterations+1))  # +1 for initial belief

    # Initial beliefs: each player assumes that the other is equally likely to choose any action
    history_A[:, 0] = 1/2
    history_B[:, 0] = 1/2

    # Fictitious play simulation
    for t in range(1, num_iterations + 1):
        expected_payoff_A = payoff_matrix_A @ history_B[:, t-1]
        expected_payoff_B = payoff_matrix_B.T @ history_A[:, t-1]
        
        # Best response for Player A
        best_response_A = np.argmax(expected_payoff_A)
        # Best response for Player B
        best_response_B = np.argmax(expected_payoff_B)
        
        # Update history of actions
        history_A[best_response_A, t] = history_A[best_response_A, t-1] + 1
        history_A[:, t] /= np.sum(history_A[:, t])
        
        history_B[best_response_B, t] = history_B[best_response_B, t-1] + 1
        history_B[:, t] /= np.sum(history_B[:, t])

    # Plotting the actions chosen by each agent over time
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history_A[0, :], label='Ballet (B)')
    plt.plot(history_A[1, :], label='Football (F)')
    plt.title('Player A Strategy Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Probability of Choosing Action')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_B[0, :], label='Ballet (B)')
    plt.plot(history_B[1, :], label='Football (F)')
    plt.title('Player B Strategy Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Probability of Choosing Action')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_fictitious_play()
