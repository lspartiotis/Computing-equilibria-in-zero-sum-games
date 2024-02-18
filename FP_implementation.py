'''
Prisoners dillema

              confess  don't confess
confess       (-6,-6)  |  (0,-10) 
don't confess (-10,0)  |  (-1,-1)
'''
import numpy as np
import matplotlib.pyplot as plt

def best_response(belief, player_payoff):
    # Function to calculate the best response based on the current belief
    coop_util = belief[0] * player_payoff[0, 0] + belief[1] * player_payoff[0, 1]
    defect_util = belief[0] * player_payoff[1, 0] + belief[1] * player_payoff[1, 1]

    return np.argmax([coop_util, defect_util])

def update_beliefs_frequency(beliefs, opponent_action, total_rounds):
    # Function to update beliefs based on the frequency of the opponent's actions
    beliefs[opponent_action] = ((total_rounds - 1) * beliefs[opponent_action] + 1) / total_rounds
    beliefs[1 - opponent_action] = 1 - beliefs[opponent_action]

    return beliefs

if __name__ == "__main__":
    # Payoff matrices
    player1_payoff = np.array([[-6, 0], [-10, -1]])
    player2_payoff = np.array([[-6, 10], [0, -1]])

    # Initial beliefs
    belief_p1 = np.array([0.5, 0.5])
    belief_p2 = np.array([0.5, 0.5])

    # Initialize lists to store beliefs over time
    beliefs_p1_over_time = []
    beliefs_p2_over_time = []

    # Simulation parameters
    num_rounds = 100

    for round in range(1, num_rounds + 1):
        # Players choose their best response
        action_p1 = best_response(belief_p1, player1_payoff)
        action_p2 = best_response(belief_p2, player2_payoff)
        
        # Update beliefs based on the opponent's action using frequency-based approach
        belief_p1 = update_beliefs_frequency(belief_p1, action_p2, round)
        belief_p2 = update_beliefs_frequency(belief_p2, action_p1, round)

        # Store beliefs
        beliefs_p1_over_time.append(belief_p1[0])
        beliefs_p2_over_time.append(belief_p2[0])

    # Plotting the beliefs over time
    plt.figure(figsize=(10, 5))
    plt.plot(beliefs_p1_over_time, label='Player 1 Belief (P2 cooperates)')
    plt.plot(beliefs_p2_over_time, label='Player 2 Belief (P1 cooperates)')
    plt.xlabel('Round')
    plt.ylabel('Belief Probability')
    plt.title('Beliefs Convergence Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()










