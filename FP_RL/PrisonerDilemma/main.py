
import numpy as np
import matplotlib.pyplot as plt

class PrisonersDilemma:
    def __init__(self):
        self.payoff_matrix = np.array([[3, 0], [5, 1]])  # Payoffs for (Cooperate, Defect)
        self.strategies = ['Cooperate', 'Defect']
        self.history_A = np.zeros(2)
        self.history_B = np.zeros(2)

    def play_round(self, strategy_A, strategy_B):
        return self.payoff_matrix[strategy_A, strategy_B], self.payoff_matrix[strategy_B, strategy_A]

    def fictitious_play(self, num_rounds=1000):
        cumulative_rewards_A = np.zeros(num_rounds)
        cumulative_rewards_B = np.zeros(num_rounds)
        actions_A = np.zeros(num_rounds, dtype=int)
        actions_B = np.zeros(num_rounds, dtype=int)

        for t in range(num_rounds):
            if t > 0:
                avg_strategy_A = self.history_A / t
                avg_strategy_B = self.history_B / t
                best_response_A = np.argmax(self.payoff_matrix @ avg_strategy_B)
                best_response_B = np.argmax(self.payoff_matrix @ avg_strategy_A)
            else:
                best_response_A, best_response_B = np.random.choice([0, 1], size=2)

            reward_A, reward_B = self.play_round(best_response_A, best_response_B)
            cumulative_rewards_A[t] = cumulative_rewards_A[t-1] + reward_A if t > 0 else reward_A
            cumulative_rewards_B[t] = cumulative_rewards_B[t-1] + reward_B if t > 0 else reward_B

            actions_A[t] = best_response_A
            actions_B[t] = best_response_B

            self.history_A[best_response_A] += 1
            self.history_B[best_response_B] += 1

        return cumulative_rewards_A, cumulative_rewards_B, actions_A, actions_B

    def plot_results(self, cumulative_rewards_A, cumulative_rewards_B, actions_A, actions_B):
        plt.figure(figsize=(14, 6))

        # Plot cumulative rewards
        plt.subplot(1, 2, 1)
        plt.plot(cumulative_rewards_A, label='Player A')
        plt.plot(cumulative_rewards_B, label='Player B')
        plt.title('Cumulative Rewards')
        plt.xlabel('Rounds')
        plt.ylabel('Cumulative Reward')
        plt.legend()

        # Plot actions
        plt.subplot(1, 2, 2)
        plt.plot(actions_A, label='Player A Actions')
        plt.plot(actions_B, label='Player B Actions', linestyle='--')
        plt.yticks([0, 1], ['Cooperate', 'Defect'])
        plt.title('Actions Over Time')
        plt.xlabel('Rounds')
        plt.ylabel('Action')
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    game = PrisonersDilemma()
    cumulative_rewards_A, cumulative_rewards_B, actions_A, actions_B = game.fictitious_play()
    game.plot_results(cumulative_rewards_A, cumulative_rewards_B, actions_A, actions_B)
