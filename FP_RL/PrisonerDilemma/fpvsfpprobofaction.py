import numpy as np
import matplotlib.pyplot as plt

class FPAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.action_counts = np.zeros(num_actions)
        
    def choose_action(self):
        if np.sum(self.action_counts) == 0:
            return np.random.choice(self.num_actions)
        strategy = self.action_counts / np.sum(self.action_counts)
        return np.random.choice(self.num_actions, p=strategy)
    
    def update(self, action):
        self.action_counts[action] += 1
        
    def get_action_probabilities(self):
        total = np.sum(self.action_counts)
        if total == 0:
            return np.ones(self.num_actions) / self.num_actions
        return self.action_counts / total

def simulate_game_fp_vs_fp(payoff_matrix, num_iterations=1000):
    num_actions = payoff_matrix.shape[0]
    agent1 = FPAgent(num_actions)
    agent2 = FPAgent(num_actions)
    
    action_probabilities_history_agent1 = []
    action_probabilities_history_agent2 = []
    
    for _ in range(num_iterations):
        action1 = agent1.choose_action()
        action2 = agent2.choose_action()
        
        # Agent 1 observes Agent 2's action and updates
        agent1.update(action2)
        # Agent 2 observes Agent 1's action and updates
        agent2.update(action1)
        
        # Record action probabilities
        action_probabilities_history_agent1.append(agent1.get_action_probabilities())
        action_probabilities_history_agent2.append(agent2.get_action_probabilities())
    
    return action_probabilities_history_agent1, action_probabilities_history_agent2

# Prisoner's Dilemma Payoff Matrix
payoff_matrix = np.array([
    [(3, 3), (0, 5)],
    [(5, 0), (1, 1)]
])

# Run the simulation
num_iterations = 1000
action_probabilities_history_agent1, action_probabilities_history_agent2 = simulate_game_fp_vs_fp(payoff_matrix, num_iterations)

# Plot the results for Agent 1
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plt.plot(np.array(action_probabilities_history_agent1))
plt.title('FP Agent 1 Action Probabilities Over Time')
plt.xlabel('Iterations')
plt.ylabel('Probability')
plt.legend(['Cooperate', 'Defect'])

# Plot the results for Agent 2
plt.subplot(1, 2, 2)
plt.plot(np.array(action_probabilities_history_agent2))
plt.title('FP Agent 2 Action Probabilities Over Time')
plt.xlabel('Iterations')
plt.ylabel('Probability')
plt.legend(['Cooperate', 'Defect'])
plt.tight_layout()
plt.show()
