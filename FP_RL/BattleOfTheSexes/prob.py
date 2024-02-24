import numpy as np
import matplotlib.pyplot as plt

class Game:
    def __init__(self, payoff_matrix):
        self.payoff_matrix = payoff_matrix

    def play(self, action1, action2):
        return self.payoff_matrix[action1][action2]

class FPAgent:
    def __init__(self, num_actions, payoff_matrix):
        self.num_actions = num_actions
        self.action_counts = np.zeros(num_actions)
        self.payoff_matrix = payoff_matrix

    def choose_action(self):
        if np.sum(self.action_counts) == 0:
            return np.random.choice(self.num_actions)
        opponent_strategy = self.action_counts / np.sum(self.action_counts)
        my_payoffs = np.dot(self.payoff_matrix, opponent_strategy)
        return np.argmax(my_payoffs)

    def update(self, action):
        self.action_counts[action] += 1

    def get_action_probabilities(self):
        total = np.sum(self.action_counts)
        if total == 0:
            return np.ones(self.num_actions) / self.num_actions
        return self.action_counts / total

class RLAgent:
    def __init__(self, num_actions):
        self.q_values = np.zeros(num_actions)
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.num_actions = num_actions

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.q_values)

    def update(self, action, reward):
        self.q_values[action] += self.alpha * (reward - self.q_values[action])

    def get_action_probabilities(self):
        return softmax(self.q_values)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def simulate_game_with_probabilities(game, agent1, agent2, iterations=1000):
    agent1_action_probs = []
    agent2_action_probs = []
    
    for _ in range(iterations):
        action1 = agent1.choose_action()
        action2 = agent2.choose_action()
        
        reward1, reward2 = game.play(action1, action2)
        
        agent1.update(action1)
        agent2.update(action2, reward2)
        
        agent1_action_probs.append(agent1.get_action_probabilities())
        agent2_action_probs.append(agent2.get_action_probabilities())
    
    return agent1_action_probs, agent2_action_probs

# Battle of the Sexes Payoff Matrix
battle_of_the_sexes_payoffs = np.array([
    [(2, 1), (0, 0)],
    [(0, 0), (1, 2)]
])
game = Game(battle_of_the_sexes_payoffs)

# Initialize agents
fp_agent = FPAgent(2, battle_of_the_sexes_payoffs[:, :, 1])  # FP agent views the world from its perspective
rl_agent = RLAgent(2)  # RL Agent

# Simulate game
fp_action_probs, rl_action_probs = simulate_game_with_probabilities(game, fp_agent, rl_agent, 1000)

# Plotting action probabilities
iterations = np.arange(1, 1001)
plt.figure(figsize=(10, 5))
plt.plot(iterations, np.array(fp_action_probs)[:, 0], label='FP Agent - Opera', linestyle='--')
plt.plot(iterations, np.array(rl_action_probs)[:, 0], label='RL Agent - Opera', linestyle=':')
plt.title('Action Probabilities Over Time: Battle of the Sexes')
plt.xlabel('Iterations')
plt.ylabel('Probability of Choosing Opera')
plt.legend()
plt.show()

