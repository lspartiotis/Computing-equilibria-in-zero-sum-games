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

class MinMaxRLAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.q_values = np.zeros((num_actions, num_actions))  # [Agent Action][Opponent Action]
        self.alpha = 0.1
        self.epsilon = 0.1

    def choose_action(self):
        avg_q_values = np.mean(self.q_values, axis=1)
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(len(avg_q_values)))
        else:
            return np.argmax(avg_q_values)

    def update(self, action, opponent_action, reward):
        self.q_values[action, opponent_action] += self.alpha * (reward - self.q_values[action, opponent_action])

    def get_action_probabilities(self):
        avg_q_values = np.mean(self.q_values, axis=1)
        return softmax(avg_q_values)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def get_reward(action1, action2):
    if action1 == action2:
        return 0, 0
    elif (action1 - action2) % 3 == 1:
        return 1, -1
    else:
        return -1, 1

def simulate_game_and_track_probabilities(agent1, agent2, iterations=1000):
    outcomes_agent1 = np.zeros(iterations)
    outcomes_agent2 = np.zeros(iterations)
    probabilities_agent1 = np.zeros((iterations, 3))
    probabilities_agent2 = np.zeros((iterations, 3))
    
    for i in range(iterations):
        action1 = agent1.choose_action()
        action2 = agent2.choose_action()

        reward1, reward2 = get_reward(action1, action2)

        agent1.update(action1, action2, reward1)
        agent2.update(action2)  # FP agent does not use reward for update

        if reward1 > 0:  # Agent1 wins
            outcomes_agent1[i] = 1
        elif reward1 < 0:  # Agent2 wins
            outcomes_agent2[i] = 1
        
        probabilities_agent1[i] = agent1.get_action_probabilities()
        probabilities_agent2[i] = agent2.get_action_probabilities()

    # Calculate cumulative wins for win/loss ratio
    cumulative_wins_agent1 = np.cumsum(outcomes_agent1)
    cumulative_wins_agent2 = np.cumsum(outcomes_agent2)
    
    win_loss_ratio_agent1 = cumulative_wins_agent1 / (np.arange(iterations) + 1)
    win_loss_ratio_agent2 = cumulative_wins_agent2 / (np.arange(iterations) + 1)
    
    return win_loss_ratio_agent1, win_loss_ratio_agent2, probabilities_agent1, probabilities_agent2

# Initialize agents
rl_agent = MinMaxRLAgent(num_actions=3)
fp_agent = FPAgent(num_actions=3)

# Simulate the game and track win/loss ratios and action probabilities
win_loss_ratio_rl, win_loss_ratio_fp, probabilities_rl, probabilities_fp = simulate_game_and_track_probabilities(rl_agent, fp_agent, iterations=1000)

# Plotting win/loss ratios
plt.figure(figsize=(10, 6))
plt.plot(win_loss_ratio_rl, label='MinMax RL Agent Win/Loss Ratio')
plt.plot(win_loss_ratio_fp, label='FP Agent Win/Loss Ratio')
plt.title('Win/Loss Ratio Over Time')
plt.xlabel('Iterations')
plt.ylabel('Win/Loss Ratio')
plt.legend()
plt.grid(True)
plt.show()

# Plotting action probabilities for MinMax RL Agent
plt.figure(figsize=(10, 6))
for i, action in enumerate(['Rock', 'Paper', 'Scissors']):
    plt.plot(probabilities_rl[:, i], label=f'{action} Probability')
plt.title('MinMax RL Agent Action Probabilities')
plt.xlabel('Iterations')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()

# Plotting action probabilities for FP Agent
plt.figure(figsize=(10, 6))
for i, action in enumerate(['Rock', 'Paper', 'Scissors']):
    plt.plot(probabilities_fp[:, i], label=f'{action} Probability')
plt.title('FP Agent Action Probabilities')
plt.xlabel('Iterations')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()
