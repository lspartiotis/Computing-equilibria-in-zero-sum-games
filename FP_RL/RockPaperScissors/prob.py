import numpy as np
import matplotlib.pyplot as plt

class FPAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.action_counts = np.zeros(num_actions)

    def choose_action(self):
        if np.sum(self.action_counts) == 0:
            return np.random.choice(self.num_actions)
        # Use the action counts to determine the strategy
        strategy = self.action_counts / np.sum(self.action_counts)
        return np.random.choice(self.num_actions, p=strategy)

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
        self.alpha = 0.1  # Learning rate
        self.epsilon = 0.1  # Exploration rate

    def choose_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(len(self.q_values)))
        else:
            return np.argmax(self.q_values)

    def update(self, action, reward):
        self.q_values[action] += self.alpha * (reward - self.q_values[action])

    def get_action_probabilities(self):
        return np.exp(self.q_values) / np.sum(np.exp(self.q_values))

def get_reward(action1, action2):
    # Rock=0, Paper=1, Scissors=2
    if action1 == action2:
        return 0, 0
    elif (action1 - action2) % 3 == 1:
        return 1, -1
    else:
        return -1, 1

def simulate_game(agent1, agent2, iterations=1000):
    agent1_action_probs = []
    agent2_action_probs = []

    for _ in range(iterations):
        action1 = agent1.choose_action()
        action2 = agent2.choose_action()

        reward1, reward2 = get_reward(action1, action2)

        # Check the type of agent1 and call update method accordingly
        if isinstance(agent1, FPAgent):
            agent1.update(action1)  # FPAgent only needs the action
        elif isinstance(agent1, RLAgent):
            agent1.update(action1, reward1)  # RLAgent needs action and reward

        # Repeat the check for agent2
        if isinstance(agent2, FPAgent):
            agent2.update(action2)  # Same for FPAgent
        elif isinstance(agent2, RLAgent):
            agent2.update(action2, reward2)  # RLAgent needs action and reward

        agent1_action_probs.append(agent1.get_action_probabilities())
        agent2_action_probs.append(agent2.get_action_probabilities())

    return agent1_action_probs, agent2_action_probs


# Initialize agents
fp_agent = FPAgent(num_actions=3)
rl_agent = RLAgent(num_actions=3)

# Simulate the game
fp_action_probs, rl_action_probs = simulate_game(fp_agent, rl_agent, iterations=1000)

# Plotting action probabilities over time
plt.figure(figsize=(12, 6))
actions = ['Rock', 'Paper', 'Scissors']
for i, action in enumerate(actions):
    plt.plot(np.array(fp_action_probs)[:, i], label=f'FP {action}')
    plt.plot(np.array(rl_action_probs)[:, i], linestyle='--', label=f'RL {action}')

plt.title('Action Probabilities Over Time: Rock-Paper-Scissors')
plt.xlabel('Iterations')
plt.ylabel('Probability')
plt.legend()
plt.show()
