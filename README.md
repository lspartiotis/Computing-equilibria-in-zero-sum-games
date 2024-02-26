# Computing Equilibria in Zero-Sum Games

This project explores strategies and computes equilibria in various classic games using Fictitious Play (FP) and Reinforcement Learning (RL) approaches.

## Overview of Python Scripts

### Rock Paper Scissors Simulations
- **`nashfp-rlminmax.py`**
  - **Game**: Rock Paper Scissors
  - **Agents**: Fictitious Play vs. RL Minimax
  - **Plots/Analysis**: Nash equilibria between FP and RL Minimax strategies

- **`mmrlvsmmrlnash.py`**
  - **Game**: Rock Paper Scissors
  - **Agents**: Minimax RL vs. Minimax RL
  - **Plots/Analysis**: Nash equilibria for competing Minimax RL agents

- **`probofactionminvssarsa.py`**
  - **Game**: Rock Paper Scissors
  - **Agents**: Minimax vs. SARSA
  - **Plots/Analysis**: Action probabilities highlighting strategic adjustments

- **`winlossminmaxvsfp.py`**
  - **Game**: Rock Paper Scissors
  - **Agents**: Minimax vs. Fictitious Play
  - **Plots/Analysis**: Win/loss ratios showcasing performance differences

### Prisoner's Dilemma Simulations
- **`prob_minvsfp.py`**
  - **Game**: Prisoner's Dilemma
  - **Agents**: Minimax vs. Fictitious Play
  - **Plots/Analysis**: Action probabilities illustrating strategy adaptation

- **`fpvsfpprobofaction.py`**
  - **Game**: Prisoner's Dilemma
  - **Agents**: Fictitious Play vs. Fictitious Play
  - **Plots/Analysis**: Action probabilities for FP agents showing strategy convergence

- **`fpvsfpnash.py`**
  - **Game**: Prisoner's Dilemma
  - **Agents**: Fictitious Play vs. Fictitious Play
  - **Plots/Analysis**: Nash equilibria for FP vs. FP interactions

### Battle of the Sexes Simulations
- **`prob.py`** (Battle of the Sexes)
  - **Game**: Battle of the Sexes
  - **Agents**: General utility for probability computation
  - **Plots/Analysis**: Action probabilities, used in various contexts within the game

- **`fpvsfpactions.py`**
  - **Game**: Battle of the Sexes
  - **Agents**: Fictitious Play vs. Fictitious Play
  - **Plots/Analysis**: Evolution of action choices in FP vs. FP setup

- **`fpvsfpnash.py`**
  - **Game**: Battle of the Sexes
  - **Agents**: Fictitious Play vs. Fictitious Play
  - **Plots/Analysis**: Computing and analyzing Nash equilibria for FP vs. FP
