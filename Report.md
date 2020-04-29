# Report for Tennis

## State and action space, rewards in environment

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of **+0.1**. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of **-0.01**. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of **8** variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, agents must get an average score of **+0.5** (over 100 consecutive episodes, after taking the maximum over both agents). In particular,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.

The environment is considered solved, when the **average (over 100 episodes)** of those scores is at least **+0.5**.

## Learning algorithm

The agents training is conducted using `ddpg` function in the `Tennis.ipynb` notebook. Agents are trained episodically until `n_episodes` is reached or until the environment is solved. Each episode continues until `max_t` time-steps is reached or until the environment is done.

Agent class is contained in `agents.py` file and neural networks for actor and critic defined in `actor_critic.py` file.

Additionaly Agent used following mechanisms:
* Replay buffer of experiences
* Soft targets with specified "softness"
* Discounted rewards
* Noise generation with Ornstein-Uhlenbeck process
* Critic's weights decay
* Delayed updates with specified update period and number of updates per cycle

DDPG Agent Hyper Parameters:
- BUFFER_SIZE (int): replay buffer size
- BATCH_SIZE (int): mini batch size
- GAMMA (float): discount factor
- TAU (float): for soft update of target parameters
- LR_ACTOR (float): learning rate for Actor
- LR_CRITIC (float): learning rate for Critic
- WEIGHT_DECAY (float): weight decay for Critic's optimizer
- UPDATE_STEP (int): specify updates delay
- N_UPDATES (int): specify no. of updates per update cycle

Additionally, `theta=0.15` and `sigma=0.15` are set fixed for Ornstein-Uhlenbeck process.

`BUFFER_SIZE = int(1e6)`, `BATCH_SIZE = 512`, `GAMMA = 0.99`, `TAU = 0.001`, `LR_ACTOR = 0.0001`, `LR_CRITIC = 0.001`, `WEIGHT_DECAY = 0`, `UPDATE_STEP = 4` and `N_UPDATES = 1`

Used hyperparameters values were found by trial and error until satisfactory results were reached. `UPDATE_STEP` turned out to be key hyperparameter to reach the goal quickly. Also different `N_UPDATES` values were tested but no concrete improvement was found for values higher than 1. Introducing weight decay seems to destabilize steady growth of reward, so it was set to 0.

### DDPG Hyper Parameters  

- n_episodes (int): maximum number of training episodes
- max_t (int): maximum number of timesteps per episode

Where
`n_episodes=5000` and `max_t=1500`

Used hyperparameters values were found by trial and error until satisfactory results were reached. `max_t` was increased from 1000 after introducing delayed updates mechanic and seems to provide quite nice results.

### Neural Networks
In used Multi-Agent DDPG algorithm two deep neural networks are used and are characterised by following architectures:
- Actor    
    - Hidden 1: (24, 512)   - ReLU
    - Hidden 2: (512, 256)  - ReLU
    - Output: (256, 4)      - TanH

- Critic
    - Hidden 1: (24, 512)   - ReLU
    - Hidden 2: (536, 256)  - ReLU
    - Output: (256, 1)      - Linear


Used architecture is the same as in [`ddpg-pendulum/DDPG.ipynb`](https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/DDPG.ipynb) except for number of neurons in both hidden layers of critic and actor.

## Plot of average score

![Score](./img/Scores.png)

```
Episode 100	Average Score: 0.00	Max score: 0.00
Episode 200	Average Score: 0.00	Max score: 0.00
Episode 300	Average Score: 0.00	Max score: 0.10
Episode 400	Average Score: 0.02	Max score: 0.10
Episode 500	Average Score: 0.02	Max score: 0.09
Episode 600	Average Score: 0.04	Max score: 0.00
Episode 700	Average Score: 0.04	Max score: 0.20
Episode 800	Average Score: 0.18	Max score: 2.60
Environment solved in 835 episodes!	Average Score: 0.50
```

## Ideas for improvement

Currently agent are trained on the same actor-critic networks. It may be good idea to train them independently as seperate agents and compare results to proposed solution.

