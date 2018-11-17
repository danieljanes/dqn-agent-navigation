# Report

## Algorithm

### Model

The model is a simple fully-connected neural network consisting of two hidden layers with 64 units each using ReLU activation functions [[Nair & Hinton](http://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)]. The input layer has 37 units corresponding to the dimension of the input vector. The output layer has four units corresponding to the four actions described in [README.md](README.md). It uses a linear activation function. One could also use softmax, however since our agent either chooses the action with the maximum Q-value or a random action there is no benefit in having all Q-values add up to 1.

The following code snippts show the implementation of this model in PyTorch:

```
    def __init__(self):
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_n = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc_n(x)
```

### Training Algorithm

We train our model using the Deep Q-Learning algorithm [[Mnih et al., 2015](https://www.nature.com/articles/nature14236)]. To avoid issues with overestimation of action values and to stabilize training we combine it with Double Q-learning [[Hasselt et al., 2016](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12389/11847)].

**Experience Replay** helps to learn from single experiences multiple times and (more importantly) breaks the correlation between subsequent experiences.

**Fixed Q-Targets** stabilize the training process by using fixed parameters to calculate target values during training.

**Exploration/Exploitation** is controlled using parameter espilon. Epsilon slowly decreases from 1.0 to 0.01 after every episode.

**Policy Refinement** happens after every episode. This can be customized by setting the value of `UPDATE_EVERY` to values `>1`.

**Double Q-learning** avoids overestimation of action values. This especially helps during early stages of training when the Q-values proposed by the model are not representative of the underlying Markov Process yet. By using separate networks for choosing an action and evaluating an action the training becomes more stable and converges faster. One can compare training with and without Double Q-learning by setting the boolean `DOUBLE_Q_LEARNING` (in file [agent.py](src/navigation/agent.py)) to `True` or `False` respectively.

### Training Runs

Console logs from first converging training run:

```
Episode 100	Average Score: 0.271
Episode 200	Average Score: 2.98
Episode 300	Average Score: 6.50
Episode 400	Average Score: 8.99
Episode 500	Average Score: 12.13
Episode 550	Average Score: 13.03
Environment solved in 450 episodes!	Average Score: 13.03
```

Console logs from saved model training run:

```
Episode 100	Average Score: 1.066
Episode 200	Average Score: 3.78
Episode 300	Average Score: 6.80
Episode 400	Average Score: 8.68
Episode 500	Average Score: 11.34
Episode 600	Average Score: 12.09
Episode 628	Average Score: 13.07
Environment solved in 528 episodes!	Average Score: 13.07
```

The saved model can be found in `weights/checkpoint.pth`. To watch an agent interact with the environment using the saved model weights and a greedy policy, please run:

```
pipenv run python3 src/navigation/watch.py
```

## Further Ideas

Several improvement could help to yield faster convergence or more stable training:

- Use Prioritized Experience Replay [[Schaul et al., 2015](https://arxiv.org/abs/1511.05952)] instead of uniform random sampling to improve learning efficiency.
- Use Dueling Network Architectures [[Wang et al., 2015](https://arxiv.org/abs/1511.06581)] to better generalize learning across different actions.
- Combine several improvements to the plain DQN algrorithm as shown by DeepMind's Rainbow paper [[Hessel et al., 2017](https://deepmind.com/research/publications/rainbow-combining-improvements-deep-reinforcement-learning/)].
- Explore usage of policy-based methods. This could offer profound advantages: When running the agent using the committed weights and a greedy policy one can observe an interesting behaviour. The agent seems to be doing well for some amount of time until it gets stuck oscillating between two states until the end of an episode. Without the randomness introduced by an epsilon-greedy policy this is similar to an endless loop where e.g. in state A action "turn right" and in state B action "turn left" each have the highest Q-value thus resulting in an agent which cannot progress beyond those states until the episode is over.
