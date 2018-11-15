# Deep Reinforcement Learning: Navigation Agent

The presented agent leverages Q-learning using a Deep Q-Network (DQN) [[Mnih et al., 2015](https://www.nature.com/articles/nature14236)], along with improvements such as Double Q-learning [[Hasselt et al., 2016](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12389/11847)] to improve training stability.

## Project Environment Details

The agent must navigate the given environment with the goal of collecting as many yellow bananas as possible whilst avoiding blue bananas. Collecting yellow bananas yields a reward of `+1`, collecting blue bananas yields a return of `-1`. The environment is based on the [Unity ML-Agents Toolkit](https://github.com/Unity-Technologies/ml-agents).

![Agent Environment](https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif)

GIF source: [Udacity](https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif)

### State Space

States are represented using 37-dimensional vectors containing information about the agent's velocity and ray-based perception of objects around the agent's forward direction [[Source](https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation)]. As an example please see the following state:

```
[1.         0.         0.         0.         0.84408134 0.
 0.         1.         0.         0.0748472  0.         1.
 0.         0.         0.25755    1.         0.         0.
 0.         0.74177343 0.         1.         0.         0.
 0.25854847 0.         0.         1.         0.         0.09355672
 0.         1.         0.         0.         0.31969345 0.
 0.        ]
```

### Action Space

An agent can perform the following four discrete actions:

- `0`: Move forward
- `1`: Move backward
- `2`: Move left
- `3`: Move right

### Solving the environment

This environment is considered to be solved when the agent achieves an average score of `+13` over `100` consecutive episodes. An indication given by Udacity mentions that one should aim for solving the environment in less than `1800` episodes. The committed weights (in [weights/checkpoint.pth](weights/checkpoint.pth)) resulted from a training run which converged after `543` episodes with an average reward of `13.06`. Details about the agent's architecture are given in [Report.md](Report.md).

## Project Setup

### Prerequisites

This project requires Python 3.6 in a virtual environment using Pipenv:

- [Python 3.6](https://www.python.org/downloads/)
- [Pipenv](https://github.com/pypa/pipenv)

To verify that the necessary prerequisites are installed run the following commands:

```
$ python3 --version
Python 3.6.5

$ pipenv --version
pipenv, version 2018.7.1
```

### Initial Setup

Setting up the project consists of cloning the repository, unpacking the environment, and installing the dependencies:

```
# clone repository
git clone git@github.com:danieljanes/drlnd-project-1-navigation.git
cd dqn-navigation

# unzip environment
unzip -a unity_env/Banana.app.zip -d unity_env

# install dependencies
pipenv run pip install -U pip==18.0  # use compatible version of pip inside this virtual environment
pipenv install -e .
```

To verify your setup please run:

```
pipenv run python3 src/navigation/verify.py
```

## Run Project

### Train Agent

```
pipenv run python3 src/navigation/train.py
```

If training converges then weights get saved to a file following the naming convention `weights/checkpoint_yyy-mm-ddTHH-MM-SSZ.pth` such that subsequent training runs do not override previous training results.

### Watch Trained Agent

```
pipenv run python3 src/navigation/watch.py
```

By default this uses the saved and committed weights. If different weights from other training runs should be used one can adjust the path where PyTorch reads the weights from in `src/navigation/watch.py`.

## Report

The project report resides in [Report.md](Report.md).
