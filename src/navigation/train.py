from collections import deque
import datetime

import torch
from typing_extensions import Final
import numpy as np

from unityagents import UnityEnvironment
from navigation.agent import Agent


EPISODES: Final = 2000
EPSILON_START: Final = 1.0
EPSILON_END: Final = 0.01
EPSILON_DECAY: Final = 0.995


def main():
    env = UnityEnvironment(file_name="unity_env/Banana.app")
    
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    
    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)
    
    # examine the state space
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    agent = Agent(state_size, action_size, seed=0)

    # Training
    scores = []
    scores_window = deque(maxlen=100)
    epsilon = EPSILON_START
    for i_episode in range(1, EPISODES+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for _ in range(300):
            action = agent.act(state, epsilon)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        # Decrease epsilon
        epsilon = max(EPSILON_END, epsilon*EPSILON_DECAY)
        # Record score
        scores.append(score)
        scores_window.append(score)
        # Logging
        mean_score = np.mean(scores_window)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, mean_score))
        if mean_score >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, mean_score))
            utc_datetime = datetime.datetime.utcnow()
            dt = utc_datetime.strftime("%Y-%m-%dT%H:%M:%SZ")
            torch.save(agent.dqn_policy.state_dict(), 'weights/checkpoint_' + dt + '.pth')
            break
    env.close()


if __name__ == "__main__":
    main()
