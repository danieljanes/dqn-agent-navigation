import torch
from typing_extensions import Final

from unityagents import UnityEnvironment
from navigation.agent import Agent


def main():
    env = UnityEnvironment(file_name="unity_env/Banana.app")
    
    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Reset environment to get state and action sizes
    env_info = env.reset(train_mode=False)[brain_name]
    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    # Initialize agent and load saved weights
    agent = Agent(state_size, action_size, seed=0)
    agent.dqn_policy.load_state_dict(torch.load('weights/checkpoint.pth'))

    # Greedy policy
    epsilon = 0

    # Watch the agent for NUM_EPISODES
    NUM_EPISODES: Final = 2
    for i_episode in range(NUM_EPISODES):
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for _ in range(300):
            action = agent.act(state, epsilon)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            state = next_state
            score += reward
            if done:
                break
        print('\rEpisode {}\t Score: {:.2f}'.format(i_episode, score))
    env.close()


if __name__ == "__main__":
    main()
