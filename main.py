import numpy as np
from Agents import DeepQNetwork
from ACL import Bandit
from Environments import LongHallway

MIN_HALLWAY_LENGTH = 10
MAX_HALLWAY_LENGTH = 100
N_TASKS = MAX_HALLWAY_LENGTH - MIN_HALLWAY_LENGTH +1
MAX_EPISODE_LENGTH = 500
N_TOTAL_EPISODES = 100


def run(env,agent,n_episodes):
    reward_list = []
    for i in range(n_episodes):
        observation = env.reset()
        ep_r = 0
        while True:
            action = agent.choose_action(observation)

            observation_, reward, done = env.step(action)

            agent.store_transition(observation, action, reward, observation_)

            ep_r += reward
            agent.learn()

            if done:
                print('Episode: ', i, ' done.'
                      ' Reward: ', round(ep_r, 2))
                reward_list.append(ep_r)
                break

            observation = observation_
    return reward_list

if __name__ == "__main__":
    env = LongHallway(MAX_HALLWAY_LENGTH,MAX_EPISODE_LENGTH)
    agent = DeepQNetwork(n_actions=env.n_actions,
                      n_features=env.n_features,
                      learning_rate=0.01, e_greedy=0.9,
                      replace_target_iter=100, memory_size=2000)
    #Run without curriculum
    reward_list = run(env,agent,N_TOTAL_EPISODES)

    #Run with manual curriculum
    agent.reset()
    for hallway_length in np.rint(np.linspace(MIN_HALLWAY_LENGTH,MAX_HALLWAY_LENGTH,N_TOTAL_EPISODES/10)):
        print('Hallway length', hallway_length)
        #agent.flush_memory()
        env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
        run(env, agent, 10)

    #Run with bandit curriculum
    agent.reset()
    bandit = Bandit(N_TASKS)
    for i in range(N_TOTAL_EPISODES):
        idx = bandit.sample_arm()
        hallway_length = MIN_HALLWAY_LENGTH + idx
        print('Hallway length', hallway_length)
        env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
        r = run(env,agent,1)[0]
        bandit.update_weights(idx,r)


    #Run with RL curriculum
