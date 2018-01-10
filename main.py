#import gym
from Agents import DeepQNetwork
from Environments import LongHallway

env = LongHallway(5,1000)
agent = DeepQNetwork(n_actions=env.n_actions,
                  n_features=env.n_features,
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=2000)

total_steps = 0


for i_episode in range(100):

    observation = env.reset()
    ep_r = 0
    while True:
        action = agent.choose_action(observation)

        observation_, reward, done = env.step(action)

        agent.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            agent.learn()

        if done:
            print('episode: ', i_episode,
                  'ep_r: ', round(ep_r, 2),
                  ' epsilon: ', round(agent.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

agent.plot_cost()