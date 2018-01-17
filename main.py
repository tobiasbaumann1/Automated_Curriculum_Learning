import numpy as np
from scipy import stats
from Agents import DeepQNetwork
from ACL import Bandit
from Environments import LongHallway

MIN_HALLWAY_LENGTH = 10
MAX_HALLWAY_LENGTH = 100
N_TASKS = 10
BUFFER_SIZE = 5
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
                print('Episode done.'
                      ' Reward: ', round(ep_r, 2))
                reward_list.append(ep_r)
                break

            observation = observation_
    return reward_list

def calc_slope(Y):
    X = np.arange(len(Y))
    slope, _, _, _, _ = stats.linregress(X, Y)
    return slope

def calc_slope_on_all_tasks(rewards_buffer, trial_count):
    slope = []
    for i in range(N_TASKS):
        buffer_idx = int(trial_count[i] % BUFFER_SIZE)
        slope.append(calc_slope(np.concatenate([rewards_buffer[i,(buffer_idx+1):],rewards_buffer[i,:(buffer_idx+1)]])))
    return np.array(slope)

if __name__ == "__main__":
    env = LongHallway(MAX_HALLWAY_LENGTH,MAX_EPISODE_LENGTH)
    agent = DeepQNetwork(n_actions=env.n_actions,
                      n_features=env.n_features,
                      learning_rate=0.01, e_greedy=0.9,
                      replace_target_iter=100, memory_size=2000)
    tasks = np.rint(np.linspace(MIN_HALLWAY_LENGTH,MAX_HALLWAY_LENGTH,N_TASKS))
    # #Run without curriculum
    # reward_list = run(env,agent,N_TOTAL_EPISODES)

    # #Run with manual curriculum
    # agent.reset()
    # for hallway_length in np.rint(np.linspace(MIN_HALLWAY_LENGTH,MAX_HALLWAY_LENGTH,N_TOTAL_EPISODES/10)):
    #     print('Hallway length', hallway_length)
    #     #agent.flush_memory()
    #     env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
    #     run(env, agent, 10)


    #Run with bandit curriculum
    # agent.reset()
    # bandit = Bandit(N_TASKS)
    # rewards_buffer = np.zeros([N_TASKS, BUFFER_SIZE])
    # trial_count = np.zeros(N_TASKS)
    # for i in range(N_TASKS):
    #     hallway_length = tasks[i]
    #     env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
    #     rewards_buffer[i] = np.array(run(env, agent, BUFFER_SIZE))
    # for _ in range(N_TOTAL_EPISODES-N_TASKS*BUFFER_SIZE):
    #     idx = bandit.sample_arm()
    #     hallway_length = tasks[idx]
    #     print('Hallway length:', hallway_length)
    #     env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
    #     buffer_idx = int(trial_count[idx] % BUFFER_SIZE)
    #     rewards_buffer[idx][buffer_idx] = run(env,agent,1)[0]
    #     trial_count[idx] += 1
    #     learning_progress = calc_slope(np.concatenate([rewards_buffer[idx,(buffer_idx+1):],rewards_buffer[idx,:(buffer_idx+1)]]))    
    #     print('Learning progress:', learning_progress)
    #     bandit.update_weights(idx,learning_progress)

    #Run with RL curriculum
    agent.reset()
    curr_agent = DeepQNetwork(n_actions=N_TASKS,
                      n_features=N_TASKS,
                      str = 'meta_net',
                      learning_rate=0.01, e_greedy=0.4,
                      replace_target_iter=5, memory_size=2000)
    rewards_buffer = np.zeros([N_TASKS, BUFFER_SIZE])
    trial_count = np.zeros(N_TASKS)
    for i in range(N_TASKS):
        hallway_length = tasks[i]
        env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
        rewards_buffer[i] = np.array(run(env, agent, BUFFER_SIZE))
    for _ in range(N_TOTAL_EPISODES-N_TASKS*BUFFER_SIZE):
        s = calc_slope_on_all_tasks(rewards_buffer, trial_count)
        idx = curr_agent.choose_action(s)
        hallway_length = tasks[idx]
        print('Hallway length:', hallway_length)
        env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
        buffer_idx = int(trial_count[idx] % BUFFER_SIZE)
        rewards_buffer[idx][buffer_idx] = run(env,agent,1)[0]
        trial_count[idx] += 1
        s_ = calc_slope_on_all_tasks(rewards_buffer, trial_count)
        learning_progress = s_[idx]
        print('Learning progress:', learning_progress)
        curr_agent.store_transition(s, idx, learning_progress, s_)
        curr_agent.learn()

