import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from Agents import DeepQNetwork
from ACL import Bandit, Contextual_Bandit
from Environments import LongHallway

MIN_HALLWAY_LENGTH = 10
MAX_HALLWAY_LENGTH = 100
N_TASKS = 10
BUFFER_SIZE = 10
MAX_EPISODE_LENGTH = 500
N_TOTAL_EPISODES = 1000


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
                if i % 10 == 0 and i > 0:
                    print('Episode {} done.'.format(i))
                reward_list.append(ep_r)
                break

            observation = observation_
    return reward_list

def calc_slope(Y):
    X = np.arange(len(Y))
    slope, _, _, _, _ = stats.linregress(X, Y)
    return slope

def calc_slope_on_all_tasks(rewards_buffer, n_trials):
    slope = []
    for i in range(N_TASKS):
        buffer_idx = int(n_trials[i] % BUFFER_SIZE)
        slope.append(calc_slope(np.concatenate([rewards_buffer[i,(buffer_idx+1):],rewards_buffer[i,:(buffer_idx+1)]])))
    return np.array(slope)

def train_on_final_task_only(agent):    
    env = LongHallway(MAX_HALLWAY_LENGTH,MAX_EPISODE_LENGTH)
    agent.reset()
    return run(env,agent,N_TOTAL_EPISODES)

def train_with_manual_curriculum(agent):    
    agent.reset()
    rewards = []
    n_trials_list = []
    n_trials = np.zeros(N_TASKS)
    for i in range(N_TASKS):
        hallway_length = tasks[i]
        env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
        n_episodes = int(N_TOTAL_EPISODES/N_TASKS)
        rewards.extend(run(env, agent, n_episodes))
        for _ in range(n_episodes):
            n_trials[i] += 1
            n_trials_list.append(n_trials.copy())
    return rewards, n_trials_list

def train_with_uniform_samplling(agent):    
    agent.reset()
    rewards = []
    n_trials_list = []
    n_trials = np.zeros(N_TASKS)
    for _ in range(N_TOTAL_EPISODES):
        idx = np.random.choice(N_TASKS)
        hallway_length = tasks[idx]
        env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
        r = run(env,agent,1)[0]
        rewards.append(r)
        n_trials[idx] += 1
        n_trials_list.append(n_trials.copy())
    return rewards, n_trials_list

def init_rewards_buffer(agent):
    rewards_buffer = np.zeros([N_TASKS, BUFFER_SIZE])
    for i in range(N_TASKS):
        hallway_length = tasks[i]
        env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
        rewards_buffer[i] = np.array(run(env, agent, BUFFER_SIZE))
    return rewards_buffer

def train_with_bandit_ACL(agent, use_slope_as_reward = True):
    agent.reset()
    bandit = Bandit(N_TASKS)    
    rewards = []
    n_trials_list = []
    learning_progress_list = []
    rewards_buffer = init_rewards_buffer(agent)
    n_trials = np.zeros(N_TASKS)+BUFFER_SIZE
    for _ in range(N_TOTAL_EPISODES-N_TASKS*BUFFER_SIZE):
        idx = bandit.sample_arm()
        hallway_length = tasks[idx]
        env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
        buffer_idx = int(n_trials[idx] % BUFFER_SIZE)
        r = run(env,agent,1)[0]        
        avg_reward_so_far = np.mean(rewards_buffer[idx])
        rewards_buffer[idx][buffer_idx] = r
        rewards.append(r)
        n_trials[idx] += 1
        n_trials_list.append(n_trials.copy())
        if use_slope_as_reward:
            learning_progress = calc_slope(np.concatenate([rewards_buffer[idx,(buffer_idx+1):],rewards_buffer[idx,:(buffer_idx+1)]]))
        else:
            learning_progress = r - avg_reward_so_far
        learning_progress_list.append(learning_progress)
        bandit.update_weights(idx,abs(learning_progress))
    return rewards, n_trials_list, learning_progress_list

def train_with_Contextual_Bandit_ACL(agent, use_slope_as_reward= True):
    agent.reset()    
    rewards = []
    n_trials_list = []
    learning_progress_list = []
    if use_slope_as_reward:
        name = 'contextual_bandit_net_slope'
    else:
        name = 'contextual_bandit_net_diff2mean'
    contextual_bandit = Contextual_Bandit(n_actions=N_TASKS,
                      n_features=N_TASKS*BUFFER_SIZE,
                      str = name,
                      learning_rate=0.01, e_greedy=0.9, e_greedy_increment = 0.005,
                      replace_target_iter=5, memory_size=100)
    n_trials = np.zeros(N_TASKS)+BUFFER_SIZE
    rewards_buffer = init_rewards_buffer(agent)
    for _ in range(N_TOTAL_EPISODES-N_TASKS*BUFFER_SIZE):
        s = np.hstack(rewards_buffer)
        idx = contextual_bandit.sample_arm(s)
        hallway_length = tasks[idx]
        print('Hallway length:', hallway_length)
        env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
        buffer_idx = int(n_trials[idx] % BUFFER_SIZE)
        r = run(env,agent,1)[0]
        avg_reward_so_far = np.mean(rewards_buffer[idx])
        rewards_buffer[idx][buffer_idx] = r
        rewards.append(r)
        n_trials[idx] += 1
        n_trials_list.append(n_trials.copy())
        if use_slope_as_reward:
            learning_progress = calc_slope(np.concatenate([rewards_buffer[idx,(buffer_idx+1):],rewards_buffer[idx,:(buffer_idx+1)]]))
        else:
            learning_progress = r - avg_reward_so_far
        learning_progress_list.append(learning_progress)
        contextual_bandit.learn(s,idx,learning_progress)
    return rewards, n_trials_list, learning_progress_list

def train_with_RL_ACL(agent, use_slope_as_reward= True):
    agent.reset()    
    rewards = []
    n_trials_list = []
    learning_progress_list = []
    if use_slope_as_reward:
        name = 'rl_curr_learning_net_slope'
    else:
        name = 'rl_curr_learning_net_diff2mean'
    curr_agent = DeepQNetwork(n_actions=N_TASKS,
                      n_features=N_TASKS*BUFFER_SIZE,
                      str = name,
                      learning_rate=0.01, e_greedy=0.9, e_greedy_increment = 0.005,
                      replace_target_iter=5, memory_size=100)
    n_trials = np.zeros(N_TASKS)+BUFFER_SIZE
    rewards_buffer = init_rewards_buffer(agent)
    for _ in range(N_TOTAL_EPISODES-N_TASKS*BUFFER_SIZE):
        s = np.hstack(rewards_buffer)
        idx = curr_agent.choose_action(s)
        hallway_length = tasks[idx]
        print('Hallway length:', hallway_length)
        env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
        buffer_idx = int(n_trials[idx] % BUFFER_SIZE)
        r = run(env,agent,1)[0]
        avg_reward_so_far = np.mean(rewards_buffer[idx])
        rewards_buffer[idx][buffer_idx] = r
        rewards.append(r)
        n_trials[idx] += 1
        n_trials_list.append(n_trials.copy())
        s_ = np.hstack(rewards_buffer)
        if use_slope_as_reward:
            learning_progress = calc_slope(np.concatenate([rewards_buffer[idx,(buffer_idx+1):],rewards_buffer[idx,:(buffer_idx+1)]]))
        else:
            learning_progress = r - avg_reward_so_far
        learning_progress_list.append(learning_progress)
        curr_agent.store_transition(s, idx, learning_progress, s_)
        curr_agent.learn()
    return rewards, n_trials_list, learning_progress_list

def plot_data(data, y_label, str):
    plt.plot(data)
    plt.xlabel('Episode')
    plt.ylabel(y_label)
    plt.title(str)
    plt.savefig('./Results/'+str)
    plt.show()

def plot_n_trials(n_trials, y_label, str):
    plt.plot(np.array(n_trials))
    plt.xlabel('Episode')
    plt.ylabel(y_label)    
    plt.title(str)
    plt.legend(tasks)
    plt.savefig('./Results/'+str)
    plt.show()


if __name__ == "__main__":
    agent = DeepQNetwork(n_actions=3,
                      n_features=2,
                      learning_rate=0.01, e_greedy=0.9,
                      replace_target_iter=100, memory_size=2000)
    tasks = np.rint(np.linspace(MIN_HALLWAY_LENGTH,MAX_HALLWAY_LENGTH,N_TASKS))
    #Run without curriculum
    print('Training on final task only...')
    rewards_final_task_only = train_on_final_task_only(agent)

    #Run with manual curriculum
    print('Training with manual curriculum...')
    rewards_manual_curr, n_trials_manual_curr = train_with_manual_curriculum(agent)

    #Run with manual curriculum
    print('Training with uniform sampling...')
    rewards_uniform_sampling, n_trials_uniform_sampling = train_with_uniform_samplling(agent)

    #Run with bandit curriculum
    print('Automated curriculum learning using bandit algorithm, using slope as reward...')
    rewards_bandit_acl_slope, n_trials_bandit_acl_slope, learning_progress_bandit_acl_slope = train_with_bandit_ACL(agent, True)
    print('Automated curriculum learning using bandit algorithm, using difference to mean as reward...')
    rewards_bandit_acl_diff2mean, n_trials_bandit_acl_diff2mean, learning_progress_bandit_acl_diff2mean = train_with_bandit_ACL(agent, False)

    #Run with RL curriculum
    print('Automated curriculum learning using reinforcement learning, using slope as reward...')
    rewards_rl_acl_slope, n_trials_rl_acl_slope, learning_progress_rl_acl_slope = train_with_RL_ACL(agent, True)
    print('Automated curriculum learning using reinforcement learning, using difference to mean as reward...')
    rewards_rl_acl_diff2mean, n_trials_rl_acl_diff2mean, learning_progress_rl_acl_diff2mean = train_with_RL_ACL(agent, False)

    plot_data(rewards_final_task_only,'Reward','rewards_final_task_only')
    plot_data(rewards_manual_curr,'Reward','rewards_manual_curr')
    plot_data(rewards_uniform_sampling,'Reward','rewards_uniform_sampling')
    plot_data(rewards_bandit_acl_slope,'Reward','rewards_bandit_acl_slope')
    plot_data(rewards_rl_acl_slope,'Reward','rewards_rl_acl_slope')
    plot_data(rewards_bandit_acl_diff2mean,'Reward','rewards_bandit_acl_diff2mean')
    plot_data(rewards_rl_acl_diff2mean,'Reward','rewards_rl_acl_diff2mean')

    plot_n_trials(n_trials_manual_curr,'Number of trials','n_trials_manual_curr')
    plot_n_trials(n_trials_uniform_sampling,'Number of trials','n_trials_uniform_sampling')
    plot_n_trials(n_trials_bandit_acl_slope,'Number of trials','n_trials_bandit_acl_slope')
    plot_n_trials(n_trials_bandit_acl_diff2mean,'Number of trials','n_trials_bandit_acl_diff2mean')
    plot_n_trials(n_trials_rl_acl_slope,'Number of trials','n_trials_rl_acl_slope')
    plot_n_trials(n_trials_rl_acl_diff2mean,'Number of trials','n_trials_rl_acl_diff2mean')

    plot_data(learning_progress_bandit_acl_slope,'Learning progress','learning_progress_bandit_acl_slope')
    plot_data(learning_progress_bandit_acl_diff2mean,'Learning progress','learning_progress_bandit_acl_diff2mean')
    plot_data(learning_progress_rl_acl_slope,'Learning progress','learning_progress_rl_acl_slope')
    plot_data(learning_progress_rl_acl_diff2mean,'Learning progress','learning_progress_rl_acl_diff2mean')

