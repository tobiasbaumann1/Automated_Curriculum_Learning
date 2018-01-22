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
N_TOTAL_EPISODES = 20


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

def run_task(idx,agent):
    hallway_length = tasks[idx]
    env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
    return run(env,agent,1)[0]

def run_task_n_times(idx,n,agent):
    hallway_length = tasks[idx]
    env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
    return run(env,agent,n)

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

def train_final_task(agent):    
    agent.reset()
    return run_task_n_times(N_TASKS-1,N_TOTAL_EPISODES,agent), None, None

def train_manual_curriculum(agent):    
    agent.reset()
    rewards, n_trials_list, n_trials = [],[],np.zeros(N_TASKS)
    n_episodes_per_task = int(N_TOTAL_EPISODES/N_TASKS)
    for i in range(N_TASKS):
        rewards.extend(run_task_n_times(i, n_episodes_per_task, agent))
        for _ in range(n_episodes_per_task):
            n_trials[i] += 1
            n_trials_list.append(n_trials.copy())
    return rewards, n_trials_list, None

def train_uniform_sampling(agent):    
    agent.reset()
    rewards, n_trials_list, n_trials = [],[],np.zeros(N_TASKS)
    for _ in range(N_TOTAL_EPISODES):
        idx = np.random.choice(N_TASKS)
        r = run_task(idx,agent)
        rewards.append(r)
        n_trials[idx] += 1
        n_trials_list.append(n_trials.copy())
    return rewards, n_trials_list, None

def init_rewards_buffer(agent):
    rewards_buffer = np.zeros([N_TASKS, BUFFER_SIZE])
    for i in range(N_TASKS):
        rewards_buffer[i] = np.array(run_task_n_times(i, BUFFER_SIZE,agent))
    return rewards_buffer

def train_ACL_bandit(agent, use_slope_as_reward = True):
    agent.reset()
    bandit = Bandit(N_TASKS)    
    rewards, n_trials_list, learning_progress_list = [],[],[]
    rewards_buffer = init_rewards_buffer(agent)
    n_trials = np.zeros(N_TASKS)+BUFFER_SIZE
    for _ in range(N_TOTAL_EPISODES-N_TASKS*BUFFER_SIZE):
        idx = bandit.sample_arm()
        r = run_task(idx,agent)        
        buffer_idx = int(n_trials[idx] % BUFFER_SIZE)
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

def train_ACL_contextual_bandit(agent, use_slope_as_reward= True):
    agent.reset()    
    rewards, n_trials_list, learning_progress_list = [],[],[]
    contextual_bandit = Contextual_Bandit(n_arms=N_TASKS,
                      n_features=N_TASKS*BUFFER_SIZE,
                      str = 'contextual_bandit_net_'+('slope' if use_slope_as_reward else 'diff2mean'),
                      learning_rate=0.01, e_greedy=0.9, e_greedy_increment = 0.005)
    n_trials = np.zeros(N_TASKS)+BUFFER_SIZE
    rewards_buffer = init_rewards_buffer(agent)
    for _ in range(N_TOTAL_EPISODES-N_TASKS*BUFFER_SIZE):
        s = np.hstack(rewards_buffer)
        idx, _ = contextual_bandit.sample_arm(s)
        r = run_task(idx,agent)        
        buffer_idx = int(n_trials[idx] % BUFFER_SIZE)
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

def train_ACL_RL(agent, use_slope_as_reward= True):
    agent.reset()    
    rewards, n_trials_list, learning_progress_list = [],[],[]
    curr_agent = DeepQNetwork(n_actions=N_TASKS,
                      n_features=N_TASKS*BUFFER_SIZE,
                      str = 'rl_curr_learning_net_'+('slope' if use_slope_as_reward else 'diff2mean'),
                      learning_rate=0.01, e_greedy=0.9, e_greedy_increment = 0.005,
                      replace_target_iter=5, memory_size=100)
    n_trials = np.zeros(N_TASKS)+BUFFER_SIZE
    rewards_buffer = init_rewards_buffer(agent)
    for _ in range(N_TOTAL_EPISODES-N_TASKS*BUFFER_SIZE):
        s = np.hstack(rewards_buffer)
        idx = curr_agent.choose_action(s)
        r = run_task(idx,agent)        
        buffer_idx = int(n_trials[idx] % BUFFER_SIZE)
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

def plot_data(data, variant_str):
    if 'n_trials' in variant_str:
        data = np.array(data)
    plt.plot(data)
    plt.xlabel('Episode')
    y_labels = ['rewards','n_trials','learning_progress']
    for y_label in y_labels:
        if y_label in variant_str:
            plt.ylabel(y_label)    
    plt.title(variant_str)
    plt.legend(tasks)
    plt.savefig('./Results/'+variant_str)
    plt.show()

    plt.plot(data)

if __name__ == "__main__":
    agent = DeepQNetwork(n_actions=3,
                      n_features=2,
                      learning_rate=0.01, e_greedy=0.9,
                      replace_target_iter=100, memory_size=2000)
    tasks = np.rint(np.linspace(MIN_HALLWAY_LENGTH,MAX_HALLWAY_LENGTH,N_TASKS))

    str_map = {
        'final_task':'Training on final task only',
        'manual_curriculum':'Training with manual curriculum',
        'uniform_sampling':'Training with uniform sampling',
        'ACL_bandit':'Automated curriculum learning using bandit algorithm',
        'ACL_contextual_bandit':'Automated curriculum learning using contextual bandit algorithm',
        'ACL_RL':'Automated curriculum learning using reinforcement learning',
    }
    results = {}

    for k in str_map:
        print(str_map[k])
        results[k+'_rewards'],results[k+'_n_trials'],results[k+'_learning_progress'] = locals()['train_'+k](agent)

    for k in results:
        if results[k] is not None:
            plot_data(results[k],k)
    
    # print('Automated curriculum learning using bandit algorithm, using difference to mean as reward...')
    # rewards_bandit_acl_diff2mean, n_trials_bandit_acl_diff2mean, learning_progress_bandit_acl_diff2mean = train_with_bandit_ACL(agent, False)
    # print('Automated curriculum learning using reinforcement learning, using difference to mean as reward...')
    # rewards_rl_acl_diff2mean, n_trials_rl_acl_diff2mean, learning_progress_rl_acl_diff2mean = train_with_RL_ACL(agent, False)