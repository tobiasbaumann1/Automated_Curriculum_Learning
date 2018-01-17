import numpy as np
from scipy import stats
from Agents import DeepQNetwork
from ACL import Bandit
from Environments import LongHallway

MIN_HALLWAY_LENGTH = 10
MAX_HALLWAY_LENGTH = 100
N_TASKS = 10
BUFFER_SIZE = 10
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

def calc_slope_on_all_tasks(rewards_buffer, trial_count):
    slope = []
    for i in range(N_TASKS):
        buffer_idx = int(trial_count[i] % BUFFER_SIZE)
        slope.append(calc_slope(np.concatenate([rewards_buffer[i,(buffer_idx+1):],rewards_buffer[i,:(buffer_idx+1)]])))
    return np.array(slope)

def train_on_final_task_only(agent):    
    env = LongHallway(MAX_HALLWAY_LENGTH,MAX_EPISODE_LENGTH)
    agent.reset()
    return run(env,agent,N_TOTAL_EPISODES)

def train_with_manual_curriculum(agent):    
    agent.reset()
    rewards = []
    task_labels = []
    for hallway_length in tasks:
        env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
        n_episodes = int(N_TOTAL_EPISODES/N_TASKS)
        rewards.append(run(env, agent, n_episodes))
        task_labels.append([hallway_length]*n_episodes)
    return rewards, task_labels

def train_with_bandit_ACL(agent):
    agent.reset()
    bandit = Bandit(N_TASKS)    
    rewards = []
    task_labels = []
    learning_progress_list = []
    rewards_buffer = np.zeros([N_TASKS, BUFFER_SIZE])
    trial_count = np.zeros(N_TASKS)+BUFFER_SIZE
    for i in range(N_TASKS):
        hallway_length = tasks[i]
        env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
        rewards_buffer[i] = np.array(run(env, agent, BUFFER_SIZE))
    for _ in range(N_TOTAL_EPISODES-N_TASKS*BUFFER_SIZE):
        idx = bandit.sample_arm()
        task_labels.append(idx)
        hallway_length = tasks[idx]
        env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
        buffer_idx = int(trial_count[idx] % BUFFER_SIZE)
        r = run(env,agent,1)[0]
        rewards_buffer[idx][buffer_idx] = r
        rewards.append(r)
        trial_count[idx] += 1
        learning_progress = calc_slope(np.concatenate([rewards_buffer[idx,(buffer_idx+1):],rewards_buffer[idx,:(buffer_idx+1)]]))    
        learning_progress_list.append(learning_progress)
        bandit.update_weights(idx,learning_progress)
    return rewards, task_labels, learning_progress_list

def train_with_RL_ACL(agent):
    agent.reset()    
    rewards = []
    task_labels = []
    learning_progress_list = []
    curr_agent = DeepQNetwork(n_actions=N_TASKS,
                      n_features=N_TASKS,
                      str = 'meta_net',
                      learning_rate=0.01, e_greedy=0.7,
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
        task_labels.append(idx)
        hallway_length = tasks[idx]
        print('Hallway length:', hallway_length)
        env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
        buffer_idx = int(trial_count[idx] % BUFFER_SIZE)
        r = run(env,agent,1)[0]
        rewards_buffer[idx][buffer_idx] = r
        rewards.append(r)
        trial_count[idx] += 1
        s_ = calc_slope_on_all_tasks(rewards_buffer, trial_count)
        learning_progress = s_[idx]
        learning_progress_list.append(learning_progress)
        print('Learning progress:', learning_progress)
        curr_agent.store_transition(s, idx, learning_progress, s_)
        curr_agent.learn()
    return rewards, task_labels, learning_progress_list

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
    rewards_manual_curr, task_labels_manual_curr = train_with_manual_curriculum(agent)

    #Run with bandit curriculum
    print('Automated curriculum learning using bandit algorithm...')
    rewards_bandit_acl, task_labels_bandit_acl, learning_progress_bandit_acl = train_with_bandit_ACL(agent)

    #Run with RL curriculum
    print('Automated curriculum learning using reinforcement learning...')
    rewards_rl_acl, task_labels_rl_acl, learning_progress_rl_acl = train_with_RL_ACL(agent)

