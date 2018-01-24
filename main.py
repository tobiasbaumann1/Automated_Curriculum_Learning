import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from Agents import DeepQNetwork
from ACL import Bandit, Contextual_Bandit
from Environments import LongHallway
from enum import Enum

MIN_HALLWAY_LENGTH = 10
MAX_HALLWAY_LENGTH = 100
N_TASKS = 10
FINAL_TASK_INDEX = N_TASKS - 1
BUFFER_SIZE = 10
MAX_EPISODE_LENGTH = 500
N_TOTAL_EPISODES = 200
PERFORMANCE_TEST_INTERVAL = 50
N_TRAJECTORIES = 3

class Variant(Enum):
    FINAL_TASK = 'Training on final task only'
    MANUAL_CURRICULUM = 'Training with manual curriculum'
    UNIFORM_SAMPLING = 'Training with uniform sampling'
    ACL_BANDIT = 'Automated curriculum learning using bandit algorithm'
    ACL_CONTEXTUAL_BANDIT = 'Automated curriculum learning using contextual bandit algorithm'
    ACL_RL = 'Automated curriculum learning using reinforcement learning'

def run_task(idx,agent,learning_active = True):
    hallway_length = tasks[idx]
    env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
    return env.run(agent,1,learning_active)[0]

def run_task_n_times(idx,n,agent,learning_active = True):
    hallway_length = tasks[idx]
    env = LongHallway(hallway_length,MAX_EPISODE_LENGTH)
    return env.run(agent,n,learning_active)

def test_performance(agent, n_test_episodes = 10):
    return np.mean(run_task_n_times(FINAL_TASK_INDEX,n_test_episodes,agent,learning_active = False))

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

def is_automated_curriculum_learning(variant):
    return variant == Variant.ACL_BANDIT or variant == Variant.ACL_CONTEXTUAL_BANDIT or variant == Variant.ACL_RL

def init_rewards_buffer(agent):
    rewards_buffer = np.zeros([N_TASKS, BUFFER_SIZE])
    for i in range(N_TASKS):
        rewards_buffer[i] = np.array(run_task_n_times(i, BUFFER_SIZE,agent,learning_active = False))
    return rewards_buffer

def init_teacher(variant, name = ''):
    if variant == Variant.ACL_BANDIT:
        return Bandit(N_TASKS)
    if variant == Variant.ACL_CONTEXTUAL_BANDIT:
        return Contextual_Bandit(n_arms=N_TASKS,
                      n_features=N_TASKS*BUFFER_SIZE,
                      str = 'contextual_bandit_net' + name,
                      learning_rate=0.01, e_greedy=0.9, e_greedy_increment = 0.005)
    if variant == Variant.ACL_RL:
        return DeepQNetwork(n_actions=N_TASKS,
                      n_features=N_TASKS*BUFFER_SIZE,
                      str = 'rl_curr_learning_net' + name,
                      learning_rate=0.01, e_greedy=0.9, e_greedy_increment = 0.005,
                      replace_target_iter=5, memory_size=100)

def train(student,variant,teacher = None, use_slope_as_reward = True):
    student.reset()
    rewards,n_trials_list,learning_progress_list,test_performance_list,n_trials = [],[],[],[],np.zeros(N_TASKS)
    if is_automated_curriculum_learning(variant):
        if teacher is None:
            teacher = init_teacher(variant)
        rewards_buffer = init_rewards_buffer(student)
    for i in range(N_TOTAL_EPISODES):
        if i % PERFORMANCE_TEST_INTERVAL == 0:
            test_performance_list.append(test_performance(student))
        if variant == Variant.FINAL_TASK:
            idx = FINAL_TASK_INDEX
        if variant == Variant.MANUAL_CURRICULUM:
            idx = int((N_TASKS*i)/N_TOTAL_EPISODES)
        if variant == Variant.UNIFORM_SAMPLING:
            idx = np.random.choice(N_TASKS)
        if is_automated_curriculum_learning(variant):
            s = np.hstack(rewards_buffer)
            idx = teacher.choose_action(s)
            buffer_idx = int(n_trials[idx] % BUFFER_SIZE)
            avg_reward_so_far = np.mean(rewards_buffer[idx])
        r = run_task(idx,student)            
        rewards.append(r)
        n_trials[idx] += 1
        n_trials_list.append(n_trials.copy())    
        if is_automated_curriculum_learning(variant):
            rewards_buffer[idx][buffer_idx] = r
            if use_slope_as_reward:
                learning_progress = calc_slope(np.concatenate([rewards_buffer[idx,(buffer_idx+1):],rewards_buffer[idx,:(buffer_idx+1)]]))
            else:
                learning_progress = r - avg_reward_so_far
            learning_progress_list.append(learning_progress)

            if variant == Variant.ACL_BANDIT:
                teacher.update_weights(idx,abs(learning_progress))
            if variant == Variant.ACL_CONTEXTUAL_BANDIT:
                teacher.learn(s,idx,learning_progress)
            if variant == Variant.ACL_RL:
                s_ = np.hstack(rewards_buffer)
                teacher.store_transition(s, idx, learning_progress, s_)
                teacher.learn()
    return rewards, n_trials_list, learning_progress_list, test_performance_list

def plot_data(data, variant_str):
    if data:
        plt.figure()
        if 'n_trials' in variant_str:
            data = np.array(data)
            plt.legend(tasks)
        if 'one_shot' in variant_str:
            #data = np.transpose(np.array(data))
            x = np.arange(0,N_TOTAL_EPISODES,step = PERFORMANCE_TEST_INTERVAL)
            for trajectory in data:
                plt.plot(x,trajectory)
            plt.legend(range(N_TRAJECTORIES))
        plt.xlabel('Episode')
        y_labels = ['rewards','n_trials','learning_progress','test_performance']
        for y_label in y_labels:
            if y_label in variant_str:
                plt.ylabel(y_label)    
        plt.title(variant_str)
        plt.savefig('./Results/'+variant_str)

def train_all_variants(student):
    results = {}
    for variant in Variant:
        description = variant.value
        print(description)
        # call corresponding training function and save results
        results[description+'_rewards'],results[description+'_n_trials'],\
        results[description+'_learning_progress'],results[description+'_test_performance'] = train(student,variant)
    return results

def train_perfect_curriculum(student, n_trajectories):
    results = {}
    for variant in filter(is_automated_curriculum_learning, Variant):
        description = variant.value
        print('Finding perfect curriculum: ', description)
        teacher = init_teacher(variant,'perfect_curriculum')
        test_performance_trajectories = []
        # call corresponding training function n_trajectories times
        for _ in range(n_trajectories):
            _,_,_,test_performance_list = train(student,variant,teacher = teacher)
            print(test_performance_list)
            test_performance_trajectories.append(test_performance_list)
        print(test_performance_trajectories)        
        results[description+'_one_shot'+'_test_performance'] = test_performance_trajectories
    return results    

def train_curriculum_many_students(students):
    results = {}
    for variant in filter(is_automated_curriculum_learning, Variant):
        description = variant.value
        print('Finding perfect curriculum: ', description)
        teacher = init_teacher(variant,'perfect_curriculum')
        test_performance_trajectories = []
        for student in students:
            # call training function
            _,_,_,test_performance_list = train(student,variant,teacher = teacher)
            test_performance_trajectories.append(test_performance_list)
        
        results[description+'_many_students'+'_test_performance'] = test_performance_trajectories
    return results    

if __name__ == "__main__":
    student = DeepQNetwork(n_actions=3, n_features=2, learning_rate=0.01, 
        e_greedy=0.9, replace_target_iter=100, memory_size=2000)
    tasks = np.rint(np.linspace(MIN_HALLWAY_LENGTH,MAX_HALLWAY_LENGTH,N_TASKS))

    results = {}
    #results.update(train_all_variants(student))
    results.update(train_perfect_curriculum(student, N_TRAJECTORIES))
    for k in results:
        plot_data(results[k],k)
    
