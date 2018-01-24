import numpy as np

class Environment(object):
    def __init__(self, N_ACTIONS, N_FEATURES, MAX_EPISODE_LENGTH):
        self.n_actions = N_ACTIONS
        self.n_features = N_FEATURES
        self.max_episode_length = MAX_EPISODE_LENGTH
        self.step_ctr = 0
        self.ep_ctr = 0
        self.actions_list = []

    def step(self, a):
        self.update_state(a)
        self.actions_list.append(a)
        r = self.calculate_reward(a)
        self.step_ctr += 1
        return self.state_to_observation(), r, self.is_done()

    def reset(self):
        self.s = self.initial_state()
        self.actions_list = []
        self.step_ctr = 0
        self.ep_ctr += 1
        return self.state_to_observation()

    def reset_ep_ctr(self):
        self.ep_ctr = 0

    def state_to_observation(self):
        return self.s

    def is_done(self):
        if self.step_ctr >= self.max_episode_length:
            return True
        else:
            return False

    def run(self,agent,n_episodes,learning_active = True):
        reward_list = []
        for i in range(n_episodes):
            observation = self.reset()
            ep_r = 0
            while True:
                action = agent.choose_action(observation)

                observation_, reward, done = self.step(action)

                agent.store_transition(observation, action, reward, observation_)

                ep_r += reward
                if learning_active:
                    agent.learn()

                if done:
                    if i % 10 == 0 and i > 0:
                        print('Episode {} done.'.format(i))
                    reward_list.append(ep_r)
                    break

                observation = observation_
        return reward_list

class LongHallway(Environment):
    def __init__(self, hallway_length, MAX_EPISODE_LENGTH):
        super().__init__(3, 2, MAX_EPISODE_LENGTH) #2 features: task label (the hallway length) and current position
        self.hallway_length = hallway_length
        self.reset()

    def update_state(self, a):
        self.s += a - 1 # action is encoded as 0 (move left), 1 (don't move), and 2 (move right)
        self.s = np.clip(self.s,0,self.hallway_length) #clip to the possible states (i.e. don't move if you hit a wall)

    def initial_state(self):
        return np.zeros(1) # start at the left of the hallway

    def state_to_observation(self):
        return np.array([self.s,self.hallway_length])

    def calculate_reward(self, actions):
        if self.s == self.hallway_length:
            return 1
        else:
            return -1/self.max_episode_length

    def is_done(self):
        return (self.s == self.hallway_length) or super().is_done()
