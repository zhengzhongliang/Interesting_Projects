import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6
N_ACTIONS = 2    # 0=left, 1=right
EPSILON = 0.9    #in some cases, the agent will choose an action randomly
FRESH_TIME = 0.3
GAMMA = 0.9
ALPHA = 0.1

class Agent():
    def __init__(self):
        self.currentState = 2
        self.availableActions = np.arange(N_ACTIONS)
        self.availableStates = np.arange(N_STATES)
        self.qTable = np.zeros((N_STATES, N_ACTIONS))
    def choose_action(self):   #choose actioin according to Q-table value
        state_actions = self.qTable[self.currentState,:]
        if (np.random.uniform()>EPSILON) or (np.count_nonzero(state_actions)==0):
            action = int(np.round((np.random.uniform())))
        else:
            action = np.argmax(state_actions)
        print('state actions:',state_actions)
        print('action:',action)
        return action
    def get_feedback(self, state, action):
        if action == 1:
            if state ==N_STATES -2:
                new_state = 5
                reward = 1
            else:
                new_state = state + 1
                reward = 0
        else:
            reward = 0
            if state == 0:
                new_state = state
            else:
                new_state = state - 1
        return new_state, reward

class Epoch():
    def __init__(self):
        self.agent = Agent()
    def update_epoch(self):
        env_list = ['-']*(N_STATES-1) + ['T']
        if self.agent.currentState == 5:
            #interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
            #print('\r{}'.format(interaction), end='')
            #time.sleep(2)
            #print('\r                                ', end='')
            print('Bingo!!')
        else:
            print('current state:',self.agent.currentState)
            env_list[self.agent.currentState] = 'o'
            interaction = ''.join(env_list)
            print('\r{}'.format(interaction), end='')
            time.sleep(FRESH_TIME)
    def run_epoch(self):
        is_terminated = False
        while not is_terminated:
            action = self.agent.choose_action()
            new_state, reward = self.agent.get_feedback(state = self.agent.currentState, action = action)
            q_predict = self.agent.qTable[self.agent.currentState, action]
            if new_state != 5:
                q_target = reward + GAMMA * np.amax(self.agent.qTable[new_state, :])
            else:
                q_target = reward
                is_terminated = True
            self.agent.qTable[self.agent.currentState, action] += ALPHA * (q_target - q_predict)
            self.agent.currentState = new_state

            self.update_epoch()

    def reset(self):
        self.agent.currentState = 0



class Simulation():
    def __init__(self):
        self.epoch = Epoch()
    def run_sim(self):
        for i in np.arange(20):
            self.epoch.run_epoch()
            self.epoch.reset()



sim1 = Simulation()
sim1.run_sim()
