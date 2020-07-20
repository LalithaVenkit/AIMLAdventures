# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [[j,i] for j in range (0,m) for i in range(0,m) if ((i!=j) or (i==0 and j==0))]
        self.state_space = [ [loc,t,d,a[0],a[1]] for loc in range(0,5) for d in range(0,7) for t in range(0,24) for a in self.action_space]
        self.state_init = random.choice(self.state_space)[0:3]
        self.time_elapsed = 0
        self.terminal_hours = 30*24
        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input
    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = np.zeros(m + t + d)
        state_encod[state[0]] = 1
        state_encod[m+state[1]] = 1
        state_encod[m+t+state[2]] = 1
        return state_encod
    

    # Use this function if you are using architecture-2 
    def state_encod_arch2(self, state, action):
        #convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a           #vectorformat. Hint: The vector is of size m + t + d + m + m."""
        state_encod = np.zeros(m + t + d + m + m)
        state_encod[state[0]] = 1
        state_encod[m+state[1]] = 1
        state_encod[m+t+state[2]] = 1
        state_encod[m+t+d+action[0]] = 1
        state_encod[m+t+d+m+action[1]] = 1
        return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        elif location ==1:
            requests = np.random.poisson(12)
        elif location ==2:
            requests = np.random.poisson(4)
        elif location ==3:
            requests = np.random.poisson(7)
        elif location ==4:
            requests = np.random.poisson(8)    

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]
        actions.append([0,0])
        possible_actions_index.append(0)

        return possible_actions_index,actions   


    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        curr_state = state.copy()
        if(action == [0,0]):
            reward = -C
        else:
            time_to_start = 0
            if(state[0] != action[0]):
                time_to_start=Time_matrix[curr_state[0]][action[0]][curr_state[1]][curr_state[2]]
                t,d = self.get_new_time([curr_state[1],curr_state[2]],time_to_start)
                curr_state[0]=action[0]
                curr_state[1]=t
                curr_state[2]=d
            trip_time = Time_matrix[action[0]][action[1]][curr_state[1]][curr_state[2]]    
            reward = R * trip_time - C * (trip_time + time_to_start)
        return reward


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        is_terminal = False
        time_taken = 0
        if(action == [0,0]):
            next_state = state.copy()
            time_taken=1
            t,d = self.get_new_time([state[1],state[2]],time_taken)
            next_state[1]=t
            next_state[2]=d
        else:
            next_state = state.copy()
            if(state[0] != action[0]):
                time_taken=Time_matrix[state[0]][action[0]][state[1]][state[2]]
                t,d = self.get_new_time([state[1],state[2]],time_taken)
                next_state[0]=action[0]
                next_state[1]=t
                next_state[2]=d
            trip_time = Time_matrix[action[0]][action[1]][next_state[1]][next_state[2]]
            t,d = self.get_new_time([state[1],state[2]],trip_time)
            next_state[0]=action[1]
            next_state[1]=t
            next_state[2]=d
            time_taken+=trip_time
        self.time_elapsed += time_taken
        if(self.time_elapsed >= self.terminal_hours):
            is_terminal = True
        
        return next_state, is_terminal


    def reset(self):
        return self.action_space, self.state_space, self.state_init
    
    def get_new_time(self,t_d,hours):
        t= t_d[0]
        d= t_d[1]
        t = t+hours
        if(t>23):
            t=t-23-1
            d=d+1
        if(d>6):
            #Assuming no trip is more than 24 hours
            d=0
        return t,d
