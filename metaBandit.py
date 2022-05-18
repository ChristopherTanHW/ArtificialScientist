import numpy as np
import random

class QBandit:
    def __init__(self, alpha, gamma, epsilon, num_actions):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = self.initialise_q_table(num_actions)
        self.num_actions = num_actions

    def initialise_q_table(self, num_actions):
        q_table = np.ones(num_actions)
        return q_table
    
    def policy(self):
        if random.uniform(0, 1) > self.epsilon:
            action = np.random.choice(np.flatnonzero(self.q_table == self.q_table.max()))
        else:
            action = random.randint(0, len(self.q_table)-1)              
        return action
        
    def update_Q(self, next_state, action_str, reward):
            #calculating value function update
            old_value = self.row[action_str]
            new_row = self.q_table.loc[(self.q_table['s0'] == tuple(next_state[0])) & 
                      (self.q_table['s1'] == tuple(next_state[1]))]
            next_max = self.row[action_space_str].values.max()
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)

            #applying the update
            self.q_table.loc[self.row.index.item(), action_str] = new_value.item()


class UCBBandit:
  def __init__(self, number_of_arms, bonus_multiplier):
    self._number_of_arms = number_of_arms
    self._bonus_multiplier = bonus_multiplier
    self.reset()
    self.t = 1
    self.counts = np.ones(number_of_arms)
    self.Q = np.zeros(number_of_arms)

  def step(self, previous_action, reward):
    #update statistics
    if not(previous_action == None):
      self.counts[previous_action] += 1
      self.Q[previous_action] = self.Q[previous_action] + (
          reward - self.Q[previous_action]) / self.counts[previous_action]
      self.t += 1

    #get new action
    action = np.argmax(self.Q + self._bonus_multiplier * np.sqrt(
        np.log(self.t) / self.counts))
    return action

  def reset(self):
    self.t = 1
    self.counts = np.ones(self._number_of_arms)
    self.Q = np.zeros(self._number_of_arms)