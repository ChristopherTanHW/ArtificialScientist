import numpy as np
import random

class TabQ:
    def __init__(self, alpha, gamma, epsilon, num_states, num_actions):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = self.initialise_q_table(num_states, num_actions)
        self.num_actions = num_actions
        self.num_states = num_states

    def initialise_q_table(self, num_states, num_actions):
        q_table = np.ones((num_states, num_actions))
        return q_table
    
    def step(self, state):
        state_action_row = self.q_table[state]
        if random.uniform(0, 1) > self.epsilon:
            action = np.random.choice(np.flatnonzero(state_action_row == state_action_row.max()))
        else:
            action = random.randint(0, len(state_action_row)-1)              
        return action
        
    def update_Q(self, current_state, next_state, action, reward):
            #calculating value function update
            current_Q = self.q_table[current_state][action]
            Q_next_state = self.q_table[next_state]
            next_state_best_A = np.random.choice(np.flatnonzero(Q_next_state == Q_next_state.max()))
            Q_next_max = self.q_table[next_state][next_state_best_A]
            new_value = current_Q + self.alpha * (reward + self.gamma * Q_next_max - current_Q)
            #applying the update
            self.self.q_table[current_state][action] = new_value


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

class DQNWriter:
  def __init__(self):
    pass