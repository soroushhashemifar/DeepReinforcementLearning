import numpy as np
import random


class DecisionPolicy:
  
  def select_action(self, q_values, test_phase=False):
    pass

  def decrement_parameter(self):
    pass


class EpsilonGreedyDecisionPolicy(DecisionPolicy):
  
  def __init__(self, epsilon, epsilon_min=0.1, epsilon_decay=0.):
    self.epsilon = epsilon
    self.epsilon_min = epsilon_min
    self.epsilon_decay = epsilon_decay

  def select_action(self, q_values, test_phase=False):
    if random.random() > self.epsilon or test_phase:
        return np.argmax(q_values, axis=0)
    else:
        return random.choice(range(q_values.shape[0]))

  def decrement_parameter(self):
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay


class BoltzmannQDecisionPolicy(DecisionPolicy):
  
  def __init__(self, tau, clip=(-100, 100)):
    self.tau = tau
    self.clip = clip

  def select_action(self, q_values, test_phase=False):
    exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
    probs = np.squeeze(exp_values) / np.sum(exp_values)

    return np.random.choice(np.arange(probs.shape[0]), p=probs)


class MaxBoltzmannQDecisionPolicy(DecisionPolicy):
  
  def __init__(self, tau, epsilon, epsilon_min=0.1, epsilon_decay=0., clip=(-100, 100)):
    self.tau = tau
    self.epsilon = epsilon
    self.clip = clip

  def select_action(self, q_values, test_phase=False):
    if random.random() < self.epsilon:
      exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
      probs = np.squeeze(exp_values / np.sum(exp_values, axis=0))

      return np.random.choice(np.arange(probs.shape[0]), p=probs)
    else:
      return np.argmax(q_values, axis=0)
