from agents import *
import numpy as np


def run_simulation_dqn(episodes, batch_size, environment, model, memory, policy, init_reward=0, target_update_interval=1000, gamma=0.99, dueling=None):
  if target_update_interval is None:
    method = DeepQLearningMethod(environment.observation_space.shape[0], memory, policy, model, gamma=gamma, dueling=dueling)
  else:
    method = DoubleDeepQLearningMethod(environment.observation_space.shape[0], memory, policy, model, target_update_interval=target_update_interval, gamma=gamma, dueling=dueling)

  total_steps = 0
  for episode in range(1, episodes):
      observation = environment.reset()
      done = False
      episode_reward = init_reward
      steps = 0

      while not done:
        action = method.select_action(np.asmatrix(observation), total_steps)
        next_observation, reward_step, done, _ = environment.step(action)

        episode_reward += reward_step
        method.memory.memorize((np.asmatrix(observation), action, episode_reward, np.asmatrix(next_observation), done))
    
        if method.memory.size() >= batch_size:
          method.update_q(batch_size, modify_parameter=False)

        observation = next_observation
        steps += 1

        if done: break

      total_steps += steps

      if total_steps >= batch_size:
        # print("episode", episode, "episode reward", episode_reward, "steps per episode", steps, "epsilon", method.policy.epsilon, "mean loss", np.mean(method.loss), "mean mae", np.mean(method.mae))
        print("episode", episode, "| episode reward", episode_reward, "| steps per episode", steps, "| total steps", total_steps, "| mean loss", np.mean(method.loss), "| mean mae", np.mean(method.mae), "| mean val_loss", np.mean(method.val_loss), "| mean val_mae", np.mean(method.val_mae))

  env.close()


def run_simulation_sarsa(episodes, batch_size, environment, model, memory, policy, init_reward=0, target_update_interval=1000, gamma=0.99, dueling=None):
  if target_update_interval is None:
    method = SARSALearningMethod(environment.observation_space.shape[0], memory, policy, model, gamma=gamma, dueling=dueling)
  else:
    method = DoubleSARSALearningMethod(environment.observation_space.shape[0], memory, policy, model, target_update_interval=target_update_interval, gamma=gamma, dueling=dueling)

  total_steps = 0
  for episode in range(1, episodes):
    observation = environment.reset()
    action = method.select_action(np.asmatrix(observation), total_steps)
    
    done = False
    episode_reward = init_reward
    steps = 0
    while not done:
      next_observation, reward_step, done, _ = environment.step(action)
      episode_reward += reward_step

      next_action = method.select_action(np.asmatrix(next_observation), total_steps)

      method.memory.memorize((np.asmatrix(observation), action, episode_reward, np.asmatrix(next_observation), next_action, done))
  
      if method.memory.size() >= batch_size:
        method.update_q(batch_size, modify_parameter=False)

      action = next_action
      observation = next_observation
      steps += 1

      if done: break

    total_steps += steps

    if total_steps >= batch_size:
      # print("episode", episode, "episode reward", episode_reward, "steps per episode", steps, "epsilon", method.policy.epsilon, "mean loss", np.mean(method.loss), "mean mae", np.mean(method.mae))
      print("episode", episode, "| episode reward", episode_reward, "| steps per episode", steps, "| total steps", total_steps, "| mean loss", np.mean(method.loss), "| mean mae", np.mean(method.mae), "| mean val_loss", np.mean(method.val_loss), "| mean val_mae", np.mean(method.val_mae))

  env.close()
