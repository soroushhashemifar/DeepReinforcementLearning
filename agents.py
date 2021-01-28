class LearningMethod:

  def select_action(self, current_state, step, test_phase=False):
    pass

  def get_batch(self, batch_size):
    pass

  def update_q(self, batch_size, modify_parameter=True):
    pass

  def split_batches(self, batch_size, method=None):
    minibatch = self.memory.sample(batch_size)

    current_states_ = np.array([np.array(i[0])[0] for i in minibatch])
    actions_ = np.array([i[1] for i in minibatch])
    rewards_ = np.array([i[2] for i in minibatch])
    next_states_ = np.array([np.array(i[3])[0] for i in minibatch])

    if method == "sarsa":
      next_actions_ = np.array([i[4] for i in minibatch])
      done_flags_ = np.array([i[5] for i in minibatch])

      return current_states_, actions_, rewards_, next_states_, next_actions_, done_flags_

    done_flags_ = np.array([i[4] for i in minibatch])

    return current_states_, actions_, rewards_, next_states_, done_flags_


class DeepQLearningMethod(LearningMethod):

  """
  DQN: https://arxiv.org/pdf/1312.5602.pdf
  Dueling: https://arxiv.org/pdf/1511.06581
  """

  def __init__(self, state_dimension, memory, policy, model, gamma=0.01, dueling=None):
    self.memory = memory
    self.policy = policy
    self.gamma = gamma

    self.model = model
    self.loss = []
    self.mae = []
    self.val_loss = []
    self.val_mae = []

    if dueling is not None:
      # get the second last layer of the model, abandon the last layer
      layer = self.model.layers[-2]
      
      nb_action = self.model.output.shape[-1]

      # layer y has a shape (nb_action+1,)
      # y[:,0] represents V(s;theta)
      # y[:,1:] represents A(s,a;theta)
      y = tf.keras.layers.Dense(nb_action + 1, activation='linear')(layer.output)
      
      # avg dueling: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
      # max dueling: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
      # naive dueling: Q(s,a;theta) = V(s;theta) + A(s,a;theta)
      if dueling == 'avg':
          outputlayer = tf.keras.layers.Lambda(lambda a: tf.keras.backend.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.keras.backend.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_action,))(y)
      elif dueling == 'max':
          outputlayer = tf.keras.layers.Lambda(lambda a: tf.keras.backend.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.keras.backend.max(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_action,))(y)
      elif dueling == 'naive':
          outputlayer = tf.keras.layers.Lambda(lambda a: tf.keras.backend.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)

      self.model = Model(inputs=self.model.input, outputs=outputlayer)
      self.model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.00001), 
                          loss=tf.keras.losses.MeanSquaredError(), 
                          metrics=["mae"])
      
  def select_action(self, current_state, step, test_phase=False):
    action_q_vals = self.model.predict(current_state)
    action_q_vals = np.squeeze(action_q_vals)
    action = self.policy.select_action(action_q_vals)
    
    return action

  def get_batch(self, batch_size):
    current_states_, actions_, rewards_, next_states_, done_flags_ = self.split_batches(batch_size)

    action_q_vals = self.model.predict(current_states_)
    next_action_q_vals = self.model.predict(next_states_)

    for index, action, reward, done_flag, next_action_q_val in zip(range(batch_size), actions_, rewards_, done_flags_, next_action_q_vals):
      if done_flag:
        action_q_vals[index, action] = reward
      else:
        action_q_vals[index, action] = reward + self.gamma * np.max(next_action_q_val)

    return current_states_, action_q_vals

  def update_q(self, batch_size, modify_parameter=True):
    current_states, action_q_vals = self.get_batch(batch_size)
    current_states_test, action_q_vals_test = self.get_batch(batch_size)

    history = self.model.fit(current_states, action_q_vals, 
                             validation_data=(current_states_test, action_q_vals_test), 
                             epochs=1, verbose=0)
    self.loss.append(history.history['loss'][0])
    self.mae.append(history.history['mae'][0])
    self.val_loss.append(history.history['val_loss'][0])
    self.val_mae.append(history.history['val_mae'][0])
    
    if modify_parameter:
        self.policy.decrement_parameter()


class DoubleDeepQLearningMethod(LearningMethod):

  """
  DDQN: https://arxiv.org/pdf/1509.06461.pdf
  Dueling: https://arxiv.org/pdf/1511.06581
  """

  def __init__(self, state_dimension, memory, policy, model, target_update_interval=1000, gamma=0.01, dueling=None):
    self.memory = memory
    self.policy = policy
    self.gamma = gamma
    self.target_update_interval = target_update_interval

    self.local_model = model

    self.loss = []
    self.mae = []
    self.val_loss = []
    self.val_mae = []

    # self.plot = PlotLearning()  

    if dueling is not None:
      # get the second last layer of the model, abandon the last layer
      layer = self.local_model.layers[-2]
      
      nb_action = self.local_model.output.shape[-1]

      # layer y has a shape (nb_action+1,)
      # y[:,0] represents V(s;theta)
      # y[:,1:] represents A(s,a;theta)
      y = tf.keras.layers.Dense(nb_action + 1, activation='linear')(layer.output)
      
      # avg dueling: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
      # max dueling: Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
      # naive dueling: Q(s,a;theta) = V(s;theta) + A(s,a;theta)
      if dueling == 'avg':
          outputlayer = tf.keras.layers.Lambda(lambda a: tf.keras.backend.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.keras.backend.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_action,))(y)
      elif dueling == 'max':
          outputlayer = tf.keras.layers.Lambda(lambda a: tf.keras.backend.expand_dims(a[:, 0], -1) + a[:, 1:] - tf.keras.backend.max(a[:, 1:], axis=1, keepdims=True), output_shape=(nb_action,))(y)
      elif dueling == 'naive':
          outputlayer = tf.keras.layers.Lambda(lambda a: tf.keras.backend.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)

      self.local_model = tf.keras.models.Model(inputs=self.local_model.input, outputs=outputlayer)
      self.local_model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.00001), 
                                loss=tf.keras.losses.MeanSquaredError(), 
                                metrics=["mae"])  

    self.target_model = tf.keras.models.clone_model(self.local_model)
    self.target_model.build((state_dimension,))
    self.target_model.compile(optimizer='adam', loss='mse')

  def select_action(self, current_state, step, test_phase=False):
    if step % self.target_update_interval == 0:
      self.target_model.set_weights(self.local_model.get_weights())

    action_q_vals = self.local_model.predict(current_state)
    action_q_vals = np.squeeze(action_q_vals)
    action = self.policy.select_action(action_q_vals)

    return action

  def get_batch(self, batch_size):
    current_states_, actions_, rewards_, next_states_, done_flags_ = self.split_batches(batch_size)

    action_q_vals = self.local_model.predict(current_states_)
    next_action_q_vals = self.target_model.predict(next_states_)

    for index, action, reward, done_flag, next_action_q_val in zip(range(batch_size), actions_, rewards_, done_flags_, next_action_q_vals):
      if done_flag:
        action_q_vals[index, action] = reward
      else:
        action_q_vals[index, action] = reward + self.gamma * next_action_q_val[np.argmax(action_q_vals[index])]

    return current_states_, action_q_vals

  def update_q(self, batch_size, modify_parameter=True):
    current_states, action_q_vals = self.get_batch(batch_size)
    current_states_test, action_q_vals_test = self.get_batch(batch_size)

    history = self.local_model.fit(current_states, action_q_vals, 
                                   validation_data=(current_states_test, action_q_vals_test), 
                                   epochs=1, verbose=0)#, callbacks=[self.plot])
    self.loss.append(history.history['loss'][0])
    self.mae.append(history.history['mae'][0])
    self.val_loss.append(history.history['val_loss'][0])
    self.val_mae.append(history.history['val_mae'][0])
    
    if modify_parameter:
        self.policy.decrement_parameter()


class SARSALearningMethod(DeepQLearningMethod):

  """
  https://sci-hub.do/10.1016/j.engappai.2013.06.016
  """

  def get_batch(self, batch_size):
    current_states_, actions_, rewards_, next_states_, next_actions_, done_flags_ = self.split_batches(batch_size, method="sarsa")

    action_q_vals = self.model.predict(current_states_)
    next_action_q_vals = self.model.predict(next_states_)

    for index, action, reward, done_flag, next_action_q_val, next_action in zip(range(batch_size), actions_, rewards_, done_flags_, next_action_q_vals, next_actions_):
      if done_flag:
        action_q_vals[index, action] = reward
      else:
        action_q_vals[index, action] = reward + self.gamma * next_action_q_val[next_action]

    return current_states_, action_q_vals


class DoubleSARSALearningMethod(DoubleDeepQLearningMethod):

  """
  https://www.scirp.org/journal/paperinformation.aspx?paperid=71237
  """

  def get_batch(self, batch_size):
    current_states_, actions_, rewards_, next_states_, next_actions_, done_flags_ = self.split_batches(batch_size, method="sarsa")

    action_q_vals = self.local_model.predict(current_states_)
    next_action_q_vals = self.target_model.predict(next_states_)

    for index, action, reward, done_flag, next_action_q_val, next_action in zip(range(batch_size), actions_, rewards_, done_flags_, next_action_q_vals, next_actions_):
      if done_flag:
        action_q_vals[index, action] = reward
      else:
        action_q_vals[index, action] = reward + self.gamma * next_action_q_val[next_action]

    return current_states_, action_q_vals
