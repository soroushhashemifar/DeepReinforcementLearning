from utils import *
from policies import *
import simulations import *

def get_model(input_shape, output_shape):
  input_layer = tf.keras.layers.Input(shape=input_shape)
  layer = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal", bias_initializer="zeros")(input_layer)
  layer = tf.keras.layers.Dropout(0.2)(layer)
  layer = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal", bias_initializer="zeros")(layer)
  layer = tf.keras.layers.Dropout(0.2)(layer)
  layer = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal", bias_initializer="zeros")(layer)
  layer = tf.keras.layers.Dropout(0.2)(layer)
  layer = tf.keras.layers.Dense(128, activation="relu", kernel_initializer="he_normal", bias_initializer="zeros")(layer)
  layer = tf.keras.layers.Dropout(0.2)(layer)
  output_layer = tf.keras.layers.Dense(output_shape, activation="linear")(layer)

  model = tf.keras.models.Model(input_layer, output_layer)
  model.compile(optimizer=tf.keras.optimizers.Nadam(lr=0.00001), 
                loss=tf.keras.losses.MeanSquaredError(), 
                metrics=["mae"])

  return model


env = gym.make('CartPole-v0')
# env = gym.make('LunarLander-v2')

model = get_model(env.observation_space.shape, env.action_space.n)
memory = Memory(1000)

# policy = EpsilonGreedyDecisionPolicy(1., epsilon_min=0.3, epsilon_decay=0.99)
policy = BoltzmannQDecisionPolicy(100., clip=(-20, 20))
# policy = MaxBoltzmannQDecisionPolicy(100., 1., epsilon_min=0.3, epsilon_decay=0.99, clip=(-20, 20))

run_simulation_dqn(episodes=50000, 
                   batch_size=128, 
                   environment=env, 
                   model=model, 
                   memory=memory, 
                   policy=policy, 
                   init_reward=0, 
                   target_update_interval=1000, 
                   gamma=0.99)