# DeepReinforcementLearning (DeepRL)

DeepRL is a python framework for Deep Reinforcement Learning. It is build with modularity in mind so that it can easily be adapted to any need. It provides many possibilities out of the box such as Simple/Double Q-learning and Simple/Double SARSA. Many Open-AI environment examples are also compatible with the framework.

## Dependencies

This framework is tested to work under Python 3.7. The required dependencies are NumPy >= 1.19 and Tensorflow >= 2.3.

## Available agents

- Deep Q Network (with dueling)
- Double Deep Q Network (with dueling)
- Deep SARSA (with dueling)
- Double Deep SARSA (with dueling)

## TODO

- Actor-Critic
- Advantage Actor Critic (A2C)
- Asynchronous Advantage Actor Critic (A3C)
- Deep Deterministic Policy Gradient (DDPG)
- Prioritized Experience Replay technique

## References

[1] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller, Playing Atari with Deep Reinforcement Learning, NIPS Deep Learning Workshop 2013

[2] Hado van Hasselt, Arthur Guez, and David Silver, Deep Reinforcement Learning with Double Q-learning, AAAI 2016

[3] Yin-Hao Wang, Tzuu-Hseng S. Li, and Chih-Jui Lin, Backward Q-learning: The combination of Sarsa algorithm and Q-learning, Engineering Applications of Artificial Intelligence, Elsevier, 2013

[4] Michael Ganger, Ethan Duryea, and Wei Hu, Double Sarsa and Double Expected Sarsa with Shallow and Deep Learning, Journal of Data Analysis and Information Processing, 2016

[5] Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, and Nando de Freitas, Dueling Network Architectures for Deep Reinforcement Learning, Proceedings of The 33rd International Conference on Machine Learning, 2016
