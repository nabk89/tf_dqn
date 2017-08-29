# tf_dqn
Deep Q-Networks coded by TensorFlow

## Requirements
- Python 2.7
- [gym](https://github.com/openai/gym)
- [tqdm](https://github.com/tqdm/tqdm)
- [OpenCV2](http://opencv.org/)
- [TensorFlow](https://www.tensorflow.org/) (current: 1.2.1)

## Usage
Train with DQN model described in [[2]](#tf_dqn) (Hyper-parameters are described in main.py):

    $ python main.py

## TODO
Even though the codes work, it seems there are bugs (training doesn't work well).
I'm debugging.

## References
Papers
1) [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602) 
2) [Human-Level Control through Deep Reinforcement Learning](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf)

Implementations
1) [deep-rl-tensorflow](https://github.com/carpedm20/deep-rl-tensorflow)
2) [DQN](https://github.com/yjhong89/DQN)
