# Deep Reinforcement Learning Nanodegree 

## Project 1: Navigation: REPORT

[image1]: 

![image1](https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif)

### Introduction

This report provides a description of the implementation for the Deep Reinforcement Learning Nanodegree Project 1, where I had to train an agent to navigate (and collect bananas) in a large, square world. Please refer to the [README.md]() on this repository for more information



### Learning Algorithm & Plot of Rewards

1. **The Agent**:

    The agent architecture can be found on "**dqn_agent.py**" . This file implements an "Agent" class that holds:

    * args (class defined on the notebook): A set of parameters that will define the agent hyperparameters
    * state_size (int): dimension of each state
    * action_size (int): dimension of each action

    The agent's Neural Network (NN) used is implemented and described on "**model.py**".  In this file you will find and interesting addition added to the basic and simple NN proposed by Udacity. This addition is "**Noisy Nets [8]**". In paper [8] the authors introduce "NoisyNet", a deep reinforcement learning agent with parametric noise added to its weights, and show that the induced stochasticity of the agent's policy can be used to aid efficient exploration. The parameters of the noise are learned with gradient descent along with the remaining network weights. NoisyNet is straightforward to implement and adds little computational overhead. We find that replacing the conventional exploration heuristics for A3C, DQN and dueling agents (entropy reward and ϵ-greedy respectively) with NoisyNet yields substantially higher scores for a wide range of Atari games, in some cases advancing the agent from sub to super-human performance. For the sake of this study, and due to the simple architecture of the network, I will compare  the performance results while using the NN with 3 linear layers (Udacity proposed simple NN. class DQN on model.py) and the 3 Noisy linear layers instead ( class DQN_NoisyNet)

    **Both networks implemented uses just 3 layers ** due to 1) the small number of  inputs and 2) just to get a feeling for the entire DRL development process; including parameter tuning, testing and evaluation. 

    Simple Comparative using Basic DQN Vs DQN_NoisyNet

    ```
    n_episodes=2000, max_t=1000, eps_start = 0.5, eps_end=0.01, eps_decay=0.995
    ```

    | DQN Model              | Episodes Req to PASS (>=13) | Avg reward after 2000 episodes |
    | :--------------------- | :-------------------------: | :----------------------------: |
    | Basic (3 inear layers) |             315             |             14.24              |
    | NoisyNet               |             301             |             15.60              |

    Multiple tests (1 comparative only shown here) suggest that NoisyNet does in fact improve the scores of our agent.
      

    **The agent uses 2 of these networks to implement a Double DQN [3]**. As explained on Udacity's lectures, the idea of Double Q-learning is to reduce overestimations by decomposing the max operation in the target into action selection and action evaluation. Although not fully decoupled, the target network in the DQN architecture provides a natural candidate for the second value function, without having to introduce additional networks. On the referenced paper, they proposed to evaluate the greedy policy according to the online network, but using the target network to estimate its value. You'll see the details of this implementation mostly on the "learn" method of the Agent class.

    Last but not least for this initial simple project, **the agent also uses the Experience Replay.** Experience replay lets online reinforcement learning agents remember and reuse experiences from the past. In this project, experience transitions were uniformly sampled from a replay memory. However, this approach simply replays transitions/experiences at the same frequency that they were originally experienced, regardless of their significance. Including a priority or importance to these experiences will be implemented very soon following Prioritized Experience Replay [4]

2. Hyper-parameters:

    The most relevant hyper-parameters used in this implementation are:

    * Epsilon parameters: eps_start, eps_end, eps_decay=0.995

      (1-eps), as you already know, represents "How" greedy you want to be while selecting your "best" action (Exploitation Vs Exploration). In the original Udacity's "Deep_Q_Network_Solution.ipynb" eps_start was set to 1, eps_end = 0.01 and with a decay of 0.995. In this project, this start value proved to slow down the "learning" process. Let's look at the different learning curves and maximum rewards obtained in 2000 episodes.

      ```
      n_episodes=2000, max_t=1000, eps_end=0.01, eps_decay=0.995
      ```

      | eps_start | Episodes Req to PASS (>=13) | Max Avg reward (after 2000 episodes) | Img  |
      | :-------: | :-------------------------: | :----------------------------------: | ---- |
      |   1.00    |             374             |                14.65                 | 1    |
      |   0.70    |             316             |                15.93                 | 2    |
      |   0.60    |             352             |                17.17                 | 3    |
      |   0.50    |             301             |                15.60                 | 4    |
      |   0.40    |             364             |                                      | 5    |
      |   0.30    |             135             |                14.66                 | 6    |
      |   0.20    |     1! (officially 140)     |                15.28                 | 7    |

      Image 1

      ![](/home/munirjojo-verge/deep-reinforcement-learning/p1_navigation/Banana_Jojo_basic/Images/Image100.png)

      Image 2

      ![](/home/munirjojo-verge/deep-reinforcement-learning/p1_navigation/Banana_Jojo_basic/Images/Image070.png)

      Image 3

      ![](/home/munirjojo-verge/deep-reinforcement-learning/p1_navigation/Banana_Jojo_basic/Images/Image065.png)

      Image 4

      ![](/home/munirjojo-verge/deep-reinforcement-learning/p1_navigation/Banana_Jojo_basic/Images/Image050_1.png)

      Image 5

      ![](/home/munirjojo-verge/deep-reinforcement-learning/p1_navigation/Banana_Jojo_basic/Images/Image040_Normal.png)

      Image 6

      ![](/home/munirjojo-verge/deep-reinforcement-learning/p1_navigation/Banana_Jojo_basic/Images/Image030.png)

      Image 7

      ![](/home/munirjojo-verge/deep-reinforcement-learning/p1_navigation/Banana_Jojo_basic/Images/Image020.png)



      It's clear that by changing (reducing) only the eps_start we see a speeding up in the learning process as a reflection that the agent reaches the PASS criteria earlier. The Agent does NOT need too much "exploration" to realize the underlying rules of the environment and can start "exploiting" this knowledge faster. We also see a clear decay in the rate between scores and episodes after about 500-700 episodes.  Although the numbers show a continues average reward growth, its gradient significantly decreases and seems to reach a plateau. The question that rises is: Is there nothing else to learn or is the Agent unable to learn anything else? To answer this question a few other questions rise: Should we increase the number of episodes? Should we increase the length of each episode? Should we decrease the epsilon decay so, even though the agent will learn slower, we might give more chances to "explore" and learn more the underlying patterns and dynamics of the environment? These questions do not have a straight and simple answers and we need to explore them.  

      Let's start with making the length of each episode larger: max_t = 2000. Let's do 2 comparatives, first with the eps_start = 0.3 and go from max_t =1000 to max_t=2000

      ```
      n_episodes=2000, =2000, eps_start = 0.3, eps_end=0.01, eps_decay=0.995
      ```

      | max_t | Episodes Req to PASS (>=13) | Avg reward after 2000 episodes | Img  |
      | ----- | --------------------------- | ------------------------------ | ---- |
      | 1000  | 135                         | 14.66                          | 1    |
      | 2000  | 142                         | 15.64                          | 2    |

      Image 1

      ![](/home/munirjojo-verge/deep-reinforcement-learning/p1_navigation/Banana_Jojo_basic/Images/Image030.png)

      Image 2

      ![](/home/munirjojo-verge/deep-reinforcement-learning/p1_navigation/Banana_Jojo_basic/Images/Image030_2.png)

      Let's try now with the eps_start = 0.4 and go from max_t =1000 to max_t=2000

      ```
      n_episodes=2000, =2000, eps_start = 0.4, eps_end=0.01, eps_decay=0.995
      ```

      | max_t | Episodes Req to PASS (>=13) | Avg reward after 2000 episodes | Img  |
      | ----- | --------------------------- | ------------------------------ | ---- |
      | 1000  | 364                         | 15.76                          | 1    |
      | 2000  | 313                         | 15.61                          | 2    |

      Image 1

      ![](/home/munirjojo-verge/deep-reinforcement-learning/p1_navigation/Banana_Jojo_basic/Images/Image040_Normal.png)

      Image 2

      ![](/home/munirjojo-verge/deep-reinforcement-learning/p1_navigation/Banana_Jojo_basic/Images/Image040_long2.png)

      As you can see, there are no significant changes in the Agent's performance and it doesn't look like extending the episodes' lengths is a path we want to explore deeper before checking the other parameters.

      It's worth trying to decrease the epsilon decay rate (by increasing the eps_decay). Remember, for instance, that when epsilon starts at 0.7 it will take only 850 episodes to reach it's minimum value of 0.01. From that point on (850 through 2000 episodes) the agent is as greedy as it can reasonably be. During this period (when epsilon it's at is minimum and the agent should have learned all it could) we can still see significant instability measured by the standard deviation between each episode reward and the average for the last 100 episodes. This fact is easily visible on the graphs (Scores Vs Episode#) by looking at the width of the plot. We see very close episodes (40 episodes difference) going from 9.00 score to 27.0. Bellow is a comparative table with the associated graphs that show that "learning slower is not learning more/better" We can still observe a clear instability. Therefore, instability is not solved by learning slower (smaller epsilon decay)

      ```
      n_episodes=2000, max_t=1000, eps_start = 0.5, eps_end=0.01
      ```

      | eps_decay | Episodes Req to PASS (>=13) | Max Avg reward (after 2000 episodes ) |
      | :-------: | :-------------------------: | :-----------------------------------: |
      |   0.995   |             301             |                 15.60                 |
      |   0.999   |            1425             |                 13.96                 |



      After hours and hours of experimentation it seems evident that the core of the problem resides in the "not to adequate" selection of the NN that lays as the foundation of this DRL DQN algorithm. The simplicity/low depth of the Network seems NOT to address enough features on the environment and therefore unable to to "Extract" more subtle patterns on the environment dynamics. Experimenting with "deeper" (more sophisticated) networks is definitely a must area of future study.

### Future Work

All this results and conclusions suggest a series of changes (Future Work) to improve the agent's performance and to reduce it's instability. These series of changes must include:

* Prioritized Experience Replay [4]
* Dueling Network Architectures for Deep Reinforcement Learning [5]
* A Distributional Perspective on Reinforcement Learning [7]
* Study different and more complex NN' architectures applicable to the problem.

Which will lead us to:

* Rainbow: Combining Improvements in Deep Reinforcement Learning [1]
* Playing Atari with Deep Reinforcement Learning [2]

### References

[1] Rainbow: Combining Improvements in Deep Reinforcement Learning  (https://arxiv.org/abs/1710.02298)
[2] Playing Atari with Deep Reinforcement Learning (http://arxiv.org/abs/1312.5602)
[3] Deep Reinforcement Learning with Double Q-learning (https://arxiv.org/abs/1509.06461)
[4] Prioritized Experience Replay (https://arxiv.org/abs/1511.05952)
[5] Dueling Network Architectures for Deep Reinforcement Learning
[6] Reinforcement Learning: An Introduction
[7] A Distributional Perspective on Reinforcement Learning (https://arxiv.org/abs/1707.06887)
[8] Noisy Networks for Exploration (https://arxiv.org/abs/1706.10295)