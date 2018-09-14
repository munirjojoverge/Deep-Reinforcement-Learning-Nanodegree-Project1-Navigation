[//]: # "Image References"

# Deep Reinforcement Learning Nanodegree 

## Project 1: Navigation

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif	"Trained Agent"

### Introduction

For this project, I had to train an agent to navigate (and collect bananas) in a large, square world.  

![Trained Agent][image1]

A reward of **+1** is provided for collecting a yellow banana, and a reward of **-1** is provided for collecting a blue banana.  Thus, the goal of my agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The **state space has 37 dimensions** and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and in order to **solve the environment**, the agent must get an average score of +13 over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)

    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)

    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)

    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

    - This project uses Unity's rich environments to design, train, and evaluate deep reinforcement learning algorithms. **To run this project you'll need to install Unity ML-Agents.**You can read more about ML-Agents and how to install it by perusing the [GitHub repository](https://github.com/Unity-Technologies/ml-agents). 

      > **Note: The Unity ML-Agent team frequently releases updated versions of their environment. We are using the v0.4 interface. To avoid any confusion, please use the workspace we provide here or work with v0.4 locally.**

    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions and descriptions (step by step) in `Navigation.ipynb` to follow the algorithm details: From training to testing the agent behavior in this environment.  

## Remark

------

Please exercise extreme caution in robotic applications!

![Found on Twitter](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/April/5ae1ee84_bananas-save/bananas-save.gif)

Found on Twitter