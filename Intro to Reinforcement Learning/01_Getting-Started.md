## 1.1 ML Refresher

In supervised learning, every training sample from the dataset has a corresponding label associated with it, which essentially tells the machine learning algorithm what the training sample is. As a result, the algorithm can then learn from this data to predict labels for unseen data in the future.

In unsupervised learning, there are no labels for the training data. The machine learning algorithm tries to learn the underlying patterns or distributions that govern the data.

Reinforcement learning is very different to supervised and unsupervised learning. In reinforcement learning, the algorithm learns from experience and experimentation. Essentially, it learns from trial and error.

## 1.2 Intro to RL

Reinforcement learning consists of several key concepts:

* **Agent** is the entity being trained. In our example, this is a dog.
* **Environment** is the “world” in which the agent interacts, such as a park.
* **Actions** are performed by the agent in the environment, such as running around, sitting, or playing ball.
* **Rewards** are issued to the agent for performing good actions.

**Atari Breakout**

* **Agent** is the paddle;
* **Environment** is the game scenes with the bricks and boundaries;
* **Actions** are the movement of the paddle; and
* **Rewards** are issued by the reinforcement learning model based upon the number of bricks hit with the ball.

**Traffic Signals**

* **Agent** is the traffic light control system;
* **Environment** is the road network;
* **Actions** are changing the traffic light signals (red-yellow-green); and
* **Rewards** are issued by the reinforcement learning model based upon traffic flow and throughput in the road network.

**Autonomous Vehicles**

* **Agent** is the car (or, more correctly, the self-driving software running on the car);
* **Environment** is the roads and surrounds on which the car is driving;
* **Actions** are things such as steering angle and speed; and
* **Rewards** are issued by the reinforcement learning model based upon how successfully the car stays on the road and drives to the destination.

## 1.3 RL with a Mechanical Computer

This chapter uses the game of Hexapawn and a simple mechanical computer to provide insight into how reinforcement learning works. The computer is made out of 24 matchboxes that represent every state in the game and all possible moves from within each state. The matchboxes are filled with a few colored beads that each represent a move possible in that state. The computer is punished when it makes a poor decision by removing the bead corresponding to that move so that it can not be repeated. Through additional turns and negative feedback, the computer learns to optimize its moves and eventually win games against human players.

To learn more about Hexapawn, you can find the original article written by AI researcher Martin Gardner here: “How to build a game-learning machine and then teach it to play and win” by Martin Gardner, Scientific American: [http://cs.williams.edu/~freund/cs136-073/GardnerHexapawn.pdf](http://cs.williams.edu/~freund/cs136-073/GardnerHexapawn.pdf) .
