## 2.1 RL in AWS DeepRacer

AWS DeepRacer is a 1/18th scale racing car, with the objective being to drive around a track as fast as possible. To achieve this goal, AWS DeepRacer uses reinforcement learning.

* The **agent** is the AWS DeepRacer car (or, more specifically, the software running on the car);
* The agent wants to achieve the goal of finishing laps around the track as fast as possible, so the track is the  **environment** .
* The agent knows about the environment through the **state** which is the portion of the environment known to the agent. In the case of AWS DeepRacer, it is the images being captured by the camera.
* Once the agent knows its state in the environment, it can perform **actions** in the environment to help it achieve its goal. In the case of DeepRacer, this might be accelerating, braking, turning left, turning right, or going straight.
* The agent then receives feedback in the form of a **reward** about how well that action contributed towards achieving its goal.
* And all this happens within an  **episode** . This can be thought of as a cycle of the agent performing an action in the environment (based upon the state it has observed) and then receiving feedback in the form of a reward which informs future actions it might take.

## 2.2 Training an AWS DR Model

Creating a model in AWS DeepRacer Student is a six-step process:

* Name your model
* Choose track
* Choose algorithm type
* Customize reward function
* Choose duration
* Train your model

My suggestion is using the track or race name and a version number

Note that the more challenging the track, the more training time you will need - and possibly also a more complex reward function.

As a rule of thumb, if you are competing in the Student League then train on the track which the competition is using.
