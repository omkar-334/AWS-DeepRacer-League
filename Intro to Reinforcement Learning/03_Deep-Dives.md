## 3.1 Training Algorithms

* Proximal Policy Optimization (PPO)
* Soft Actor Critic (SAC)

A policy defines the action that the agent should take for a given state. This could conceptually be represented as a table - given a particular state, perform this action.

This is called a **deterministic policy** , where there is a direct relationship between state and action. This is often used when the agent has a full understanding of the environment and, given a state, always performs the same action.

Consider the classic game of rock, paper, scissors. An example of a deterministic policy is always playing rock. Eventually the other players are going to realize that you are always playing rock and then adapt their strategy to win, most likely by always playing paper. So in this situation it’s not optimal to use a deterministic policy.

So, we can alternatively use a **stochastic policy** . In a stochastic policy you have a range of possible actions for a state, each with a probability of being selected. When the policy is queried to return an action for a state it selects one of these actions based on the probability distribution.

This would obviously be a much better policy option for our rock, paper, scissors game as our opponents will no longer know exactly which action we will choose each time we play.

You might now be asking, with a stochastic policy how do you determine the value of being in a particular state and update the probability for the action which got us into this state? This question can also be applied to a deterministic policy; how do we pick the action to be taken for a given state?

Well, we somehow need to determine how much benefit we have derived from that choice of action. We can then update our stochastic policy and either increase or decrease the probability of that chosen action being selected again in the future, or select the specific action with the highest likelihood of future benefit as in our deterministic policy.

If you said that this is based on the reward, you are correct. However, the reward only gives us feedback on the value of the single action we just chose. To truly determine the value of that action (and resulting state) we should not only look at the current reward, but future rewards we could possibly get from being in this state.

This is done through the  **value function** . Think of this as looking ahead into the future and figuring out how much reward you expect to get given your current policy.

Say the DeepRacer car (agent) is approaching a corner. The algorithm queries the policy about what to do, and it says to accelerate hard. The algorithm then asks the value function how good it thinks that decision was - but unfortunately the results are not too good, as it’s likely the agent will go off-track in the future due to his hard acceleration into a corner. As a result, the value is low and the probabilities of that action can be adjusted to discourage selection of the action and getting into this state.

This is an example of how the value function is used to critique the policy, encouraging desirable actions while discouraging others. We call this adjustment a **policy update** , and this regularly happens during training. In fact, you can even define the number of episodes that should occur before a policy update is triggered.

In practice the value function is not a known thing or a proven formula. The reinforcement learning algorithm will estimate the value function from past data and experience.

**PPO** uses **“on-policy” learning** . This means it learns only from observations made by the current policy exploring the environment - using the most recent and relevant data. Say you are learning to drive a car, on-policy learning would be analogous to you reviewing a video of your most recent lesson and taking note of what you did well, and what needs improvement.

In contrast, **SAC** uses **“off-policy” learning.** This means it can use observations made from previous policies exploration of the environment - so it can also use old data. Going back to our learning to drive analogy, this would involve reviewing videos of your driving lessons from the last few weeks. Even though you have probably improved since those lessons, it can still be helpful to watch those videos in order to reinforce good and bad things. It could also include reviewing videos of other drivers to get ideas about good and bad things they might be doing.

* PPO generally needs more data as it has a reasonably narrow view of the world, since it does not consider historical data - only the data in front of it during each policy update. In contrast, SAC does consider historical data so it needs less new data for each policy update.
* That said, PPO can produce a more stable model in the short-term as it only considers the most recent, relevant data - compared with SAC which might produce a less stable model in the short-term since it considers less relevant, historical data.


## 3.2 Reward Functions

There are over 20 parameters available for use, and the reward function is simply a piece of code which uses the input parameters to do some calculations and then output a number, which is the reward.

The reward function is written in Python as a standard function, but it must be called r*eward_function* with a single parameter - which is a Python dictionary containing all the input parameters provided by the AWS DeepRacer system.

## 3.3 How the AWS DR works

The AWS DeepRacer device is a 1/18th scale race car. It has an onboard computer equipped with Wi-Fi and an Intel Atom processor running Ubuntu Linux 20.04. The onboard computer stores machine learning models and provides the processing power that allows the car to make decisions and navigate the race track. The AWS DeepRacer device can be equipped with up to two single-lens cameras and a LiDAR sensor.

We can measure the performance of inference with two metrics:

* The "inference rate" which is the number of inferences which can be done per second; and
* The "inference time" or "inference latency" which is time taken to run a single inference.

Ultimately, we want to maximize the rate of inference and therefore the number of decisions made every second. However, the rate of inference is dependent on many factors. The most obvious is the performance of the machine running the inference, such as the CPU (Central Processing Unit) speed, whether any GPU (Graphical Processing Unit) acceleration is present, and the amount of system RAM (Random Access Memory) available.

There are also other factors. One is the machine learning framework used for training. There are many different frameworks available, such as TensorFlow, PyTorch, and Apache MXNet. These frameworks provide the tools and services to build machine learning models, including the training algorithms such as PPO (Proximal Policy Optimization) and SAC (Soft Actor Critic), which you might be familiar with from chapter **3.1 Training algorithms** in this module.

**Challenges**

However, what happens when sending the data to a central server for inference isn’t a good option and we need to perform inference locally? This is the situation with AWS DeepRacer. It would take too long to send the data to a remote server, perform the inference, and return the result to the AWS DeepRacer car. By that time the vehicle might have already driven off the track! In this situation we need to perform the inference locally on the device. We call this “inference at edge”.

Latency is the time that it takes to receive a result back from a remote computer from the moment the data is sent out from your device. In most cases, inference at the edge results in lower latency because the input doesn’t need to be sent out—It can all be done in one place. This is particularly important for the AWS DeepRacer device which needs to make real-time decisions. However, when you create a deep learning model for the AWS DeepRacer device, you have to consider that edge devices are generally less powerful and have less compute power than what is available in the cloud.

**Solutions**

To infer data through a deep learning model we need an inference engine. If you think of the inference engine as the car’s engine then the deep learning model is the fuel. This combination determines how fast and how efficient the car will run. You want to make sure the fuel (your model) is appropriate for the engine (the device) it is running on.

OpenVINO stands for Open Visual Inferencing and Neural Network Optimization and is Intel’s developer tool kit for machine learning inference. It contains a series of components and tools which help developers improve the inference performance of their models on Intel hardware, such as the AWS DeepRacer device, which has a compute module with an Intel Atom processor.

The Intel OpenVINO toolkit has the following components: the Deep Learning Deployment Toolkit (includes the model optimizer), inference engine, and pre-trained models. It also includes advanced tools and libraries for expert programmers.

**Optimization Process**

In the first step, we will run the model optimizer to prepare the pre-trained model for inferencing. It will generate a set of different files, known as an Intermediate Representation (IR), that the OpenVINO toolkit inference engine in AWS DeepRacer can understand and use. The model optimizer performs a number of optimizations on the model to make it run faster and more efficiently on the Intel Atom powered AWS DeepRacer. This is all done automatically when you export your model from the AWS DeepRacer Console, so that in most situations you’re done and can jump directly to step three and deploy the model to the AWS DeepRacer device.

The Post-Training Optimization Toolkit (POT) is a tool included with the OpenVINO toolkit, which can be used to make the model more efficient. Think of this tool as a slider with accuracy at one end and performance at the other.

The POT allows you to tune a model for extra performance by reducing the precision of the model and therefore, sacrificing some accuracy. Some Intel processors come with accelerators to turbo charge performance of lower precision models and this is where the POT is particularly handy, because it is similar to putting the right fuel in a car to make sure you get the best performance from the engine.

There are also benchmarking and accuracy checker tools available, which will allow you to get key performance and accuracy data for a model on the target hardware, such as AWS DeepRacer. This data is useful as you go through the process of refining and optimizing your model to understand how it impacts inference performance and accuracy.

## 3.4 Top Tips

* use specific reward things
* small changes have big impacts

* Drastic changes introduce instability
* make things simpler

* optimize complex functions
* observe and analysis

* reward often
* fast laps > slow laps

* dont count number of steps
