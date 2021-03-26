# Project Architecture - Reinforcement Learning in an Imagined Latent State

<!-- TOC -->
- [Prior Setup](#prior-setup)
  - [Initial Data Collection](#initial-data-collection)
  - [Reward Signal formulation](#reward-signal-formulation)
  - [Dataset Construction/Consolidation](#dataset-constructionconsolidation)
- [Maintenance Detection Model](#maintenance-detection-model)
  - [Maintenance Requirements](#maintenance-requirements)
  - [Detection Model](#detection-model)
  - [Quality Estimator](#quality-estimator)
- [State Encoder (SE)](#state-encoder-se)
- [Recurrent State Space Model (RSSM)](#recurrent-state-space-model-rssm)
- [Reinforcement Learning Model (Actor Critic / AC)](#reinforcement-learning-model-actor-critic--ac)
- [Putting Everything Together](#putting-everything-together)
  - [Model Learning](#model-learning)
  - [Agent Learning](#agent-learning)
  - [Deployment](#deployment)
  - [Active Learning](#active-learning)
<!-- TOC -->

<hr>

## Prior Setup
Several assumptions are needed for this system to work.

### Initial Data Collection
As with any deep learning task, we require data. To obtain this data, ideally, the drone is initialy operated by a human operator in the intended environment. The data that is to be collected comes from a range of sensors, the most important ones are:
  - Perceptual cameras
    - cameras that allow the drone a 3D perception of the world
  - Sensing camera
    - Camera pointed downward looking at the track/road, used for detecting maintenance needs
  - Typical IMU suite
  - LIDAR or ultrasonic sensors scanning on a horizontal plane
    - Object proximity detection
- Throughout the human-piloted flight, collect at every time step:
  - Perceptual camera footage
    - $X = \{x_0, x_1, ..., x_T\} \in \mathbb{R}^{n_1 \times n_2}$
  - Control inputs
    - $A = \{a_0, a_1, ..., a_T\} \in \mathbb{R}^m$
  - Sensor data (IMU, LIDAR, etc)
    - $Q = \{q_1, q_2, ..., q_T\} \in \mathbb{R}^o$
  - Maintenance camera footage
    - $M = \{m_1, m_2, ..., m_T\} \in \mathbb{R}^{p_1 \times p_2}$

### Reward Signal formulation
Because a majority of the system is based on reinforcement learning, there needs to be a reward signal. We formulate the reward signal at every time step $i$ according to:

$$
  r_i = f(\hat{q}_{image, i}, q_i, )
$$

where $\hat{q}_{image, i}$ denotes the [perceived image quality](#quality-estimator) of the [maintenance detection model](#maintenance-detection-model). This allows the attainment of a new set of data, known here as the reward signal.
- Reward signals
  - $R = \{r_1, r_2, ..., r_T\} \in \mathbb{R}$

### Dataset Construction/Consolidation
The training dataset will then consist of the three classical requirements for reinforcement learning
- State
  - $S = \{s_1, s_2, ..., s_T\}$
  - $s_i = \{x_i, q_i\} \forall i \in [1, 2, ..., T]$
  - Optionally keep $X$ a separate variable from $S$, explained [later](#recurrent-state-space-model)
- Action
  - $A = \{a_0, a_1, ..., a_T\} \in \mathbb{R}^m$
  - Nothing special about this, it's the same one as [above](#initial-data-collection)
- Reward
  - $R = \{r_1, r_2, ..., r_T\} \in \mathbb{R}$
  - Again, nothing special here, same as [above](#reward-signal-formulation)

<hr>

## Maintenance Detection Model
There are two core purposes of the computer vision model. The first is to predict any and all [maintenance requirements](#maintenance-requirements), as part of the task at hand. The second is to provide part of the [reward signal](#reward-signal-formulation) to the performance of the flight in the form of a [quality estimate](#quality-estimator). The core architecture of the maintenance detection model should then resemble the following model:

![Maintenance Model](Maintenance%20Model.png)

### Maintenance Requirements
Some of the maintenance requirements taken from literature include the following:
- **Railway**
    - Missing fasteners
    - Cracked sleepers
    - Switch inspection
    - Things that are difficult to detect
      - Track twist
      - Spring nest displacement
      - Rail longitudinal/lateral deflection from normal
- **Road**
  - Cracks
    - Alligator cracks
    - Block cracking
    - Longitudinal/lateral cracking
    - Reflection cracking
  - Rutting
  - Potholes
  - Other debris

> Classically, detection of maintenance on rails requires laser scanners for rail deflection. Road maintenance tools classically include lidar, ultrasonic measurement, and IR measurements. For this reason, the maintenance issues are generally limited to a subset of existing maintenance issues because a more sophisticated suite of sensors are needed (this restriction is mostly in the rail, in addition, using these sensors on a flying drone is virtually impossible because they measure defects in the order of 1-10 mm, the height deviation from mere hovering will demolish this precision).

### Detection Model
The detection model should not be hard to develop, seeing as computer vision with a given dataset is, in this day and age, a trivial task to solve.

Some good brush-up papers are:
  - [Object Detection with Deep Learning, A Review](https://ieeexplore.ieee.org/abstract/document/8627998/?casa_token=cYO3ecdowBoAAAAA:NrqIoOjyfDbZyeBp5cTnIx4jBeHSsbfZ3KrFESme3p82scQOOINHqVc4ATvJnrssrHNT8s2q5sU)
  - [Image Segmentation with Deep Learning, A Survey](https://arxiv.org/abs/2001.05566)
  - [FasteNet, my paper](https://arxiv.org/abs/2012.07968)

This part of the system should not be a concern.

### Quality Estimator
When performing detection/segmentation on the images, it is not possible for all the images to be well represented in the training dataset. To detect less-than-ideal images during deployment, we append an auxialliary branch to the computer vision model that predicts a quality metric, $\hat{q}_{image}$.

Training this branch is easy: after the main branch of the model is trained, we add less-than-ideal images to the base image dataset $M$ and freeze the detection branch. Then, make the auxilliary branch predict the error in detection between the ideal case and the non-ideal case.

<hr>

## State Encoder (SE)
When performing conventional path planning on drones, typically, the flow of information can be represented by the following model:

![Conventional Estimation and Control](State%20Estimation%20and%20Control.png)

This presents several challenges in design:
- How do we represent the state of the the entire system? Global position, obstacles, world features, track position, etc...
- What's important for a good flight path? If the core computer vision model is regarded as a data-driven process, inherently there is no metric or heuristic that can be optimized upon to get a good flight path.
- What sensors are necessary, and how many of them do we need at what precision?

Admittedly, while an architecture where everything is deterministic and has an explicit formulation is possible, it is not entirely scalable from one scenario to the next. In addition, the phd title has 'AI-enabled drones' in it, may as well take that notion to the extreme...

The core idea of the state encoder is to allow the drone to represent the system state, $s \in S$ in the form of a hidden latent variable, $z \in Z \sim p_{e, \theta}(z|s)$. Hidden here implies that the meaning of this variable is non-human-interpretable. Note that this is a stochastic distribution. We denote with the vector $\theta$ distributions that are parameterized by a neural network.

The purpose of this is to reduce the very high-dimensional state information $s \in S$ into a much lower dimensional latent space $z \in Z$, where $\dim(s) \approx 10e6$ and $\dim(z) \approx 10e2$. This is necessary because current reinforcement learning algorithms do not function well with very high dimensional data, so this reduction of data dimension is necessary. In the [future](#putting-everything-together), we also see that this allows a dramatic speed-up in training by allowing the neural network to perform training in many more batches than when training with raw states $s$.

To achieve this meaningful encoding of the system state, a modified [variational auto encoder](https://arxiv.org/abs/1312.6114) can be used. The paper in the link is rather dated, but should give a good representation of what is planned to be achieved. In short, we represent the latent variable according to $z \in Z \sim p_{e, \theta}(z|s)$, and then have another distribution reproduce $\hat{s} \in \hat{S} \sim p_{d, \theta}(\hat{s}|z)$. We then regress $\hat{s}$ onto $s$ through standard back propogation. To enforce closeness in the space of $Z$ (to prevent overfitting), an additional distribution loss of $L_{KL} = KL(p(Z)||p(\bar{Z}))$ can be added, where $\bar{Z}$ is some prior, predefined distribution, usually chosen as a Gaussian ball with unit variance. The $KL$ term simply denotes the Kullback-Leibler divergence, which is a measure of distances between distributions. The formulation classically used is:

$$
  L_{KL} = KL(p(Z)||p(\bar{Z})) \\
  L_{KL} = \mathbb{E}_Z [log(\frac{p_{e, \theta}(z|s)}{p(\bar{z})})] \\
  L_{KL} = \frac{1}{T}\sum_i^T[log(p_{e, \theta}(z_i|s)) - log(p(\bar{z}_i))]
$$

> Note that the distribution of $z$ is stochastic, that is, the neural network outputs a distribution, and then the latent variable $z$ is sampled from this distribution via the [reparameterization trick](https://arxiv.org/abs/1312.6114). It is therefore convenient to represent the distributions as simple Gaussians.

As can be seen, simply summing all the data at a sufficient scale is enough to enforce this constraint. Unfortunately, doing this can be disadvantageous by forcing the network to place all the state representations very closely within the latent space, potentially losing out on representational capacity. To circumvent this, I propose that we simply replace $\bar{z} = z_{i-1}$ (some notation change in the expectation is needed, but this is fairly trivial to do and the end result is the same). The intuition is that this enforces adjacent states $s$ (which are by definition very close to each other being only one time step away from each other) to be represented closely in latent space, but states that are very far away (for example $s_3$ and $s_{100}$) can be placed very far apart in latent space. By intuition, in the limit, all the latent space variables will eventually be a coherent mass, but this still requires mathematical proof.

> Optionally, to enforce a _meaningful_ representation of the state $s$ in $z$, we can train the SE to output segmentation masks generated by an off-the-shelf semantic segmentation model. This way, things like tree colour, sky colour, etc, are not represented in the latent variable $z$.

<hr>

## Recurrent State Space Model (RSSM)

Online reinforcement learning is not data efficient. Many attributes cause this, most notable the sparsity of rewards and the exploration issue with learning from existing data. To circumvent this, many model-based reinforcement learning architectures have been proposed, where a world model is learned via supervised learning, and then reinforcement learning is done using this world model. This is the exact approach used here, except with latent space variables.


> We denote the latent state space variable here as $z \in Z$, but note that there are two possible formulations for this latent space variable. The first is as described in the [state encoder](#state-encoder), but an alternate, more sensible approach, is to have $s \in S$ be solely derived from $X$, and then concatenate $q \in Q$ into $z$. That way, the state encoder is only used to encode the perceptual camera information, and the sensor measurements taken by the IMU (or optionally filtered and estimated position, velocity, etc) be kept as a separate variable, simply concatenated to the latent variable. For brevity, I will just refer to this variable as a latent state variable $z \in Z$.

The RSSM's main goal is to predict the next state and reward given the current state and action pair, where the states are represented as latent state variables. If we assume that the whole process is first-order Markovian (as in a fully observable MDP), then it is sufficient for the neural networok to be a normal feedforward network. However, most of the time, such complex systems are partially observable, thus the Markovian nature is very difficult to be enforced. For this reason, it is likely that a recurrent model should be used for the RSSM instead, in the form of a GRU, LSTM, simple RNN, or optionally a Transformer. Thus, we introduce a recurrent variable here as $h$. The RSSM is then described by $(z_i, r_i) \sim f_{\theta}(z_{i-1}, a_{i-1}, h_{i-1})$. Note that this process is deterministic. The recurrent varialble can then be a distribution of several things; $h_i \sim  g(\bullet | z_i, h_{i-1}, ...)$, note that this is not a neural network process, but a fixed stochastic operation.

<hr>

## Reinforcement Learning Model (Actor Critic / AC)
The AC model will arguably be the most touchy component of the whole system architecture, but its core concept is the easiest to understand. The base idea is to have the actor produce a distribution over the action space of the most likely action that will maximize the expected sum of future discounted rewards. Ie:

$$
  \max_\theta \mathbb{E}_{z_i \sim c(z_{i-1}, \hat{a}), \hat{a} \sim \pi_\theta(\bullet | z)} V_i \\
  V_i = \sum_{t=i}^{T} \gamma^{t-i} r_t
$$

where $\gamma$ is some discount factor in $[0, 1]$, usually chosen in the range of 0.9. And the expectation is done over the state transition distribution $z_i \sim c(\bullet | z_{i-1}, a)$ and actions are taken according to an action policy $a \sim \pi_\theta(\bullet | z)$.

There exist several methods of solving this problem, the popular and successful ones are usually simple to understand with slightly involved math. I will just link some of the more popular ones here with increasing levels of difficulty:
- [Deterministic Policy Gradient](http://proceedings.mlr.press/v32/silver14.html)
- [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)
- [Pathwise Derivatives](http://proceedings.mlr.press/v89/jankowiak19a/jankowiak19a.pdf)
- [What matters in on policy learning? A large scale empirical study](https://arxiv.org/abs/2006.05990)
- [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

Optionally, read this [blog post](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html) on reinforcement learning algorithms and [this excellent blog](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html) post on all current SOTA reinforcement learning algorithms for on policy learning.

In short, the actor critic model requires an action value distribution approximator and a value function approximator, both in the form of neural netowrks:

$$
  \hat{a}_i \sim \pi_{a, \theta}(\bullet | z_i) \\
  \hat{V}_i = \pi_{v, \theta}(z_i)
$$

<hr>

## Putting Everything Together
Thus far, we have:
- [The State Encoder (and decoder)](#state-encoder-se)
  - $z \sim p_{e, \theta} (\bullet | s)$
  - $\hat{s} \sim p_{d, \theta} (\bullet | z)$
- [The Recurrent State Space Model](#recurrent-state-space-model-rssm)
  -  $(z_i, r_i) = f_{\theta}(z_{i-1}, a_{i-1}, h_{i-1})$
-  [Actor Critic Model](#reinforcement-learning-model-actor-critic--ac)
   - $\hat{V}_i = \pi_{v, \theta}(z_i)$
   - $\hat{a}_i \sim \pi_{a, \theta}(\bullet | z_i)$

To preface, the base idea is to train the AC model on the imagined latent states generated by the RSSM, rolled out for $n$ steps into the future at a time.

There are two main regimes for the training of this system.
- [Model Learning](#model-learning)
  - This is where the dataset is used to build the RSSM and the SE
- [Reinforcement Learning](#reinforcement-learning)
  - This is where the actual training of the actor critic agent is done

After these two are done, the system is decomposed to its core components, and a subset of them are used during [deployment](#deployment).

### Model Learning
In this stage of learning, we first train the SE using the dataset. Training to completion may not be too beneficial, but this remains to be seen. Regardless, some pre-training is required to allow the RSSM to have reasonable latent space variables to regress to, which is done right after the SE is trained/half-trained. A figure depicting this step of the training regime is shown here (dotted lines represent KL loss, or a loss function between two distributions):

![RSSM SE training](RSSM%20SE%20training.png)
****

### Agent Learning
Once the world model is suitably learned, it can then be used to train the reinforcement learning agent. This is much more efficient because now, the state of the world is encoded in a compressed form, and performing computations on this compressed state representation accelerates learning by allowing training to be done in very large batch sizes. To train, we simply encode a state $s_i$ at a time $t$ using the encoder, and then use the RSSM to perform rollouts given actions from the actor critic model. These rollouts can be performed in the field of 5 seconds to prevent the RSSM from accumulating errors and venturing into unrepresented areas of the state space. Given that the dataset is full of state data, we can then randomly sample the initial image from the dataset, and then perform rollouts on each image. This is graphically represented as:

![AC learning](AC%20learning.png)

> Optionally, to accelerate learning, we can also use mimic learning via supervised learning to first learn a good set of actions to take given states from the dataset. Once this is learned sufficiently well, reinforcement learning can be used to fine tune the actor to take more optimal actions. This is shown as:
>
> ![AC prelearning \label{prelearning}](AC%20prelearning.png)

### Deployment
Finally, in deployment, we discard everything but the most essential. Effectively, we replicate the operation shown in the prior image, but this time, actions are directly taken onto the world state instead. The RSSM and decoder are not used.

Ideally, the system should be trained in a recurring fashion, that is, after training on the initial dataset, the system should be allowed to interact with the environment on its own and collect additional data to be added to the dataset, and then retrained on this additional data. This should be done for several episodes until an optimum level of performance is achieved.

### Active Learning
During complete deployment, there is an option to perform active learning with the system. Because the computer vision model conveniently outputs a measure of performance, we can use this metric to detect instances where the model is having trouble with its task. These instances can then be collected, and the network explicitly trained on these scenarios after the fact. This is known as hard negative mining.








