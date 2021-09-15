import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

CLIP_EDGE = 1e-8


def print_state(state, step, reward=None):
    format_string = 'Step {0} - Cart X: {1:.3f}, Cart V: {2:.3f}, ' \
                    'Pole A: {3:.3f}, Pole V:{4:.3f}, Reward:{5}'
    print(format_string.format(step, *tuple(state), reward))


env = gym.make('CartPole-v0')
test_gamma = .5  # Please change me to be between zero and one
episode_rewards = [-.9, 1.2, .5, -.6, 1, .6, .2, 0, .4, .5]


def discount_episode(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)  # array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    total_rewards = 0
    for t in reversed(range(len(rewards))):
        # print(t)  # 9, 8, ..., 0
        total_rewards = rewards[t] + total_rewards * gamma
        discounted_rewards[t] = total_rewards
    return discounted_rewards


discount_episode(episode_rewards, test_gamma)


def custom_loss(y_true, y_pred):
    y_pred_clipped = K.clip(y_pred, CLIP_EDGE, 1-CLIP_EDGE)  # Element-wise value clipping
    log_likelihood = y_true * K.log(y_pred_clipped)
    return K.sum(-log_likelihood*g)


def build_networks(
        state_shape, action_size, learning_rate, hidden_neurons):
    """Creates a Policy Gradient Neural Network.

    Creates a two hidden-layer Policy Gradient Neural Network. The loss
    function is altered to be a log-likelihood function weighted
    by the discounted reward, g.

    Args:
        space_shape: a tuple of ints representing the observation space.
        action_size (int): the number of possible actions.
        learning_rate (float): the nueral network's learning rate.
        hidden_neurons (int): the number of neurons to use per hidden
            layer.
    """
    state_input = layers.Input(state_shape, name='frames')
    g = layers.Input((1,), name='g')

    hidden_1 = layers.Dense(hidden_neurons, activation='relu')(state_input)
    hidden_2 = layers.Dense(hidden_neurons, activation='relu')(hidden_1)
    probabilities = layers.Dense(action_size, activation='softmax')(hidden_2)

    def custom_loss(y_true, y_pred):
        y_pred_clipped = K.clip(y_pred, CLIP_EDGE, 1 - CLIP_EDGE)
        log_lik = y_true * K.log(y_pred_clipped)
        return K.sum(-log_lik * g)

    policy = models.Model(
        inputs=[state_input, g], outputs=[probabilities])
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    policy.compile(loss=custom_loss, optimizer=optimizer)

    predict = models.Model(inputs=[state_input], outputs=[probabilities])
    return policy, predict


space_shape = env.observation_space.shape
action_size = env.action_space.n

# Feel free to play with these
test_learning_rate = .2
test_hidden_neurons = 10

test_policy, test_predict = build_networks(
    space_shape, action_size, test_learning_rate, test_hidden_neurons)

state = env.reset()
test_predict.predict(np.expand_dims(state, axis=0))


class Memory:
    """Sets up a memory replay buffer for Policy Gradient methods.

    Args:
        gamma (float): The "discount rate" used to assess TD(1) values.
    """
    def __init__(self, gamma):
        self.buffer = []
        self.gamma = gamma

    def add(self, experience):
        """Adds an experience into the memory buffer.

        Args:
            experience: a (state, action, reward) tuple.
        """
        self.buffer.append(experience)

    def sample(self):
        """Returns the list of episode experiences and clears the buffer.

        Returns:
            (list): A tuple of lists with structure (
                [states], [actions], [rewards]
            }
        """
        batch = np.array(self.buffer).T.tolist()
        states_mb = np.array(batch[0], dtype=np.float32)
        actions_mb = np.array(batch[1], dtype=np.int8)
        rewards_mb = np.array(batch[2], dtype=np.float32)
        self.buffer = []
        return states_mb, actions_mb, rewards_mb


test_memory = Memory(test_gamma)
actions = [x % 2 for x in range(200)]
state = env.reset()
step = 0
episode_reward = 0
done = False

while not done and step < len(actions):
    action = actions[step]  # In the future, our agents will define this.
    state_prime, reward, done, info = env.step(action)
    episode_reward += reward
    test_memory.add((state, action, reward))
    step += 1
    state = state_prime

test_memory.sample()


class Partial_Agent:
    """Sets up a reinforcement learning agent to play in a game environment."""
    def __init__(self, policy, predict, memory, action_size):
        """Initializes the agent with Policy Gradient networks
            and memory sub-classes.

        Args:
            policy: The policy network created from build_networks().
            predict: The predict network created from build_networks().
            memory: A Memory class object.
            action_size (int): The number of possible actions to take.
        """
        self.policy = policy
        self.predict = predict
        self.action_size = action_size
        self.memory = memory

    def act(self, state):
        """Selects an action for the agent to take given a game state.

        Args:
            state (list of numbers): The state of the environment to act on.

        Returns:
            (int) The index of the action to take.
        """
        # If not acting randomly, take action with highest predicted value.
        state_batch = np.expand_dims(state, axis=0)
        probabilities = self.predict.predict(state_batch)[0]
        action = np.random.choice(self.action_size, p=probabilities)
        return action


test_agent = Partial_Agent(test_policy, test_predict, test_memory, action_size)
action = test_agent.act(state)
print("Push Right" if action else "Push Left")


def learn(self, print_variables=False):
    """Trains a Policy Gradient policy network based on stored experiences."""
    state_mb, action_mb, reward_mb = self.memory.sample()
    # One hot enocde actions
    actions = np.zeros([len(action_mb), self.action_size])
    actions[np.arange(len(action_mb)), action_mb] = 1
    if print_variables:
        print("action_mb:", action_mb)
        print("actions:", actions)

    # Apply TD(1) and normalize
    discount_mb = discount_episode(reward_mb, self.memory.gamma)
    discount_mb = (discount_mb - np.mean(discount_mb)) / np.std(discount_mb)
    if print_variables:
        print("reward_mb:", reward_mb)
        print("discount_mb:", discount_mb)
    return self.policy.train_on_batch([state_mb, discount_mb], actions)

Partial_Agent.learn = learn
test_agent = Partial_Agent(test_policy, test_predict, test_memory, action_size)

test_gamma = .9
test_learning_rate = .002
test_hidden_neurons = 50

with tf.Graph().as_default():
    test_memory = Memory(test_gamma)
    test_policy, test_predict = build_networks(
    space_shape, action_size, test_learning_rate, test_hidden_neurons)
    test_agent = Partial_Agent(test_policy, test_predict, test_memory, action_size)
    for episode in range(200):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = test_agent.act(state)
            state_prime, reward, done, info = env.step(action)
            episode_reward += reward
            test_agent.memory.add((state, action, reward))
            state = state_prime

        test_agent.learn()
        print("Episode", episode, "Score =", episode_reward)

