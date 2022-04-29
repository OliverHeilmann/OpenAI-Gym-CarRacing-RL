import numpy as np
import tensorflow as tf
from utils import *
from replay_buffer import ReplayBuffer
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

# Ref: https://github.com/usmcamp0811/DDPG-RaceCar

class SamDDPGAgent:
	def __init__(self, action_space, model_outputs=None, noise_mean=None, noise_std=None):
		# Hyperparameters
		self.gamma = 0.99
		self.actor_lr = 0.00001
		self.critic_lr = 0.002
		self.tau = 0.005
		self.memory_capacity = 60000

		self.need_decode_out = model_outputs is not None
		self.model_action_out = model_outputs if model_outputs else action_space.shape[0]
		self.action_space = action_space

		#  Initialize noise generator to implement some type of randomness
		if noise_mean is None:
			noise_mean = np.full(self.model_action_out, 0.0, np.float32)
		if noise_std is None:
			noise_std = np.full(self.model_action_out, 0.2, np.float32)

		std = self.noise = NoiseGenerator(noise_mean, noise_std)

		# Initialize the replay buffer
		self.r_buffer = ReplayBuffer(mem_size=self.memory_capacity)

		self.actor_opt = Adam(self.actor_lr)
		self.critic_opt = Adam(self.critic_lr)
		self.actor = None
		self.critic = None
		self.target_actor = None
		self.target_critic = None

	def reset(self):
		self.noise.reset()

	# actor network that is going to "play the game"
	def build_actor(self, state_shape, name="Actor"):
		inputs = layers.Input(shape=state_shape)

		x1 = layers.Conv2D(8, kernel_size=(5, 5), strides=(4, 4), use_bias=False, activation="relu")(inputs)
		x2 = layers.Conv2D(16, kernel_size=(5, 5), strides=(4, 4), use_bias=False, activation="relu")(x1)
		x3 = layers.Conv2D(32, kernel_size=(3, 3), strides=(3, 3), use_bias=False, activation="relu")(x2)
		x4 = layers.Conv2D(64, kernel_size=(3, 3), strides=(3, 3), use_bias=False, activation="relu")(x3)
		x5 = layers.Flatten()(x4)
		x6 = layers.Dense(128, activation='relu')(x5)

		y = layers.Dense(self.model_action_out, activation='tanh')(x6)

		model = Model(inputs=inputs, outputs=y, name=name)
		model.summary()
		return model

	def build_critic(self, state_shape, name="Critic"):
		state_inputs = layers.Input(shape=state_shape)

		x1 = layers.Conv2D(8, kernel_size=(5, 5), strides=(4, 4), use_bias=False, activation="relu")(state_inputs)
		x2 = layers.Conv2D(16, kernel_size=(5, 5), strides=(4, 4), use_bias=False, activation="relu")(x1)
		x3 = layers.Conv2D(32, kernel_size=(3, 3), strides=(3, 3), use_bias=False, activation="relu")(x2)
		x4 = layers.Conv2D(64, kernel_size=(3, 3), strides=(3, 3), use_bias=False, activation="relu")(x3)
		x5 = layers.Flatten()(x4)

		action_inputs = layers.Input(shape=(self.model_action_out,))

		x6 = layers.concatenate([x5, action_inputs])
		x7 = layers.Dense(64, activation='relu')(x6)
		x8 = layers.Dense(128, activation='relu')(x7)

		y = layers.Dense(1)(x8)

		model = Model(inputs=[state_inputs, action_inputs], outputs=y, name=name)
		model.summary()
		return model

	def init_networks(self, state_shape):
		self.actor = self.build_actor(state_shape)
		self.critic = self.build_critic(state_shape)

		# Build target networks
		self.target_actor = self.build_actor(state_shape, name='TargetActor')
		self.target_critic = self.build_critic(state_shape, name='TargetCritic')

		# Copy parameters from action and critic
		self.target_actor.set_weights(self.actor.get_weights())
		self.target_critic.set_weights(self.critic.get_weights())

	def get_action(self, state, add_noise=True):
		prep_state = self.preprocess(state)
		if self.actor is None:
			self.init_networks(prep_state.shape)

		# Get result from a network
		tensor_state = tf.expand_dims(tf.convert_to_tensor(prep_state), 0)
		actor_output = self.actor(tensor_state).numpy()

		# if asked, add noise to the action taken
		if add_noise:
			actor_output = actor_output[0] + self.noise.generate()
		else:
			actor_output = actor_output[0]

		if self.need_decode_out:
			env_action = self.decode_model_output(actor_output)
		else:
			env_action = actor_output

		# Clip min-max
		env_action = np.clip(np.array(env_action), a_min=self.action_space.low, a_max=self.action_space.high)
		return env_action, actor_output

	def decode_model_output(self, model_out):
		return np.array([model_out[0], model_out[1].clip(0, 1), -model_out[1].clip(-1, 0)])

	def preprocess(self, img, greyscale=False):
		img = img.copy()
		# Remove numbers and enlarge speed bar
		for i in range(88, 93 + 1):
			img[i, 0:12, :] = img[i, 12, :]

		# Unify grass color
		replace_color(img, original=(102, 229, 102), new_value=(102, 204, 102))

		if greyscale:
			img = img.mean(axis=2)
			img = np.expand_dims(img, 2)

		# Make car black
		car_color = 68.0
		car_area = img[67:77, 42:53]
		car_area[car_area == car_color] = 0

		# Scale from 0 to 1
		img = img / img.max()

		# Unify track color
		img[(img > 0.411) & (img < 0.412)] = 0.4
		img[(img > 0.419) & (img < 0.420)] = 0.4

		# Change color of kerbs
		game_screen = img[0:83, :]
		game_screen[game_screen == 1] = 0.80
		return img

	def learn(self, state, train_action, reward, new_state):
		# save the transition back to the replay buffer
		prep_state = self.preprocess(state)
		prep_new_state = self.preprocess(new_state)
		self.r_buffer.save_move(prep_state, train_action, reward, prep_new_state)

		# Sample batch from buffer and convert them to tensors
		state_batch, action_batch, reward_batch, new_state_batch = self.r_buffer.sample_buffer()

		state_batch = tf.convert_to_tensor(state_batch)
		action_batch = tf.convert_to_tensor(action_batch)
		reward_batch = tf.convert_to_tensor(reward_batch)
		reward_batch = tf.cast(reward_batch, dtype=tf.float32)
		new_state_batch = tf.convert_to_tensor(new_state_batch)

		self.update_actor_critic(state_batch, action_batch, reward_batch, new_state_batch)

		# Update target networks
		self.update_target_network(self.target_actor.variables, self.actor.variables)
		self.update_target_network(self.target_critic.variables, self.critic.variables)

	@tf.function
	def update_actor_critic(self, state, action, reward, new_state):
		# Update critic
		with tf.GradientTape() as tape:
			new_action = self.target_actor(new_state, training=True)
			y = reward + self.gamma * self.target_critic([new_state, new_action], training=True)

			critic_loss = tf.math.reduce_mean(tf.square(y - self.critic([state, action], training=True)))

		critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
		self.critic_opt.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

		# Update actor
		with tf.GradientTape() as tape:
			critic_out = self.critic([state, self.actor(state, training=True)], training=True)
			actor_loss = -tf.math.reduce_mean(critic_out)  # Need to maximize

		actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
		self.actor_opt.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

	@tf.function
	def update_target_network(self, target_weights, new_weights):
		for t, n in zip(target_weights, new_weights):
			t.assign((1 - self.tau) * t + self.tau * n)

	def save_solution(self, path='models/'):
		self.actor.save(path + 'actor.h5')
		self.critic.save(path + 'critic.h5')
		self.target_actor.save(path + 'target_actor.h5')
		self.target_critic.save(path + 'target_critic.h5')

	def load_solution(self, path='models/'):
		self.actor = tf.keras.models.load_model(path + 'actor.h5')
		self.critic = tf.keras.models.load_model(path + 'critic.h5')
		self.target_actor = tf.keras.models.load_model(path + 'target_actor.h5')
		self.target_critic = tf.keras.models.load_model(path + 'target_critic.h5')
