from .Agent import Agent
import math
import random

min_epsilon = 0.01
max_epsilon = 1.0
epsilon_decay = 250

# An agent that learns the Q function and determines the best action with it.
class QLearningAgent(Agent):
	# These values may need some tuning.
	def __init__(self):
		self.alpha = 0.1
		self.gamma = 0.9
		self.epsilon = 0.0
		self.q_values = dict()
		self.q_visits = dict()

	def encode(self, game):
		board = game.board[0:6] + game.board[7:13]
		return "".join([chr(b + 48) for b in board])

	def compute_epsilon(self, episode):
		self.epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)

	def learn(self, current_game, action, next_game):
		encoded_current = self.encode(current_game)
		encoded_next = self.encode(next_game)

		if encoded_current not in self.q_values:
			self.q_values[encoded_current] = [0] * 6
			self.q_visits[encoded_current] = 0

		if encoded_next not in self.q_values:
			self.q_values[encoded_next] = [0] * 6
			self.q_visits[encoded_next] = 0

		reward = next_game.score('x') - current_game.score('x')
		q_current = self.q_values[encoded_current][action]
		max_q_next = max(self.q_values[encoded_next])

		self.q_values[encoded_current][action] = q_current + self.alpha * (reward + self.gamma * max_q_next - q_current)

	def policy(self, game):
		max_q = float("-inf")
		best_action = -1

		encoded_game = self.encode(game)

		for action in game.actions():
			q = self.q_values[encoded_game][action] if encoded_game in self.q_values else 0
			if q <= max_q: continue

			max_q = q
			best_action = action
		return best_action

	def act(self, game):
		max_q = float("-inf")
		best_action = -1

		encoded_game = self.encode(game)

		if random.random() <= self.epsilon:
			best_action = random.randrange(len(game.actions()))
		else:
			for action in game.actions():
				q = self.q_values[encoded_game][action] if encoded_game in self.q_values else 0
				if q <= max_q: continue

				max_q = q
				best_action = action
		
		return best_action
