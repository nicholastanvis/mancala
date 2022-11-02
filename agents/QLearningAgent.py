from .Agent import Agent

# An agent that learns the Q function and determines the best action with it.
class QLearningAgent(Agent):
	# These values may need some tuning.
	def __init__(self):
		self.alpha = lambda n: 0.1
		self.gamma = 0.9
		self.q_values = dict()
		self.q_visits = dict()

	def encode(self, game):
		board = None
		if game.turn == 'x':
			board = game.board[0:6] + game.board[7:13]
		else:
			board = game.board[7:13] + game.board[0:6]

		return "".join([chr(b + 48) for b in board])

	def learn(self, current_game, action, next_game):
		encoded_current = self.encode(current_game)
		encoded_next = self.encode(next_game)

		if encoded_current not in self.q_values:
			self.q_values[encoded_current] = [0] * 6
			self.q_visits[encoded_current] = 0

		if encoded_next not in self.q_values:
			self.q_values[encoded_next] = [0] * 6
			self.q_visits[encoded_next] = 0

		self.q_visits[encoded_current] += 1

		reward = next_game.score(current_game.turn) - current_game.score(current_game.turn)
		visits = self.q_visits[encoded_current]
		q_current = self.q_values[encoded_current][action]
		max_q_next = max(self.q_values[encoded_next])

		self.q_values[encoded_current][action] = q_current + self.alpha(visits) * (reward + self.gamma * max_q_next - q_current)

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
