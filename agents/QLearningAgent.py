from .Agent import Agent

# An agent that learns the Q function and determines the best action with it.
class QLearningAgent(Agent):
	# These values may need some tuning.
	def __init__(self):
		self.alpha = lambda n: 0.1
		self.gamma = 0.9
		self.q_values = dict()
		self.q_visits = dict()

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

	def encode(self, game):
		board = None
		if game.turn == 'x':
			board = game.board[0:6] + game.board[7:13]
		else:
			board = game.board[7:13] + game.board[0:6]

		return "".join([chr(b + 48) for b in board])
