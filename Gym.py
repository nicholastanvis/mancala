from Game import Game

# A class to train an agent (Agent X) against another agent (Agent Y).
class Gym():
	def __init__(self, agent_x, agent_y, **options):
		self.game = Game()
		self.agent_x = agent_x
		self.agent_y = agent_y

		if "silent" in options:
			self.silent = options["silent"]
		else:
			self.silent = False

	def start(self, episodes=1000):
		episode_interval = 20
		total_score = 0
		for episode in range(episodes):
			self.game = Game()
			self.agent_x.compute_epsilon(episode + 1)

			while not self.game.is_over():
				game = self.game

				action = -1
				if game.turn == 'x':
					action = self.agent_x.act(game)
					next_game = game.action(action)
					self.agent_x.learn(game, action, next_game)
					self.game = next_game
				else:
					action = self.agent_y.policy(game)
					self.game = game.action(action)
			
			total_score += game.score('x') - game.score('y')
			if (episode + 1) % episode_interval == 0 and not self.silent:
				print(f'Episode {episode + 1} | Epsilon: {str(round(self.agent_x.epsilon, 3))} | Average Score: {total_score / episode_interval}')
				total_score = 0
