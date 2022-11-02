from Simulator import Simulator

# A class to train an agent against another agent.
class Trainer(Simulator):
	def before_action(self):
		self.prev_game = self.game

	def after_action(self, action):
		self.agent_x.learn(self.prev_game, action, self.game)
