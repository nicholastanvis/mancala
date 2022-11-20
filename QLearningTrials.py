from Simulator import Simulator
from agents import *
from Gym import Gym
import matplotlib.pyplot as plt

# [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
gammas = [(i + 90) / 100 for i in range(10)]
size = len(gammas)

def extract_best_discounting_factor(trials):
    won = [0] * size
    lose = [0] * size
    draw = [0] * size
    total_scores = [0] * size
    for trial in range(trials):
        for i, g in enumerate(gammas):
            learning_agent = QLearningAgent()
            training_agent = GreedyAgent()
            learning_agent.gamma = g
            trainer = Gym(learning_agent, training_agent, silent=True)
            trainer.start(1000)
            simulator = Simulator(trainer.agent_x, training_agent, silent=True)
            simulator.start()
            score = simulator.game.score('x') - simulator.game.score('y')

            if score > 0:
                won[i] += 1
            elif score < 0:
                lose[i] += 1
            else:
                draw[i] += 1
            
            total_scores[i] += score
        print(f'Trial {trial + 1} | score: {total_scores} | wins: {won}')
    best_gamma_wins = gammas[won.index(max(won))]
    best_gamma_scores = gammas[total_scores.index(max(total_scores))]
    print(f'Best gamma based on win rate: {best_gamma_wins} | {max(won)} wins')
    print(f'Best gamma based on scoring: {best_gamma_scores} | {(max(total_scores)) / trials} average score')
    _, plt1 = plt.subplots()
    plt1.plot(gammas, won)
    plt1.set_title('Number of wins of each gamma value')
    plt1.set_xlabel('Gamma')
    plt1.set_ylabel('Count')

    average_score = [(score / trials) for score in total_scores]
    _, plt2 = plt.subplots()
    plt2.plot(gammas, average_score)
    plt2.set_title('Average score of each gamma value')
    plt2.set_xlabel('Gamma')
    plt2.set_ylabel('Score')
    plt.show()


extract_best_discounting_factor(1000)
