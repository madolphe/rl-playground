# @TODO: write history function for each agent
# @TODO: print reward history
# @TODO: redefined environment variable

import gym
import numpy as np


class Agent:
    def __init__(self, activation_function, mutation_rate, nb_layers=1,):
        self.nb_layers = nb_layers
        self.weights = self.get_random_weight_matrix()
        self.activation_function = activation_function
        self.reward = 0
        self.weights_mutation_rate = mutation_rate

    @staticmethod
    def get_random_weight_matrix():
        return np.array([np.random.normal(0.5, 0.001) for _ in range(4)])

    def mutate(self):
        nb_to_update = int(np.round(self.weights.shape[0] * self.weights_mutation_rate))
        for _ in range(nb_to_update):
            mutation_rate = 0.02*np.random.random() - 0.01
            rdm_idx = np.random.randint(0, len(self.weights))
            self.weights[rdm_idx] *= (1+mutation_rate)

    def action(self, observations):
        logit = np.dot(self.weights, observations)
        activation = self.activation_function(logit)
        if activation < 0.5:
            return 0
        return 1

    def history(self):
        pass


class Population:
    def __init__(self, env, mutation_rate, population_size):
        self.agents = [Agent(self.sigmoid, 0.5, 1) for _ in range(population_size)]
        self.env = env
        self.nb_mutation = int(np.round(mutation_rate * population_size))
        self.population_size = population_size

    def __str__(self):
        description = ''
        for index, agent in enumerate(self.agents):
            description += 'Agent ' + str(index) + ': '
            description += str(agent.weights) + ' | '
            description += str(agent.reward) + ' \n '
        return description

    def get_reward_on_gen(self, seed):
        for agent in self.agents:
            agent.reward = 0
            self.env.seed(seed)
            init_obs = self.env.reset()
            done = False
            next_action = agent.action(init_obs)
            while done is False:
                observation, instant_reward, done, _ = self.env.step(next_action)
                next_action = agent.action(observation)
                agent.reward += instant_reward

    def update_gen(self):
        self.order_agents()
        self.agents = self.agents[:self.nb_mutation]
        print("{} agents are kept to be mutated".format(str(self.nb_mutation)))
        for agent in self.agents:
            agent.mutate()
        for i in range(self.population_size - len(self.agents)):
            self.agents.append(Agent(self.sigmoid, 0.5, 1))
        pass

    def order_agents(self):
        self.agents = sorted(self.agents, key=lambda x: x.reward, reverse=True)
        pass

    def show_best_agent(self):
        init_obs = self.env.reset()
        next_action = self.agents[0].action(init_obs)
        done = False
        while done is not True:
            env.render()
            observation, reward, done, _ = self.env.step(next_action)
            next_action = self.agents[0].action(observation)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


if __name__ == '__main__':
    nb_epoch = 20
    env = gym.make('CartPole-v1')
    pop = Population(env, 0.5, 5)
    seed = np.random.randint(1)
    for _ in range(nb_epoch):
        print("epoch number: ", _)
        pop.get_reward_on_gen(seed)
        pop.show_best_agent()
        pop.update_gen()
        print(pop)
    env.close()

