from odyseus_model import *

class Task(object):
    def __init__(self, start_position, end_position, map_path):
        self.net_maker = OdyseusModel.random_net
        self.model = OdyseusModel(start_position, end_position, map_path=map_path, neural_net=OdyseusModel.random_net())

    def individual_net_steps_on_map(self, net, simulation_steps):
        self.model.net._setParameters(net)
        for _ in xrange(simulation_steps):
            self.model.step()
        return self.model.fitness()



from deap import base, creator
import random
from deap import tools


class Learner(object):

    def evaluate(self, ind):
        ind_np = np.array(ind)
        number_of_sim_steps = 100
        fitness = self.task.individual_net_steps_on_map(ind_np, number_of_sim_steps)
        return fitness,


    def __init__(self, task):
        self.task = task
        IND_SIZE = self.task.net_maker().params.shape[0]
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        toolbox = base.Toolbox()
        toolbox.register("attribute", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attribute, n=IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoints)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", Learner.evaluate)
        self.toolbox = toolbox


    def main(self):
        pop = self.toolbox.population(n=50)
        CXPB, MUTPB, NGEN = 0.5, 0.2, 40

        # Evaluate the entire population
        fitnesses = [self.toolbox.evaluate(self,ind) for ind in pop]
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        for g in range(NGEN):
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = map(self.toolbox.clone, offspring)

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [self.toolbox.evaluate(self,in_ind) for in_ind in invalid_ind]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring

        return pop
