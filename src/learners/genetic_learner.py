from odyseus_model import OdyseusModel, OdyseusRecursiveTwoThrusters
import numpy as np
from deap import base, creator
import random
from deap import tools, algorithms

class Task(object):
    ModelClass = OdyseusRecursiveTwoThrusters

    def __init__(self, start_position, end_position, map_path, number_of_sim_steps=100):
        self.net_maker = Task.ModelClass.random_net
        self.number_of_sim_steps = number_of_sim_steps
        self.model = Task.ModelClass(start_position, end_position, map_path=map_path, neural_net=Task.ModelClass.random_net())

    def individual_net_steps_on_map(self, net):
        self.model.reset()
        self.model.net._setParameters(net)
        fit_list = []
        on_road_obj = 0
        for i in xrange(self.number_of_sim_steps):
            self.model.step()
            on_road = self.model.if_on_road()
            if on_road:
                on_road_obj += 1
            steps_to_destination = self.model.distance_to_destination() if on_road else 10000
            max_size = max(self.model.tab.shape)
            fit_list.append(max_size/(steps_to_destination+1))
        return sum(fit_list), on_road_obj

class GeneticLearner(object):

    def evaluate(self, ind):
        ind_np = np.array(ind)
        fitness = self.task.individual_net_steps_on_map(ind_np)
        return fitness

    def __init__(self, task,
                 generations=40,
                 population=50,
                 print_to_stdout=False,
                 crossover_probability=0.5,
                 mutation_probability=0.2):
        self.population =  population
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.task = task
        self.ngen = generations
        self.initial_individuals = None
        self.print_to_stdout = print_to_stdout
        IND_SIZE = self.task.net_maker().params.shape[0]
        creator.create("FitnessMax", base.Fitness, weights=(1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("attribute", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attribute, n=IND_SIZE)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoints)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=2, indpb=0.1)
        toolbox.register("select", tools.selRoulette)
        toolbox.register("evaluate", GeneticLearner.evaluate, self)
        self.toolbox = toolbox

    def savepath(self,p):
        self.savep = p

    def add_initial_individual(self, individual):
        if self.initial_individuals is None:
            self.initial_individuals = [individual]
        else:
            self.initial_individuals.append(individual)

    def inject_initial_individuals(self, population):
        if self.initial_individuals is not None:
            for i, ind in enumerate(self.initial_individuals):
                population[i] = creator.Individual(ind.params)
        return population

    def main(self):
        pop = self.toolbox.population(self.population)
        pop = self.inject_initial_individuals(pop)
        CXPB, MUTPB, NGEN = self.crossover_probability, self.mutation_probability, self.ngen
        hof = tools.HallOfFame(1)
        stats = tools.Statistics()
        def fun(x):
            np.savetxt(self.savep, np.array(hof), delimiter=',')
            if self.print_to_stdout:
                print "fitness", self.evaluate(hof[0])
        stats.register("hofe", fun)
        algorithms.eaMuPlusLambda(pop, self.toolbox,lambda_=self.population, mu=self.population, cxpb=CXPB, mutpb=MUTPB, ngen=NGEN, stats=stats, halloffame=hof, verbose=True)

