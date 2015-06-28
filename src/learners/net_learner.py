from pybrain.rl.learners.directsearch.enac import ENAC
from pybrain.optimization import PGPE
from pybrain.rl.agents import LearningAgent, OptimizationAgent
from pybrain.rl.experiments.episodic import EpisodicExperiment
from pybrain.rl.environments.environment import Environment

class OdyseoEnv(Environment):
    # the number of action values the environment accepts
    outdim = 2
    # the number of sensor values the environment produces
    indim = 5

    def __init__(self, model):
        self.model = model

    def getSensors(self):
        self.model.check_sensors()
        return self.model.to_net_input()

    def performAction(self, action):
        self.model.do_action(action)

    def reset(self):
        self.model.reset()



from scipy import clip, asarray

from pybrain.rl.environments import EpisodicTask
from numpy import *

class OdyseoTask(EpisodicTask):
    """ A task is associating a purpose with an environment. It decides how to evaluate the observations, potentially returning reinforcement rewards or fitness values.
    Furthermore it is a filter for what should be visible to the agent.
    Also, it can potentially act as a filter on how actions are transmitted to the environment. """

    def __init__(self, environment, maxsteps=300):
        """ All tasks are coupled to an environment. """
        self.N = maxsteps
        EpisodicTask.__init__(self, environment)
        self.lastreward = 0
        self.t = 0

    def performAction(self, action):
        self.t += 1
        self.env.performAction(action)

    def getObservation(self):
        sensors = self.env.getSensors()
        return sensors

    def getReward(self):
        on_road = self.env.model.if_on_road()
        steps_to_destination = self.env.model.distance_to_destination()# self.env.model.path_to_destination() if on_road else 0
        max_size = max(self.env.model.tab.shape)
        distance_reward = 2*max_size/(steps_to_destination+1+self.t)
        on_road_reward = 0 if on_road else 10000
        reward = distance_reward + on_road_reward
        cur_reward = self.lastreward
        self.lastreward = reward
        return cur_reward

    def reset(self):
        self.env.reset()
        EpisodicTask.reset(self)
        self.t = 0

    def isFinished(self):
        if self.t >= self.N: #and self.env.model.if_on_road():
            return True
        return False

    @property
    def indim(self):
        return self.env.indim

    @property
    def outdim(self):
        return self.env.outdim


