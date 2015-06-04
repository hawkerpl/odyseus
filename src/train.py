from pybrain.structure import LinearLayer, SigmoidLayer, FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.tools.shortcuts import buildNetwork

"""
def create_net():
	n = FeedForwardNetwork()
	inLayer = LinearLayer(7)
	hiddenLayer = LinearLayer(3)
	outLayer = LinearLayer(2)
	n.addInputModule(inLayer)
	n.addModule(hiddenLayer)
	n.addOutputModule(outLayer)
	in_to_hidden = FullConnection(inLayer, hiddenLayer)
	hidden_to_out = FullConnection(hiddenLayer, outLayer)
	n.addConnection(in_to_hidden)
	n.addConnection(hidden_to_out)
	n.sortModules()
	return n
"""
class DummyNet(object):
	def activate(self, input):
		return 2.0, 0.1

def create_net():
	net = buildNetwork(7, 3, 2, bias=True, hiddenclass=LinearLayer)
	return net

def create_dummy_net():
	return DummyNet()

from pybrain.rl.environments.environment import Environment
from odyseus_model import OdyseusModel, first_model


class MapEnviroment(Environment):
	indim = 7
	outdim = 2
	def __init__(self, model):
		self.model = model

	def getSensors(self):
		return self.model.to_net_input()

	def performAction(self, action):
		self.model.do_action(action)

	def reset(self):
		self.model = model.reset()

from pybrain.rl.environments.task import Task

class MapTask(Task):

	def __init__(self, environment):
		self.env = environment
		self.lastreward = 0

	def performAction(self, action):
		self.env.performAction(action)
        
	def getObservation(self):
		sensors = self.env.getSensors()
		return sensors
    
	def getReward(self):
		reward = self.env.model.distance_to_destination()
		cur_reward = self.lastreward
		self.lastreward = reward
		return cur_reward

	@property
	def indim(self):
		return self.env.indim
    
	@property
	def outdim(self):
		return self.env.outdim








