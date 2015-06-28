import numpy as np
from odyseus_model import OdyseusRecursiveTwoThrusters, OdyseusModel
import urllib2


model_dict = {"normal": OdyseusModel, "thrust": OdyseusRecursiveTwoThrusters}

def read_net(net_filename, ModelClass):
    params = np.genfromtxt(net_filename, delimiter=',')
    net = ModelClass.random_net()
    net._setParameters(params)
    return net

def get_coordinates(map_filename):
    coordinates_file = map_filename[:-4]+".dat"
    starting_coordinates = np.genfromtxt(coordinates_file, delimiter=',')
    return starting_coordinates

def add_common_options(parser,options):
    options_dict = {
        "-n": lambda p : p.add_option('-n', type='int', dest='inner_nods', help="number of nodes in the inner layer of net", default=8),
        "-i": lambda p : p.add_option('-i', type='str', default=None, help="initial individuals filepath", dest='initial_inds'),
        '-g': lambda p: p.add_option('-g', type='int', dest='generations', help="number of generations to learn", default=400),
        '-m': lambda p: p.add_option('-m', type='str', dest='model', help="type of model, can be \"thrust\" or \"normal\"",
                                     default="thrust"),
        '-s': lambda p: p.add_option('-s', type='int', dest='sim_steps', help="number of internal simulation steps", default=30)
    }
    for option in options:
        options_dict[option](parser)
    return parser

def read_net_from_web(adress, ModelClass):
    data = urllib2.urlopen(adress)
    params = np.genfromtxt(data, delimiter=',')
    net = ModelClass.random_net()
    net._setParameters(params)
    return net

