from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers.backprop import BackpropTrainer
from odyseus_model import OdyseusRecursiveTwoThrusters
import numpy as np
import optparse
from tools.common_run import add_common_options

if __name__ == "__main__":
    usage = "python run_pretrain.py resulting_file.neu \n this script pretrain net of model described in|" \
            " OdyseusRecursiveTwoThrusters, for 5 input nodes. Resulting network is saved to desired file"
    parser = optparse.OptionParser(usage=usage)
    opts, args = parser.parse_args()
    ds = SupervisedDataSet(5, 2)
    ds.addSample((50,50,50,0,0),(0,1))
    ds.addSample((0,50,50,50,0),(1,1))
    ds.addSample((0,0,50,0,0), (1,1))
    ds.addSample((0,0,50,50,50),(1,0))
    net = OdyseusRecursiveTwoThrusters.random_net()
    trainer = BackpropTrainer(net, ds)
    trainer.trainEpochs(3600)
    np.savetxt(args[0], trainer.module.params, delimiter=',')