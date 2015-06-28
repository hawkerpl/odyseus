import optparse
import numpy as np
import learners.net_learner as L
from tools.common_run import get_coordinates, model_dict, read_net, add_common_options



if __name__ == "__main__":
    usage = "map_picture.png net_params.dat"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('-b', type='int', dest='batch_interval', default=120,
                      help='interval generation for best individual to be saved,'
                            '\n eg. --save-interval=2 saves best individual every 2 generations')
    parser.add_option('-p', action='store_true', dest='print_to_stdout', default=True,
                      help='print the fitness and number of generation to stdout, runs on --save-interval')
    parser = add_common_options(parser,["-i","-g","-m","-n","-s"])
    opts, args = parser.parse_args()

    map_filename = args[0]
    start, finish = get_coordinates(map_filename)
    ModelClass = model_dict[opts.model]
    ModelClass.inner_layer_nods = opts.inner_nods

    learner = L.ENAC()
    """learner = L.PGPE(learningRate = 0.3,
                    sigmaLearningRate = 0.15,
                    momentum = 0.0,
                    epsilon = 2.0,
                    rprop = False,
                    storeAllEvaluations = True)
    """
    net = ModelClass.random_net()
    if opts.initial_inds:
        net = read_net(opts.initial_inds, ModelClass)
    agent = L.LearningAgent(net, learner)
    #agent = L.OptimizationAgent(net, learner)
    model = ModelClass(start, finish, map_path=map_filename)
    env = L.OdyseoEnv(model)
    task = L.OdyseoTask(env, opts.sim_steps)
    experiment = L.EpisodicExperiment(task, agent)
    x = 0
    batch = opts.batch_interval
    while x < opts.generations:
        x += batch
        experiment.doEpisodes(batch)
        if opts.print_to_stdout:
            print x
        agent.learn()
        np.savetxt(args[1],agent.module.params,delimiter=',')
