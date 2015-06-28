import optparse
from learners.genetic_learner import GeneticLearner, Task
from tools.common_run import get_coordinates, model_dict, read_net, add_common_options


if __name__ == "__main__":
    usage = "map_picture.png net_params.dat"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option('--pop', type='int', dest='population', help="size of population", default=50)
    parser.add_option('--mb', type='float', dest='mutation', help="mutation probability", default=0.2)
    parser.add_option('--cb', type='float', dest='crossover', help="crossover probability", default=0.5)
    parser.add_option('-p', action='store_true', dest='print_to_stdout', default=True,
                      help='print the fitness and number of generation to stdout, runs on --save-interval')
    parser = add_common_options(parser,["-i","-g","-m","-n","-s"])
    opts, args = parser.parse_args()

    map_filename = args[0]
    start, finish = get_coordinates(map_filename)
    ModelClass = model_dict[opts.model]
    ModelClass.inner_layer_nods = opts.inner_nods
    Task.ModelClass = ModelClass
    t = Task(start, finish, map_path=map_filename, number_of_sim_steps=opts.sim_steps)
    l = GeneticLearner(t, generations=opts.generations,
                print_to_stdout=opts.print_to_stdout, mutation_probability=opts.mutation,
                crossover_probability=opts.crossover, population=opts.population)
    if opts.initial_inds is not None:
        initial_net = read_net(opts.initial_inds, ModelClass)
        l.add_initial_individual(initial_net)
    l.savep = args[1]
    l.main()

