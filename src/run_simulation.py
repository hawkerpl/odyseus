import optparse
from visualizer import Visualize
from tools.common_run import get_coordinates, model_dict, read_net, add_common_options, read_net_from_web



if __name__ == "__main__":
    usage = "\t map_picture.png net_params.dat\n or if tou want to download trained net from web: \n \t map_picture.png" \
            " http://adress.of.file \n where address of file must return non html plaintext with net in csv file separated " \
            "by comma, and the adress must start with \"http://\""
    parser = optparse.OptionParser(usage=usage)
    parser = add_common_options(parser,["-n","-m"])
    parser.add_option('-s', type='string', dest='save_path', help="path to save the plot after simulation", default=None)
    parser.add_option('-f', type='string', dest='save_fitness_path', help="path to save the fitness data after simulation", default=None)
    opts, args = parser.parse_args()
    map_filename = args[0]
    start, finish = get_coordinates(map_filename)
    net_filename = args[1]

    ModelClass = model_dict[opts.model]
    ModelClass.inner_layer_nods = opts.inner_nods
    net = None
    if net_filename[:7] == "http://":
        net = read_net_from_web(net_filename, ModelClass)
    else:
        net = read_net(net_filename, ModelClass)
    model = ModelClass(start, finish, map_path=map_filename, neural_net=net)
    Visualize.ModelClass = ModelClass
    v = Visualize(model, save_path=opts.save_path, save_fitness_path=opts.save_fitness_path)
    v.run()
