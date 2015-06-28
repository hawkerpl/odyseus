from matplotlib import pyplot as plt
from matplotlib import animation
from odyseus_model import OdyseusModel
import csv
import sys

class OnDestination(Exception):
    pass

class Visualize(object):
    ModelClass = OdyseusModel

    @staticmethod
    def plain_sensor(starting_point, radius):
        return plt.Circle(starting_point, radius, fc='r')

    def sensors_maker(self,starting_point):
        sensors = [Visualize.plain_sensor(starting_point, Visualize.ModelClass.SensorClass.radius) for _ in xrange(5)]
        return Visualize.ModelClass.update_sensor_array_by_center(sensors, starting_point, self.model.alpha)

    def __init__(self, model, save_path=None, save_fitness_path=None):
        self.model = model
        self.im = model.img
        self.save_path = save_path
        self.fig = plt.figure()
        self.fig.set_dpi(100)
        self.fig.set_size_inches(7, 6.5)
        self.ax = plt.axes()
        self.path_of_body, = self.ax.plot([], [], 'g-', ms=7)
        self.first = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes,color="red")
        self.second_text = self.ax.text(0.02, 0.90, '', transform=self.ax.transAxes,color="red")
        self.body = plt.Circle(model.position, Visualize.ModelClass.central_radius, fc='y')
        self.destination = plt.Circle(model.destination, Visualize.ModelClass.central_radius,fc='g')
        self.x_steps = []
        self.y_steps = []
        self.sensors = self.sensors_maker(model.position)
        self.ax.imshow(self.im)
        self.on_road_obj = 0
        self.fit_list = []
        self.max_size = max(self.model.tab.shape)
        self.save_fitness_path = save_fitness_path
        self.total_fitness = []

    def save_fitness(self, i):
        on_road = self.model.if_on_road()
        if on_road:
            self.on_road_obj += 1
        steps_to_destination = self.model.distance_to_destination() if on_road else 10000
        self.fit_list.append(self.max_size/(steps_to_destination+1))
        self.total_fitness.append((sum(self.fit_list), self.on_road_obj))

    def make_objects_tuple(self):
        return self.body, self.destination, self.sensors[0], self.sensors[1], self.sensors[2],self.sensors[3],\
               self.sensors[4], self.first, self.second_text, self.path_of_body

    def init(self):
        self.first.set_text('')
        self.second_text.set_text('')
        self.ax.add_patch(self.body)
        self.ax.add_patch(self.destination)
        self.path_of_body.set_data([], [])
        for i in xrange(5):
            self.ax.add_patch(self.sensors[i])
        rtuple = self.make_objects_tuple()
        return rtuple

    def move_sensors(self, new_central_object_position, alpha):
        return OdyseusModel.update_sensor_array_by_center(self.sensors, new_central_object_position, alpha)

    def write_fitness(self):
        with open(self.save_fitness_path,'w') as f:
            csv_out = csv.writer(f)
            csv_out.writerow(['f1','f2'])
            for row in self.total_fitness:
                csv_out.writerow(row)

    def animate(self, i):
        self.model.step(i)
        x1, y1, alpha1 = self.model.position[0], self.model.position[1], self.model.alpha
        self.save_fitness(i)
        if self.model.distance_to_destination() < 20:
            if self.save_path:
                self.fig.savefig(self.save_path)
            if self.save_fitness_path:
                self.write_fitness()
            print "Destination echieved !! Ending simulation"
            sys.exit()
        self.sensors = self.move_sensors((x1,y1),alpha1)
        self.x_steps.append(x1)
        self.y_steps.append(y1)
        self.path_of_body.set_data(self.x_steps, self.y_steps)
        self.path_of_body.set_markersize(7)
        self.body.center = (x1, y1)
        first_line_text = "V: "+"{:.2f}".format(self.model.v)
        first_line_text += " Alpha: "+str("{:.2f}".format(plt.np.rad2deg(self.model.alpha)))
        first_line_text += " F:"+str(self.total_fitness[-1])
        self.first.set_text(first_line_text)
        self.model.check_sensors()
        self.second_text.set_text("S: "+" ".join(["{:.2f}".format(s) for s in self.model.sensors_to_val_array()]))
        rtuple = self.make_objects_tuple()
        return rtuple

    def run(self):
        anim = animation.FuncAnimation(self.fig, self.animate,
                                       init_func=self.init,
                                       frames=1,
                                       interval=60,
                                       blit=True)
        plt.show()
        if self.save_path:
            self.fig.savefig(self.save_path)
        if self.save_fitness_path:
            self.write_fitness()
        # ready to go, start the process
