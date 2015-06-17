from matplotlib import pyplot as plt
from matplotlib import animation

from train import *


class Visualize(object):
    @staticmethod
    def plain_sensor(starting_point, radius):
        return plt.Circle(starting_point, radius, fc='r')

    def sensors_maker(self,starting_point,sensor_length):
        sensors = [Visualize.plain_sensor(starting_point, OdyseusModel.SensorClass.radius) for _ in xrange(5)]
        return OdyseusModel.update_sensor_array_by_center(sensors, starting_point, self.model.alpha)

    def __init__(self, model):
        self.model = model
        self.im = model.img
        self.fig = plt.figure()
        self.fig.set_dpi(100)
        self.fig.set_size_inches(7, 6.5)
        self.ax = plt.axes()
        self.angle_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes,color="red")
        self.position_text = self.ax.text(0.02, 0.90, '', transform=self.ax.transAxes,color="red")
        self.body = plt.Circle(model.position, OdyseusModel.central_radius, fc='y')
        self.destination = plt.Circle(model.destination,OdyseusModel.central_radius,fc='g')
        self.sensors = self.sensors_maker(model.position,model.SensorClass.pixel_sensor_range)
        self.ax.imshow(self.im)

    def make_objects_tuple(self):
        return self.body, self.destination, self.sensors[0], self.sensors[1], self.sensors[2],self.sensors[3], self.sensors[4], self.angle_text, self.position_text

    def init(self):
        self.angle_text.set_text('')
        self.position_text.set_text('')
        self.ax.add_patch(self.body)
        self.ax.add_patch(self.destination)
        for i in xrange(5):
            self.ax.add_patch(self.sensors[i])
        rtuple = self.make_objects_tuple()
        return rtuple

    def move_sensors(self, new_central_object_position, alpha):
        return OdyseusModel.update_sensor_array_by_center(self.sensors, new_central_object_position, alpha)


    def animate(self,i):
        self.experiment.doInteractions(30)
        self.agent.learn()
        self.agent.reset()
        x1, y1, alpha1 = self.model.position[0], self.model.position[1], self.model.alpha
        self.sensors = self.move_sensors((x1,y1),alpha1)
        self.body.center = (x1, y1)
        self.angle_text.set_text(str(self.model.sensor_array))
        self.position_text.set_text(str([s.center for s in self.model.sensors]))
        rtuple = self.make_objects_tuple()
        return rtuple

    def run(self):
        anim = animation.FuncAnimation(self.fig, self.animate,
                                       init_func=self.init,
                                       frames=1,
                                       interval=60,
                                       blit=True)
        plt.show()
        # ready to go, start the process
