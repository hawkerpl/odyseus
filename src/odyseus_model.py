import numpy as np
import PIL.Image
from matplotlib import transforms
from pybrain.structure import GaussianLayer
from pybrain.tools.shortcuts import buildNetwork


class OdyseusSensor(object):
    radius = 30
    pixel_sensor_range = 50

    @classmethod
    def create_position(cls, central_object_position, angle, alpha):
        x0, y0 = central_object_position
        x = x0 + OdyseusSensor.pixel_sensor_range
        y = y0
        t = transforms.Affine2D().rotate_deg_around(x0, y0, alpha + np.rad2deg(angle)).get_matrix()
        x, y, _ = t.dot(np.array([x, y, 1]).T)
        return x, y

    def __init__(self, central_object_position, angle, alpha):
        x, y = OdyseusSensor.create_position(central_object_position, angle, alpha)
        self.center = x, y
        self.val = False

    @property
    def x(self):
        return self.center[0]

    @x.setter
    def x(self, value):
        self.center[0] = value

    @property
    def y(self):
        return self.center[1]

    @y.setter
    def y(self, value):
        self.center[1] = value


class OdyseusModel(object):
    SensorClass = OdyseusSensor
    sensor_angles = [90, 45, 0, -45, -90]
    central_radius = 30

    def __init__(self, starting_point, destination,
                 starting_alpha=0.0,
                 v=0.0,
                 map_path=None,
                 neural_net=None,
                 radius=15,
                 pixel_sensor_range=50,
                 dt=20):
        self.img = PIL.Image.open(map_path)
        self.tab = np.array(self.img)
        OdyseusModel.SensorClass.radius = radius
        OdyseusModel.SensorClass.pixel_sensor_range = pixel_sensor_range
        self.sensors = [OdyseusModel.SensorClass(starting_point, angle, starting_alpha) for angle in
                        OdyseusModel.sensor_angles]
        self.sensor_array = []
        self.starting_v = v
        self.starting_alpha = starting_alpha
        self.v = v
        self.dt = dt
        self.net = neural_net
        self.starting_position = starting_point
        self.position = starting_point
        self.destination = destination
        self.alpha = starting_alpha
        self.net = neural_net

    @staticmethod
    def random_net():
        return buildNetwork(7, 3, 2, bias=True, hiddenclass=GaussianLayer, outclass=GaussianLayer)

    @classmethod
    def update_sensor_array_by_center(cls, sensors, central_object_position, alpha):
        for sensor, angle in zip(sensors, cls.sensor_angles):
            sensor.center = cls.SensorClass(central_object_position, alpha, angle).center
        return sensors

    def update_sensors(self):
        self.sensors = OdyseusModel.update_sensor_array_by_center(self.sensors, self.position, self.alpha)
        return self.sensors


    def determine_sensor_signal(self, sensor):
        x, y = sensor.center
        value = self.tab[y, x, 0]
        sensor.val = value
        return value

    def check_sensors(self):
        [self.determine_sensor_signal(sensor) for sensor in self.sensors]
        return self.sensors

    def sensors_to_val_array(self):
        return [s.val for s in self.sensors]


    def to_net_input(self):
        net_input = self.sensors_to_val_array()
        net_input.append(self.v)
        net_input.append(self.alpha)
        return net_input

    def restrict_values(self, values):
        dv, dalpha = values
        dv_limit = 5
        dalpha_limit = 5
        if np.abs(dv) > dv_limit:
            dv = np.sign(dv) * dv_limit
        if np.abs(dalpha) > dalpha_limit:
            dalpha = np.sign(dalpha) * dalpha_limit
        return dv, dalpha

    def net_step(self, i):
        net_input = self.to_net_input()
        dv, dalpha = self.net.activate(net_input)
        dv, dalpha = self.restrict_values((dv, dalpha))
        return dv, dalpha

    def do_action(self, action):
        dv, dalpha = action #/ 10000.0
        self.alpha += dalpha
        x, y = self.position
        dx = np.cos(self.alpha) * dv  # *self.dt
        dy = np.sin(self.alpha) * dv  # *self.dt
        self.v += dv
        x = x + dx
        y = y + dy
        self.position = x, y
        self.update_sensors()
        return dx, dy, dalpha

    def fitness(self):
        distance = self.distance_to_destination()
        x, y = self.position
        position_pixel = self.tab[y, x, 0]
        punishment = 0
        if position_pixel == 0:
            punishment = 1000
        fitness = distance + punishment
        return fitness


    def distance_to_destination(self):
        x0, y0 = self.position
        x1, y1 = self.destination
        return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

    def step(self, i=1):
        self.sensors = self.check_sensors()
        self.sensor_array = self.sensors_to_val_array()
        dv, dalpha = self.net_step(i)
        self.do_action((dv, dalpha))
        #return dx, dy, dalpha

    def reset(self):
        self.tab = np.array(self.img)
        self.sensors = [OdyseusModel.SensorClass(starting_point, angle, starting_alpha) for angle in
                        OdyseusModel.sensor_angles]
        self.sensor_array = []
        self.v = self.starting_v
        self.position = self.starting_position
        self.alpha = self.starting_alpha
        return self

    @staticmethod
    def net_to_numpy(net):
        pass



def first_model():
    return OdyseusModel((200, 100), (400, 100), map_path=r"F:\repozytoria\studia\ai\odyseusz\src\maps\bw.png")