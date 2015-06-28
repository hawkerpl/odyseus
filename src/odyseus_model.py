import numpy as np
import PIL.Image
from matplotlib import transforms
from pybrain.structure import GaussianLayer, LinearLayer, SigmoidLayer
from pybrain.tools.shortcuts import buildNetwork
from tools import raytracer, pathfinder


class OdyseusSensor(object):
    radius = 10
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

class OdyseusBooleanSensor(OdyseusSensor):
    def __init__(self, central_object_position, angle, alpha):
        super(OdyseusBooleanSensor, self).__init__(central_object_position, angle, alpha)

    def determine_sensor_signal(self, tab, body_position):
        x, y = self.center
        try:
            value = tab[y, x, 0]
        except IndexError:
            value = 0
        self.val = value
        return value

class OdyseusLinearSensor(OdyseusSensor):
    def __init__(self, central_object_position, angle, alpha):
        super(OdyseusLinearSensor, self).__init__(central_object_position, angle, alpha)

    def determine_sensor_signal(self, tab, body_position):
        if 0 <= self.x < tab.shape[0] and 0<= self.y < tab.shape[1]:
            self.val = raytracer.trace_ray(tab, body_position, self.center)
            return self.val
        else:
            return 0

class OdyseusModel(object):
    SensorClass = OdyseusLinearSensor
    in_dim = 7
    out_dim = 2
    inner_layer_nods = 8
    sensor_angles = [90, 45, 0, -45, -90]
    central_radius = 15

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
        self.pathfinder_map = pathfinder.Pathfinder.map_from_binary_image(self.tab[:, :, 0])

    @classmethod
    def random_net(cls):
        return buildNetwork(cls.in_dim, cls.inner_layer_nods, cls.out_dim, bias=True, hiddenclass=SigmoidLayer, outclass=LinearLayer)

    @classmethod
    def update_sensor_array_by_center(cls, sensors, central_object_position, alpha):
        for sensor, angle in zip(sensors, cls.sensor_angles):
            sensor.center = cls.SensorClass(central_object_position, alpha, angle).center
        return sensors

    def update_sensors(self):
        self.sensors = OdyseusModel.update_sensor_array_by_center(self.sensors, self.position, self.alpha)
        return self.sensors

    def check_sensors(self):
        on_road = self.if_on_road()
        if on_road:
            [sensor.determine_sensor_signal(tab=self.tab, body_position=self.position) for sensor in self.sensors]
        else:
            for sensor in self.sensors:
                sensor.val = 0
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
        dv, dalpha = dv/10.0, dalpha/10.0
        dv_limit = 5
        dalpha_limit = 5
        if np.abs(dv) > dv_limit:
            dv = np.sign(dv) * dv_limit
        if np.abs(dalpha) > dalpha_limit:
            dalpha = np.sign(dalpha) * dalpha_limit
        return dv, dalpha

    def net_step(self, i):
        net_input = self.to_net_input()
        net_output = self.net.activate(net_input)
        return net_output

    def restrict_vel(self, vel):
        vel = max(-50,vel)
        vel = min(50,vel)
        return vel

    def restrict_dalpha(self, a):
        a = max(np.deg2rad(-5), a)
        a = min(np.deg2rad(5), a)
        return a

    def do_action(self, action):
        dv, dalpha = self.restrict_values(action)
        dalpha_restricted = self.restrict_dalpha(dalpha)
        alpha_restricted = self.alpha + dalpha_restricted
        x, y = self.position
        self.v += dv
        v_restricted = self.restrict_vel(self.v)
        dx = np.cos(alpha_restricted) * v_restricted/self.dt
        dy = np.sin(alpha_restricted) * v_restricted/self.dt
        self.v = v_restricted
        self.alpha += dalpha_restricted
        x = x + dx
        y = y + dy
        self.position = x, y
        self.update_sensors()
        return dx, dy, dalpha

    def if_on_road(self):
        x, y = map(int, self.position)

        try:
            position_pixel = self.tab[y, x, 0]
        except IndexError:
            position_pixel = 0
        finally:
            if position_pixel == 0:
                return False
            else:
                return True

    def fitness(self):
        distance = self.distance_to_destination()
        punishment = 0
        on_road = self.if_on_road()
        if not on_road:
            punishment = 1000
        fitness = distance + punishment
        return fitness


    def distance_to_destination(self):
        x0, y0 = self.position
        x1, y1 = self.destination
        return np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)

    def path_to_destination(self):
        end = map(int, self.destination)
        start = map(int, self.position)
        p = pathfinder.Pathfinder(self.pathfinder_map, 8, start, end)
        tuples_with_coordinates = p.run()
        path_length = len(tuples_with_coordinates)
        return path_length

    def step(self, i=1):
        self.sensors = self.check_sensors()
        self.sensor_array = self.sensors_to_val_array()
        net_output = self.net_step(i)
        self.do_action(net_output)
        #return dx, dy, dalpha


    def reset(self):
        self.sensors = [OdyseusModel.SensorClass(self.starting_position, angle, self.starting_alpha) for angle in
                        OdyseusModel.sensor_angles]
        self.sensor_array = []
        self.v = self.starting_v
        self.position = self.starting_position
        self.alpha = self.starting_alpha
        return self


class OdyseusNoRecursiveModel(OdyseusModel):
    in_dim = 5

    def __init__(self, *args, **kwargs):
        super(OdyseusNoRecursiveModel, self).__init__(*args, **kwargs)

    def to_net_input(self):
        net_input = self.sensors_to_val_array()
        return net_input

class OdyseusRecursiveTwoThrusters(OdyseusNoRecursiveModel):
    ax_radius = 20

    def __init__(self, *args, **kwargs):
        super(OdyseusRecursiveTwoThrusters, self).__init__(*args, **kwargs)
        self.v1 = 0.0
        self.v2 = 0.0

    def thrusters_to_vector(self, thrusters):
        axes = OdyseusRecursiveTwoThrusters.ax_radius
        dv1, dv2 = thrusters
        dv = (dv1+dv2)/2.0
        dalpha = np.arctan((dv1-dv2)/(2.0*axes))
        dv = dv *np.cos(dalpha)
        return dv, dalpha

    def restrict_values(self, values):
        thrust_dv_limit = 10
        values = min(values[0],thrust_dv_limit), min(values[1],thrust_dv_limit)
        values = max(values[0],0), max(values[1],0)
        return values

    def to_net_input(self):
        net_input = self.sensors_to_val_array()
        return net_input

    def do_action(self, action):
        dv1, dv2 = self.restrict_values(action)
        self.v1 += dv1
        self.v2 += dv2
        dv, dalpha = self.thrusters_to_vector((dv1, dv2))
        dalpha_restricted = self.restrict_dalpha(dalpha)
        self.alpha += dalpha_restricted
        self.v += dv
        v_restricted = self.restrict_vel(self.v)
        self.v = v_restricted
        x, y = self.position
        dx = np.cos(self.alpha) * self.v/self.dt
        dy = np.sin(self.alpha) * self.v/self.dt
        x = x + dx
        y = y + dy
        self.position = x, y
        self.update_sensors()
        return dx, dy, dalpha

def first_model():
    return OdyseusModel((200, 100), (400, 100), map_path=r"F:\repozytoria\studia\ai\odyseusz\src\maps\bw.png")