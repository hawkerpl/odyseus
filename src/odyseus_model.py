import numpy as np 
import PIL.Image
from matplotlib import transforms

class OdyseusSensor(object):
	radius = 30
	pixel_sensor_range = 50

	@classmethod
	def create_position(cls, central_object_position, angle, alpha):
		x0, y0 = central_object_position
		x = x0 + OdyseusSensor.pixel_sensor_range
		y = y0
		t = transforms.Affine2D().rotate_deg_around(x0,y0,alpha+np.rad2deg(angle)).get_matrix()
		x,y,_ = t.dot(np.array([x,y,1]).T)
		return x,y

	def __init__(self, central_object_position, angle, alpha):
		x, y = OdyseusSensor.create_position(central_object_position, angle, alpha)
		self.center = x,y
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
	sensor_angles = [90,45,0,-45,-90]
	central_radius = 30

	def __init__(self,starting_point,starting_alpha = 0.0, v = 0.0,map_path=None,neural_net = None, radius=15, pixel_sensor_range=50, dt=20):
		self.img = PIL.Image.open(r"F:\repozytoria\studia\ai\odyseusz\src\maps\bw.png")
		self.tab = np.array(self.img)
		OdyseusModel.SensorClass.radius = radius
		OdyseusModel.SensorClass.pixel_sensor_range = pixel_sensor_range
		self.sensors = [OdyseusModel.SensorClass(starting_point,angle,starting_alpha) for angle in OdyseusModel.sensor_angles]
		self.sensor_array = []
		self.v = v
		self.dt = dt
		self.position = starting_point
		self.alpha = starting_alpha
		if neural_net:
			self.net = neural_net
		else:
			self.net = OdyseusModel.random_net()

	@staticmethod
	def random_net():
		return None

	@classmethod
	def update_sensor_array_by_center(cls, sensors, central_object_position, alpha):
		for sensor,angle in zip(sensors,cls.sensor_angles):
			sensor.center = cls.SensorClass(central_object_position, alpha, angle).center
		return sensors

	def update_sensors(self):
		self.sensors = OdyseusModel.update_sensor_array_by_center(self.sensors,self.position,self.alpha)
		return self.sensors
		

	def determine_sensor_signal(self,sensor):
		x,y = sensor.center
		value = self.tab[y,x,0]
		sensor.val = value
		return value

	def check_sensors(self):
		[self.determine_sensor_signal(sensor) for sensor in self.sensors]
		return self.sensors

	def sensors_to_val_array(self):
		return [s.val for s in self.sensors]

	def net_step(self,sensor_array,v,alpha,i):
		dv = 1#/self.dt
		dalpha = 0.01#1.0#/self.dt
		return dv, dalpha

	def step(self,i):
		self.sensors = self.check_sensors()
		self.sensor_array = self.sensors_to_val_array()
		dv, dalpha = self.net_step(self.sensor_array,self.v,self.alpha,i)
		self.alpha += dalpha
		x,y = self.position
		dx = np.cos(self.alpha)*dv#*self.dt
		dy = np.sin(self.alpha)*dv#*self.dt
		self.v += dv
		x = x+dx
		y = y+dy
		self.position = x, y
		self.update_sensors()
		return dx, dy, dalpha

	def as_genom():
		pass