import numpy as np 
import PIL.Image
from matplotlib import transforms

class OdyseusModel(object):
	def __init__(self,starting_point,starting_alpha = 0.0, v = 0.0,map_path=None,neural_net = None, radius=30, pixel_sensor_range=50, dt=20):
		self.img = PIL.Image.open(r"F:\repozytoria\studia\ai\odyseusz\src\maps\simple2.png")
		self.tab = np.array(self.img)
		self.radius = radius
		self.v = v
		self.dt = dt
		self.sensor_angles = [90,45,0,-45,-90]
		self.position = starting_point
		self.alpha = starting_alpha
		self.pixel_sensor_range = pixel_sensor_range
		if neural_net:
			self.net = neural_net
		else:
			self.net = OdyseusModel.random_net()

	@staticmethod
	def random_net():
		return None

	def one_sensor_position(self,angle, position, alpha):
	    x0,y0 = position
	    x = x0 + self.pixel_sensor_range
	    y = y0
	    t = transforms.Affine2D().rotate_deg_around(x0,y0,angle+np.rad2deg(alpha)).get_matrix()
	    x,y,_ = t.dot(np.array([x,y,1]).T)
	    return x,y

	def determine_sensor_signal(self,position):
		x,y = position
		value = self.tab[x,y,0]
		return value


	def one_sensor_signal(self,angle, position, alpha):
		position = self.one_sensor_position(angle, position, alpha)
		signal = self.determine_sensor_signal(position)
		return signal

	def check_sensors(self, position, alpha):
		return [self.one_sensor_signal(angle,position,alpha) for angle in self.sensor_angles]

	def net_step(self,sensor_array,v,alpha,i):
		dv = 1#/self.dt
		dalpha = 0.01#1.0#/self.dt
		return dv, dalpha

	def step(self,i):
		sensor_array = self.check_sensors(self.position,self.alpha)
		print sensor_array
		dv, dalpha = self.net_step(sensor_array,self.v,self.alpha,i)
		self.alpha += dalpha
		x,y = self.position
		dx = np.cos(self.alpha)*dv#*self.dt
		dy = np.sin(self.alpha)*dv#*self.dt
		self.v += dv
		
		#print dx, dy, dv, dalpha
		x = x+dx
		y = y+dy
		self.position = x, y
		return dx, dy, dalpha

	def as_genom():
		pass