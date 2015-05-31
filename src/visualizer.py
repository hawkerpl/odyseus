import numpy as np
from matplotlib import transforms
from matplotlib import pyplot as plt
from matplotlib import animation
from odyseus_model import OdyseusModel

class Visualize(object):

  @staticmethod
  def plain_sensor(starting_point, radius):
    return plt.Circle(starting_point, radius, fc='r')

  def sensor_maker(self,starting_point,sensor_length):
    startx, starty = starting_point
    radius = 10
    sensors = [Visualize.plain_sensor((starting_point[0]+sensor_length,starting_point[1]), radius) for _ in xrange(5)]
    for sensor,alpha in zip(sensors,self.model.sensor_angles):
      t = transforms.Affine2D().rotate_deg_around(startx,starty,alpha).get_matrix()
      x,y = sensor.center
      x,y,_ = t.dot(np.array([x,y,1]).T)
      sensor.center = x,y
    return sensors 

  def __init__(self,model):
    self.model = model
    self.im = model.img
    self.fig = plt.figure()
    self.fig.set_dpi(100)
    self.fig.set_size_inches(7, 6.5)
    self.ax = plt.axes()
    self.angle_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes,color="red")
    self.position_text = self.ax.text(0.02, 0.90, '', transform=self.ax.transAxes,color="red")
    self.body = plt.Circle(model.position, model.radius, fc='y')
    self.sensors = self.sensor_maker(model.position,model.pixel_sensor_range)
    self.ax.imshow(self.im)

  def make_objects_tuple(self):
    return self.body, self.sensors[0],self.sensors[1],self.sensors[2],self.sensors[3],self.sensors[4], self.angle_text, self.position_text

  def init(self):
    #self.body.center = (200, 200)
    self.angle_text.set_text('')
    self.position_text.set_text('')
    self.ax.add_patch(self.body)
    for i in xrange(5):
      self.ax.add_patch(self.sensors[i])
    rtuple = self.make_objects_tuple()
    return rtuple

  def move_sensors(self,dx,dy,x0,y0,alpha1):
    #startx = x0+dx
    #starty = y0
    #for sensor,alpha in zip(self.sensors,self.model.sensor_angles):
    #  t = transforms.Affine2D().rotate_deg_around(startx,starty,self.model.alpha+alpha).get_matrix()
    #  x,y = sensor.center
    #  x,y,_ = t.dot(np.array([x,y,1]).T)
    #  sensor.center = x,y
    for sensor,angle in zip(self.sensors,self.model.sensor_angles):
      sensor.center = x0+dx+self.model.pixel_sensor_range, y0+dy
      t = transforms.Affine2D().rotate_deg_around(x0+dx,y0+dx,angle+np.rad2deg(alpha1)).get_matrix()
      x,y = sensor.center
      x,y,_ = t.dot(np.array([x,y,1]).T)
      sensor.center = x,y
    return self.sensors

  def animate(self,i):
    x, y = self.model.position
    dx,dy,dalpha = self.model.step(i)
    x1, y1, alpha1 = self.model.position[0], self.model.position[1], self.model.alpha
    self.sensors = self.move_sensors(dx,dy,x,y,alpha1)
    self.body.center = (x1, y1)
    rtuple = self.make_objects_tuple()
    self.angle_text.set_text(alpha1)
    self.position_text.set_text(str(x1)+" "+str(y1))
    return rtuple

  def run(self):
    anim = animation.FuncAnimation(self.fig, self.animate, 
                                   init_func=self.init, 
                                   frames=360, 
                                   interval=self.model.dt,
                                   blit=True)
    plt.show()

if __name__=="__main__":
  m = OdyseusModel((200,100))  
  Visualize(m).run()
