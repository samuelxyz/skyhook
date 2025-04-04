# 2D Rigid Body simulator
import numpy as np

GM = 3.986e5 # km^3/s^2, Earth
def accel(pos: np.ndarray, vel: np.ndarray):
    grav = -GM/np.sum(pos**2) * pos/np.linalg.norm(pos)
    # drag = -vel*1e-15 # TODO

    return grav

class Body:
    def __init__(self):
        self.state = np.asarray([7800., 0., 0., 8.]) # x, y, vx, vy
        # self.pos = np.asarray([7800, 0]) # km
        # self.vel = np.asarray([0, 9]) # km/s
        self.pos = self.state[0:2] # view into the state
        self.vel = self.state[2:4]
        self.acc = accel(self.pos, self.vel) # km/s^2
    
    def rk4_step(self, stepsize):
        v1 = self.vel + stepsize/2 * self.acc
        a1 = accel(self.pos + stepsize/2 * v1, v1)
        v2 = self.vel + stepsize/2 * a1
        a2 = accel(self.pos + stepsize/2 * v2, v2)
        v3 = self.vel + stepsize*a2
        a3 = accel(self.pos + stepsize*v2, v3)

        # self.pos += stepsize*(self.vel + stepsize/6 * (self.acc + a1 + a2))
        self.pos += stepsize/6 * (self.vel + 2*v1 + 2*v2 + v3)
        self.vel += stepsize/6 * (self.acc + 2*a1 + 2*a2 + a3)
        self.acc = accel(self.pos, self.vel)


