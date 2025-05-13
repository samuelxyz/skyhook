# 2D Rigid Body simulator
import numpy as np
import constants

class Body:
    def __init__(self, state, node_posns, node_areas, node_sections, dens):
        self.node_posns = np.asarray(node_posns)
        self.node_areas = np.asarray(node_areas)
        self.node_sections = np.asarray(node_sections)
        self.dens = dens
        self.recalc_mass_properties()

        self.state = np.asarray(state) # x, y, theta, vx, vy, omega
        self.pos3 = self.state[0:3] # view into the state. x [m], y [m], theta [rad]
        self.vel3 = self.state[3:6] # vx [m/s], vy [m/s], omega [rad/s]
        self.acc3 = self.calc_accel() # m/s^2, rad/s^2

    def recalc_mass_properties(self):
        # self.node_posns adjusted so center of mass is at origin
        self.node_lens = np.ediff1d(self.node_posns, to_end=self.node_posns[-1]-self.node_posns[-2])
        self.node_masses = self.node_lens*self.node_areas*self.dens
        self.mass = np.sum(self.node_masses)
        center_of_mass =  np.dot(self.node_masses, self.node_posns)/self.mass
        self.node_posns -= center_of_mass
        self.rot_inertia = np.dot(self.node_masses, self.node_posns**2)

    def calc_accel(self, state=None):
        '''Input pos and vel as 3D with the third dimension being rotation.'''
        if state is None:
            state = self.state
        # grav = -constants.GM/np.sum(pos**2) * pos/np.linalg.norm(pos)
        # drag = -vel*1e-15 # TODO
        
        # body centered positions (but world frame for rotation)
        node_posns_rot = self.rotated_nodes(state)

        # full world frame (earth centered)
        node_posns_world = node_posns_rot + state[:2]
        pos_norms = np.linalg.norm(node_posns_world, axis=1)
        grav_forces = (-constants.GM*self.node_masses/pos_norms**3)[:,np.newaxis] * node_posns_world
        lin_acc = np.sum(grav_forces, axis=0)/self.mass

        torques = np.cross(node_posns_rot, grav_forces)
        alpha = np.sum(torques)/self.rot_inertia

        return np.r_[lin_acc, alpha]
    
    def rot_matrix(self, state=None):
        if state is None:
            state = self.state
        theta = state[2]
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])
    
    def rotated_nodes(self, state=None):
        '''Returns an array of body centered 2D position vectors (but rotated into world frame), Nx2'''
        if state is None:
            state = self.state
        return np.matvec(self.rot_matrix(state), np.column_stack((self.node_posns, np.zeros_like(self.node_posns))))
    
    def world_nodes(self, state=None):
        '''Returns an array of body centered 2D position vectors (rotated and translated into world frame), Nx2'''
        if state is None:
            state = self.state
        return self.rotated_nodes(state) + state[:2]

    def node_vels_world(self, state=None):
        '''Returns an array of 2D velocity vectors (in the world frame), Nx2'''
        if state is None:
            state = self.state
        return np.cross(self.rotated_nodes(state), np.asarray((0, 0, state[5])))[:, :2] + state[3:5]
    
    def rk4_step(self, stepsize):
        v1 = self.vel3 + stepsize/2 * self.acc3
        a1 = self.calc_accel(np.r_[self.pos3 + stepsize/2 * v1, v1])
        v2 = self.vel3 + stepsize/2 * a1
        a2 = self.calc_accel(np.r_[self.pos3 + stepsize/2 * v2, v2])
        v3 = self.vel3 + stepsize*a2
        a3 = self.calc_accel(np.r_[self.pos3 + stepsize*v2, v3])

        # self.pos += stepsize*(self.vel + stepsize/6 * (self.acc + a1 + a2))
        self.pos3 += stepsize/6 * (self.vel3 + 2*v1 + 2*v2 + v3)
        self.vel3 += stepsize/6 * (self.acc3 + 2*a1 + 2*a2 + a3)
        self.acc3 = self.calc_accel(np.r_[self.pos3, self.vel3])


