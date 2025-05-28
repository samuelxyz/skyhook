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
        self.acc3 = self.calc_accel() # ax [m/s^2], ay [m/s^2], alpha [rad/s^2]

    def recalc_mass_properties(self):
        # self.node_posns adjusted so center of mass is at origin
        self.node_lens = np.ediff1d(self.node_posns, to_end=self.node_posns[-1]-self.node_posns[-2])
        self.node_masses = self.node_lens*self.node_areas*self.dens
        self.mass = np.sum(self.node_masses)
        center_of_mass =  np.dot(self.node_masses, self.node_posns)/self.mass
        self.node_posns -= center_of_mass
        self.rot_inertia = np.dot(self.node_masses, self.node_posns**2)

    def calc_accel(self, state=None):
        if state is None:
            state = self.state
        # drag = -vel*1e-15 # TODO
        
        # body centered positions (but world frame for rotation)
        node_posns_rot = self.rotated_nodes(state)

        # full world frame (earth centered)
        node_posns_world = node_posns_rot + state[:2]
        pos_norms = np.linalg.norm(node_posns_world, axis=1)
        grav_forces = (-constants.GM*self.node_masses/pos_norms**3)[:,np.newaxis] * node_posns_world # [Nx2]
        lin_acc = np.sum(grav_forces, axis=0)/self.mass

        torques = np.cross(node_posns_rot, grav_forces) # [Nx3]
        alpha = np.sum(torques)/self.rot_inertia # scalar

        return np.r_[lin_acc, alpha]
    
    def rot_matrix(self, state=None):
        '''Returns a 2x2 rotation matrix that rotates vectors from body frame to world frame'''
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
    
    def calc_local_forces(self, state=None):
        '''Returns Nx2 array of the 2D forces on each node in the body-anchored rotating frame.'''
        if state is None:
            state = self.state

        # body centered positions (but world frame for rotation)
        node_posns_rot = self.rotated_nodes(state)

        # full world frame (earth centered)
        node_posns_world = node_posns_rot + state[:2] # [Nx2]
        pos_norms = np.linalg.norm(node_posns_world, axis=1) # [N]
        grav_forces = (-constants.GM*self.node_masses/pos_norms**3)[:,np.newaxis] * node_posns_world # [Nx2]
        lin_acc = np.sum(grav_forces, axis=0)/self.mass # [2]

        torques = np.cross(node_posns_rot, grav_forces) # [Nx3]
        alpha = np.sum(torques)/self.rot_inertia # scalar
        alpha_vec = np.array((0, 0, alpha))

        omega_vec = np.asarray((0, 0, state[5]))
        # coriolis is zero for static structure
        centrifugal_acc = np.cross(omega_vec, np.cross(omega_vec, node_posns_rot))[:, :2] # per node, Nx2
        euler_acc = np.cross(alpha_vec, node_posns_rot)[:, :2] # Nx2

        world_forces = grav_forces - self.node_masses[:, np.newaxis] * (lin_acc + centrifugal_acc + euler_acc)
        forces = np.linalg.solve(self.rot_matrix(state)[np.newaxis, :, :], world_forces[:, :, np.newaxis])
        return np.squeeze(forces)
    
    def beam_analysis(self, state=None):
        if state is None:
            state = self.state
        
        forces = self.calc_local_forces(state) # [Nx2]

        # The following values will be calculated between each node for a total of [N+1] elements
        dx = self.node_posns[-1] - self.node_posns[-2]
        node_inter_posns = np.r_[self.node_posns - dx/2, self.node_posns[-1] + dx/2]
        tension = -np.cumulative_sum(forces[:, 0], include_initial=True)
        shear = -np.cumulative_sum(forces[:, 1], include_initial=True)

        # Interpolate back down from [N+1] to [N]
        tension = np.interp(self.node_posns, node_inter_posns, tension)
        shear = np.interp(self.node_posns, node_inter_posns, shear)
        tension_stress = tension/self.node_areas
        shear_stress = shear/self.node_areas

        moment = -np.cumulative_sum(shear*self.node_lens, include_initial=True)
        moment = np.interp(self.node_posns, node_inter_posns, moment)
        moment_stress = moment/self.node_sections

        # Principal stress calculation/Mohr's circle at max stress point
        # All values [N]
        max_x_axial_stress = tension_stress + np.copysign(moment_stress, tension_stress)
        center = max_x_axial_stress/2
        radius = np.sqrt(center**2 + shear_stress**2)
        sigma_1 = center + radius
        sigma_2 = center - radius

        von_mises_stress = np.sqrt(0.5*((sigma_1 - sigma_2)**2 + sigma_2**2 + sigma_1**2))

        return tension, shear, moment, tension_stress, shear_stress, np.abs(moment_stress), sigma_1, sigma_2, von_mises_stress