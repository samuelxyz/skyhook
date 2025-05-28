import body
import constants
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.animation as mani
import matplotlib.gridspec as mgrid
import matplotlib.ticker as mtck
import numpy as np

N = 50
# width = 50.*np.ones((N,)) # m
width = 10.*np.exp(-2*np.abs(np.linspace(-1, 1, N))**2) # m
height = 3*width
length = 1000e3 #m
skyhook = body.Body([constants.r_earth + 600e3, 0., 0., 0., 7.58e3+1000, 4*np.pi/1000.],
                    np.linspace(0, length, num=N),
                    width*height, width*height**2/ 6, constants.dens_kevlar)

num_steps = 300
duration = 15000 # seconds
stepsize = duration/num_steps
state_history = np.zeros((num_steps, 6))
nodepos_history = np.zeros((num_steps, N, 2))
nodevel_history = np.zeros((num_steps, N, 2))
beam_history = np.zeros((num_steps, 9, N))
time = np.arange(num_steps)*stepsize

for i in range(0, num_steps):
    state_history[i] = skyhook.state
    nodepos_history[i] = skyhook.world_nodes()
    nodevel_history[i] = skyhook.node_vels_world()
    beam_history[i] = skyhook.beam_analysis()
    skyhook.rk4_step(stepsize)

# Define subplots
fig = plt.figure(layout="constrained", figsize=(12, 8))
gridspec = mgrid.GridSpec(3, 3, figure=fig)
ax_orbit = fig.add_subplot(gridspec[:2, 0])
ax_props = fig.add_subplot(gridspec[2, 0])
ax_omega = fig.add_subplot(gridspec[0, 1])
ax_vel = fig.add_subplot(gridspec[1, 1], sharex=ax_omega)
ax_energy = fig.add_subplot(gridspec[2, 1], sharex=ax_omega)
ax_forcex = fig.add_subplot(gridspec[0, 2])
ax_forcey = fig.add_subplot(gridspec[1, 2], sharex = ax_forcex)
ax_moment = ax_forcey.twinx()
ax_stress = fig.add_subplot(gridspec[2, 2], sharex = ax_forcex)
tick_eng = mtck.EngFormatter()

# Orbit view
lim = (-15e6, 15e6)
ax_orbit.set_xlim(lim)
ax_orbit.set_ylim(lim)
ax_orbit.set_aspect('equal')
ax_orbit.set_xlabel('x (m)')
ax_orbit.set_ylabel('x (m)')
ax_orbit.grid()

earth_art = mpatch.Circle((0, 0), constants.r_earth, fc='blue')
ax_orbit.add_artist(earth_art)
ax_orbit.text(0, 0, 'Earth', ha='center', va='center', color='white')
ax_orbit.plot(state_history[:, 0], state_history[:, 1], alpha=0.5)
trace, = ax_orbit.plot([], [], alpha=0.5)
line, = ax_orbit.plot([], [])
body_loc = ax_orbit.scatter([], [], s=0.2)
time_text = ax_orbit.text(0, 10e6, '', ha='center')

# Properties view
ax_props.plot(skyhook.node_posns, skyhook.node_areas)
ax_props.set_xlabel('x (m)')
ax_props.set_ylabel('Structural cross section (m$^2$)')
ax_props.grid()
ax_props.xaxis.set_major_formatter(tick_eng)

# Omega plot
ax_omega.xaxis.set_major_formatter(tick_eng)
# ax_omega.yaxis.set_major_formatter(tick_eng)
ax_omega.plot(time, state_history[:, 5])
ax_omega.set_ylabel('$\\omega$ (rad/s)')
ax_omega.grid()
omega_tmark = ax_omega.axvline(0, linestyle='-', color='k', alpha=0.5)

# Velocity plot
ax_vel.plot(time, np.linalg.norm(state_history[:, 3:5], axis=1), label='Center of Mass')
ax_vel.plot(time, np.linalg.norm(nodevel_history[:, 0, :], axis=1), label='Endpoint 1')
ax_vel.plot(time, np.linalg.norm(nodevel_history[:, -1, :], axis=1), label='Endpoint 2')
# ax_vel.set_xlabel('Elapsed Time (s)')
ax_vel.set_ylabel('Velocity Magnitude (m/s)')
# ax_vel.xaxis.set_major_formatter(tick_eng)
ax_vel.grid()
ax_vel.legend()
ax_vel.yaxis.set_major_formatter(tick_eng)
vel_tmark = ax_vel.axvline(0, linestyle='-', color='k', alpha=0.5)

# Energy plot
base_grav_PE = constants.GM*skyhook.mass/constants.r_earth
# CM_Grav_PE = base_grav_PE - constants.GM*skyhook.mass/np.linalg.norm(state_history[:, :2], axis=1)
# CM_Trans_KE = 0.5*skyhook.mass*np.linalg.norm(state_history[:, 3:5], axis=1)**2
# CM_Rot_KE = 0.5*skyhook.rot_inertia*state_history[:, 5]**2
# ax_energy.stackplot(time, CM_Grav_PE, CM_Trans_KE, CM_Rot_KE, labels=('CM Gravitational PE', 'CM Translational KE', 'CM Rotational KE'))
Nodal_Grav_PE = base_grav_PE - constants.GM*np.sum(skyhook.node_masses/np.linalg.norm(nodepos_history, axis=2), axis=1)
Nodal_KE = 0.5*np.sum(skyhook.node_masses*np.linalg.norm(nodevel_history, axis=2)**2, axis=1)
ax_energy.stackplot(time, Nodal_Grav_PE, Nodal_KE, labels=('Total Gravitational PE', 'Total KE'))
ax_energy.set_xlabel('Elapsed Time (s)')
ax_energy.set_ylabel('Energy (J)')
# ax_energy.xaxis.set_major_formatter(tick_eng)
ax_energy.grid()
ax_energy.legend()
energy_tmark = ax_energy.axvline(0, linestyle='-', color='k', alpha=0.5)

# Force/moment/stress plots
forces = skyhook.calc_local_forces()
# 0      1      2       3               4             5              6        7        8
tension, shear, moment, tension_stress, shear_stress, moment_stress, sigma_1, sigma_2, von_mises = skyhook.beam_analysis()
max_stats = np.max(beam_history, axis=0) # returns [9, N] of maximum stats across time

# tick_km = mtck.EngFormatter('m')
ax_forcex.plot(skyhook.node_posns, forces[:, 0], label='Force X')
# quiver_x, quiver_y = np.meshgrid(skyhook.node_posns, [0,])
# ax_forcex.quiver(quiver_x, quiver_y, forces[:, 0], forces[:, 1], label='Force Vectors')
ax_forcex.plot(skyhook.node_posns, tension, label='Last Tension')
ax_forcex.plot(skyhook.node_posns, max_stats[0], '--', label='Max Tension')
ax_forcex.set_ylabel('X-Forces (N)')
ax_forcex.grid()
ax_forcex.legend()
ax_forcex.xaxis.set_major_formatter(tick_eng)
forcey_legend = [] # Collection of line handles to put in the legend
forcey_legend.extend(ax_forcey.plot(skyhook.node_posns, forces[:, 1], label='Force Y'))
forcey_legend.extend(ax_forcey.plot(skyhook.node_posns, shear, label='Last Shear'))
forcey_legend.extend(ax_forcey.plot(skyhook.node_posns, max_stats[1], '--', color='C1', label='Max Shear'))
ax_forcey.set_ylabel('Y-Forces (N)')
ax_forcey.grid()
# ax_forcey.yaxis.set_major_formatter(tick_eng)
forcey_legend.extend(ax_moment.plot(skyhook.node_posns, moment, color='C2', label='Last Moment'))
forcey_legend.extend(ax_moment.plot(skyhook.node_posns, max_stats[2], '--', color='C2', label='Max Moment'))
ax_moment.tick_params(axis='y', labelcolor='C2')
ax_moment.set_ylabel('Bending Moment (N m)', color='C2')
# ax_moment.yaxis.set_major_formatter(tick_eng)
ax_forcey.legend(handles=forcey_legend)
ax_stress.plot(skyhook.node_posns, shear_stress, label='Last Shear Stress')
ax_stress.plot(skyhook.node_posns, max_stats[4], '--', label='Max Shear Stress')
ax_stress.stackplot(skyhook.node_posns, tension_stress, moment_stress, labels=('Last Axial Stress (Tension)', 'Last Axial Stress (Bending)'), alpha=0.5)
ax_stress.stackplot(skyhook.node_posns, max_stats[3], max_stats[5], labels=('Max Axial Stress (Tension)', 'Max Axial Stress (Bending)'), colors=('C2','C3'), alpha=0.2)
# ax_stress.plot(skyhook.node_posns, sigma_1, '-', label='$\\sigma_1$')
# ax_stress.plot(skyhook.node_posns, sigma_2, ':', label='$\\sigma_2$')
ax_stress.plot(skyhook.node_posns, von_mises, label='Last von Mises')
ax_stress.plot(skyhook.node_posns, max_stats[8], '--', label='Max von Mises')
ax_stress.grid()
ax_stress.legend()
ax_stress.set_xlabel('x (m)')
ax_stress.set_ylabel('Stress (Pa)')
# ax_stress.yaxis.set_major_formatter(tick_eng)

def animate(i):
    trace.set_data(state_history[:i, 0], state_history[:i, 1])
    nodes = skyhook.world_nodes(state_history[i])
    line.set_data(nodes[[0, -1], 0], nodes[[0, -1], 1])
    body_loc.set_offsets((state_history[i, 0], state_history[i, 1]))
    time_text.set_text(f'Elapsed time {time[i]} s')

    for tmark in (omega_tmark, vel_tmark, energy_tmark):
        tmark.set_xdata((time[i], time[i]))
    return trace, line, body_loc, time_text, omega_tmark, vel_tmark, energy_tmark

anim = mani.FuncAnimation(fig, animate, num_steps, interval=50, blit=True) # interval is ms
# print(skyhook.state)
plt.show()
