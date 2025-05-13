import body
import constants
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.animation as mani
import matplotlib.gridspec as mgrid
import numpy as np

N = 50
width = 50.
length = 1000e3
skyhook = body.Body([constants.r_earth + 600e3, 0., 0., 0., 7.58e3+1000, 4*np.pi/1000.],
                    np.linspace(0, length, num=N),
                    width**2, width**3 / 6, constants.dens_kevlar)

num_steps = 300
duration = 15000 # seconds
stepsize = duration/num_steps
state_history = np.zeros((num_steps, 6))
state_history[0] = skyhook.state
nodevel_history = np.zeros((num_steps, N, 2))
nodevel_history[0] = skyhook.node_vels_world()
time = np.arange(num_steps)*stepsize

for i in range(1, num_steps):
    state_history[i] = skyhook.state
    nodevel_history[i] = skyhook.node_vels_world()
    skyhook.rk4_step(stepsize)

# fig, axs = plt.subplots(1, 3, figsize=(12, 5))
# ax_orbit, ax_omega, _ = axs

fig = plt.figure(layout="constrained", figsize=(12, 5))
gridspec = mgrid.GridSpec(2, 3, figure=fig)
ax_orbit = fig.add_subplot(gridspec[:, 0])
ax_omega = fig.add_subplot(gridspec[0, 1])
ax_vel = fig.add_subplot(gridspec[1, 1], sharex=ax_omega)
ax_energy = fig.add_subplot(gridspec[:, 2])

lim = (-15e6, 15e6)
ax_orbit.set_xlim(lim)
ax_orbit.set_ylim(lim)
ax_orbit.set_aspect('equal')
ax_orbit.set_xlabel('x (m)')
ax_orbit.set_ylabel('x (m)')
# ax.grid()

# Plotting
earth_art = mpatch.Circle((0, 0), constants.r_earth, fc='blue')
ax_orbit.add_artist(earth_art)
ax_orbit.text(0, 0, 'Earth', ha='center', va='center', color='white')
ax_orbit.plot(state_history[:, 0], state_history[:, 1], alpha=0.5)
trace, = ax_orbit.plot([], [], alpha=0.5)
line, = ax_orbit.plot([], [])
body_loc = ax_orbit.scatter([], [], s=0.2)
time_text = ax_orbit.text(0, 10e6, '', ha='center')

ax_omega.plot(time, state_history[:, 5])
ax_omega.set_ylabel('$\\omega$ (rad/s)')
ax_omega.grid()
omega_tmark = ax_omega.axvline(0, linestyle='-', color='k', alpha=0.5)

ax_vel.plot(time, np.linalg.norm(state_history[:, 3:5], axis=1), label='Center of Mass')
ax_vel.plot(time, np.linalg.norm(nodevel_history[:, 0, :], axis=1), label='Endpoint 1')
ax_vel.plot(time, np.linalg.norm(nodevel_history[:, -1, :], axis=1), label='Endpoint 2')
ax_vel.set_xlabel('Elapsed Time (s)')
ax_vel.set_ylabel('Velocity Magnitude (m/s)')
ax_vel.grid()
ax_vel.legend()
vel_tmark = ax_vel.axvline(0, linestyle='-', color='k', alpha=0.5)

base_grav_PE = constants.GM*skyhook.mass/constants.r_earth
CM_Grav_PE = base_grav_PE - constants.GM*skyhook.mass/np.linalg.norm(state_history[:, :2], axis=1)
CM_Trans_KE = 0.5*skyhook.mass*np.linalg.norm(state_history[:, 3:5], axis=1)**2
CM_Rot_KE = 0.5*skyhook.rot_inertia*state_history[:, 5]**2
ax_energy.stackplot(time, CM_Grav_PE, CM_Trans_KE, CM_Rot_KE, labels=('CM Gravitational PE', 'CM Translational KE', 'CM Rotational KE'))
ax_energy.set_xlabel('Elapsed Time (s)')
ax_energy.set_ylabel('Energy (J)')
ax_energy.grid()
ax_energy.legend()
energy_tmark = ax_energy.axvline(0, linestyle='-', color='k', alpha=0.5)

def animate(i):
    trace.set_data(state_history[:i, 0], state_history[:i, 1])
    nodes = skyhook.world_nodes(state_history[i])
    line.set_data(nodes[[0, -1], 0], nodes[[0, -1], 1])
    body_loc.set_offsets((state_history[i, 0], state_history[i, 1]))
    time_text.set_text(f'Elapsed time {time[i]} s')

    for tmark in (omega_tmark, vel_tmark, energy_tmark):
        tmark.set_xdata((time[i], time[i]))
    return trace, line, body_loc, time_text, omega_tmark, vel_tmark, energy_tmark

anim = mani.FuncAnimation(fig, animate, num_steps, interval=50) # interval is ms
# print(skyhook.state)
plt.show()
