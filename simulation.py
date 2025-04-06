import body
import constants
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.animation as mani
import numpy as np

N = 50
width = 50.
length = 1000e3
skyhook = body.Body([constants.r_earth + 600e3, 0., 0., 0., 7.58e3, 2*np.pi/1000.],
                    np.linspace(0, length, num=N),
                    width**2, width**3 / 6, constants.dens_kevlar)

num_steps = 300
duration = 15000 # seconds
stepsize = duration/num_steps
state_history = np.zeros((num_steps, 6))
state_history[0] = skyhook.state

for i in range(1, num_steps):
    state_history[i] = skyhook.state
    skyhook.rk4_step(stepsize)

fig, ax = plt.subplots()
lim = (-15e6, 15e6)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.set_xlabel('x (m)')
ax.set_ylabel('x (m)')
# ax.grid()

# Plotting
earth_art = mpatch.Circle((0, 0), constants.r_earth, fc='blue')
ax.add_artist(earth_art)
ax.text(0, 0, 'Earth', ha='center', va='center', color='white')
ax.plot(state_history[:, 0], state_history[:, 1], alpha=0.5)
trace, = ax.plot([], [], alpha=0.5)
line, = ax.plot([], [])
body_loc = ax.scatter([], [], s=0.2)
time_text = ax.text(0, 10e6, '', ha='center')

def animate(i):
    trace.set_data(state_history[:i, 0], state_history[:i, 1])
    nodes = skyhook.world_nodes(state_history[i])
    line.set_data(nodes[[0, -1], 0], nodes[[0, -1], 1])
    body_loc.set_offsets((state_history[i, 0], state_history[i, 1]))
    time_text.set_text(f'Elapsed time {i*stepsize} s')
    return trace, line, body_loc, time_text

anim = mani.FuncAnimation(fig, animate, num_steps, interval=50) # interval is ms
print(skyhook.state)
plt.show()
