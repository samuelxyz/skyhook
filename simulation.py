import body
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.animation as mani
import numpy as np

skyhook = body.Body()

stepsize = 100 # seconds
num_steps = 150
state_history = np.zeros((num_steps, 4))
state_history[0] = skyhook.state

for i in range(1, num_steps):
    state_history[i] = skyhook.state
    skyhook.rk4_step(stepsize)

fig, ax = plt.subplots()
lim = (-15e3, 15e3)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.set_xlabel('x (km)')
ax.set_ylabel('x (km)')
# ax.grid()

# Plotting
earth_art = mpatch.Circle((0, 0), 6378, fc='blue')
ax.add_artist(earth_art)
ax.text(0, 0, 'Earth', ha='center', va='center', color='white')
ax.plot(state_history[:, 0], state_history[:, 1], alpha=0.5)
trace, = ax.plot([], [], alpha=0.5)
body_loc = ax.scatter([], [])
time_text = ax.text(-7000, 7000, '')

def animate(i):
    trace.set_data(state_history[:i, 0], state_history[:i, 1])
    body_loc.set_offsets((state_history[i, 0], state_history[i, 1]))
    time_text.set_text(f'Elapsed time {i*stepsize} s')
    return trace, body_loc, time_text

anim = mani.FuncAnimation(fig, animate, num_steps, interval=100) # interval is ms
print(skyhook.acc)
plt.show()
