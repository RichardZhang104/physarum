import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import uniform_filter

# force matplotlib to show plot in separate window (for PyCharm)
matplotlib.use('Tkagg')
# get rid of toolbar at bottom of matplotlib window
matplotlib.rcParams['toolbar'] = 'None'


# seed random state
# np.random.seed(123)


class Physarum:
    def __init__(self,
                 agent_number=6000,
                 environment_width=200,
                 sensor_distance=9.0,
                 sensor_angle=np.pi / 4,
                 step_distance=1.0,
                 step_angle=np.pi / 4,
                 diffuse_kernel_size=3,
                 deposit_amount=5.0,
                 decay_factor=0.1):
        # number of simulated organisms
        self.agent_number = agent_number

        # environment is a square grid with side length equal to environment_width. However only pheromones stored with
        # discrete (integer) positions
        self.environment_width = environment_width

        # distance of sensors from agent
        self.sensor_distance = sensor_distance

        # angle of L and R sensors from agent heading
        self.sensor_angle = sensor_angle

        # distance an agent moves in a single step
        self.step_distance = step_distance

        # magnitude of a single heading change, if any
        self.step_angle = step_angle

        # width of kernel used in diffusion step
        self.diffuse_kernel_size = diffuse_kernel_size

        # how much pheromone a single agent deposits
        self.deposit_amount = deposit_amount

        # 1.0  means no decay
        self.decay_factor = decay_factor

        # initialize agent positions, headings and pheromone levels below

        # generate initial positions of agents
        self.positions_x = np.random.uniform(0, self.environment_width, (self.agent_number, 1))
        self.positions_y = np.random.uniform(0, self.environment_width, (self.agent_number, 1))

        # draw headings from uniform distribution on [-pi,pi]
        self.headings = np.random.uniform(-np.pi, np.pi, (self.agent_number, 1))

        # all pheromone values initialized to zero
        self.pheromones = np.zeros((self.environment_width, self.environment_width))

    # function for a single step in the simulation
    def update_positions(self):
        # shape (agent_number, 3)
        sensor_headings = self.headings + np.repeat([[self.sensor_angle, 0, -self.sensor_angle]], self.agent_number, 0)

        # shape (agent_number,3)
        sensors_x = self.positions_x + self.sensor_distance * np.cos(sensor_headings)
        sensors_y = self.positions_y + self.sensor_distance * np.sin(sensor_headings)

        # shape (agent_number, 3)
        sensor_values = self.pheromones[np.mod(sensors_x, self.environment_width).astype(int),
                                        np.mod(sensors_y, self.environment_width).astype(int)]

        # get values for each sensor as a separate list. Shape (agent_number, )
        left_sensor = sensor_values[:, 0]
        middle_sensor = sensor_values[:, 1]
        right_sensor = sensor_values[:, 2]

        # used to calculate direction each agent will turn
        right_middle = 1 * (right_sensor - middle_sensor > 0)
        middle_left = 1 * (middle_sensor - left_sensor >= 0)
        middle_right = 1 * (middle_sensor - right_sensor >= 0)

        # # array of directions in which each agent will turn. -1 is left, 0 is forwards, 1 is right.
        directions = np.select(
            [right_middle + middle_left == 2, right_middle + middle_left == 0, middle_left + middle_right == 2,
             middle_left + middle_right == 0], [1, -1, 0, 2 * np.random.randint(0, 2) - 1])

        # update agent headings. Reshape from (agent_number, ) to (agent_number, 1)
        heading_changes = np.select([directions == -1, directions == 0, directions == 1],
                                    [self.step_angle, 0, -self.step_angle]).reshape(self.agent_number, 1)
        self.headings += heading_changes

        # update agent positions using updated headings
        self.positions_x = np.mod(self.positions_x + self.step_distance * np.cos(self.headings), self.environment_width)
        self.positions_y = np.mod(self.positions_y + self.step_distance * np.sin(self.headings), self.environment_width)

        # all agents deposit pheromones. Multiple agents with identical coordinates stack
        np.add.at(self.pheromones, (self.positions_x.astype(int), self.positions_y.astype(int)), self.deposit_amount)

        # decay pheromone values
        self.pheromones *= self.decay_factor

        # diffuse pheromones
        self.pheromones = uniform_filter(self.pheromones, self.diffuse_kernel_size, mode='constant')


def init():
    ax.set_xlim(0, physarum.environment_width)
    ax.set_ylim(0, physarum.environment_width)
    frame_text.set_text('')
    return agents, frame_text


def update(frame):
    physarum.update_positions()
    agents.set_data(physarum.positions_x, physarum.positions_y)
    frame_text.set_text(str(frame))
    return agents, frame_text


# create instance of an environment
physarum = Physarum()

# create figure and axes
fig, ax = plt.subplots(figsize=(6, 6), dpi=150, facecolor=[0, 0, 0])

# set axes background colour to black
ax.set_facecolor([0, 0, 0])

agents, = ax.plot([], [], marker='.', linestyle=' ', ms=0.2, color='white')
frame_text = ax.text(0.01, 0.97, '', color='white', transform=ax.transAxes)

anim = FuncAnimation(fig=fig, func=update, frames=10000, init_func=init, interval=1, blit=True)
plt.show()
