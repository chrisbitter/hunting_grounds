import gym
from scipy.misc import imresize
import logging
from pylab import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


class HuntingGrounds(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, state_dimensions=(10, 10)):
        self.logger = logging.Logger("frozen-lake", level=logging.DEBUG)

        self.state_dimensions = state_dimensions

        self.world = np.zeros(self.state_dimensions)

        self.hunter = [0, 0]
        self.prey = np.array(self.state_dimensions) - 1

        self.movement_scaling_factors = 1 / np.array(self.state_dimensions)

        self.reset()

    def initialize_world(self, new_world=None):

        if new_world is None:
            # self.world = np.zeros(self.state_dimensions)
            self.world = np.random.random_sample(self.state_dimensions)

        elif new_world.ndim == self.state_dimensions:
            self.world = new_world
        else:
            raise ValueError("Dimensions of new states must match")

    def place_hunter(self, x=None, y=None):

        if x is None:
            x = np.random.randint(0, self.world.shape[0])

        if y is None:
            y = np.random.randint(0, self.world.shape[1])

        self.hunter = [x, y]

    def place_prey(self, x=None, y=None):
        if x is None:
            x = np.random.randint(0, self.world.shape[0])

        if y is None:
            y = np.random.randint(0, self.world.shape[1])
        self.prey = [x, y]

    def get_state_image(self, resolution=(100, 100), channels_first=False):

        fig = Figure(figsize=(5, 5))
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        ax.imshow(self.world, cmap="Blues", alpha=.25)

        ax.autoscale(False)

        ax.scatter(self.hunter[0], self.hunter[1], c="brown", s=500)

        ax.scatter(self.prey[0], self.prey[1], c="grey", s=500)

        fig.subplots_adjust(bottom=0, top=1, left=0, right=1)

        canvas.draw()  # draw the canvas, cache the renderer

        width, height = fig.get_size_inches() * fig.get_dpi()

        width, height = int(width), int(height)

        img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(
            height, width, 3)

        img = imresize(img, resolution + (3,))

        if channels_first:
            img = np.transpose(img, (2, 0, 1))

        return img

    def step(self, action):

        if action == 0:
            pass
        if action == 1:
            self.hunter[0] += 1
        if action == 2:
            self.hunter[0] -= 1
        if action == 3:
            self.hunter[1] += 1
        if action == 4:
            self.hunter[0] -= 1

        self.hunter[0] = np.clip(self.hunter[0], 0, self.world.shape[0] - 1)
        self.hunter[1] = np.clip(self.hunter[1], 0, self.world.shape[1] - 1)

        if self.prey == self.hunter:
            terminal = True
            reward = 1
        # elif self.world[tuple(self.hunter)] > np.random.random():
        #     terminal = True
        #     reward = -1
        else:
            terminal = False
            reward = -.1

        return reward, terminal

    def reset(self):
        self.initialize_world()
        self.place_hunter()

        self.prey = self.hunter

        while self.prey == self.hunter:
            self.place_prey()

        plt.close()

    def render(self, mode='human', save_file=None, alpha=1.):

        fig, ax = plt.subplots()
        ax.imshow(self.world, cmap="Blues", alpha=.25)

        ax.autoscale(False)

        ax.scatter(self.hunter[0], self.hunter[1], c="brown", s=500,
                   alpha=alpha)

        ax.scatter(self.prey[0], self.prey[1], c="grey", s=500)

        if save_file is not None:
            plt.savefig(save_file)

        plt.close()


if __name__ == "__main__":
    env = HuntingGrounds((5, 5))

    state = env.get_state_image((100,100))

    plt.imshow(state)

    plt.show()
