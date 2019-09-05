import logging
import numpy as np
from .utils.visualizer import Visualizer


class HuntingGrounds(object):
    metadata = {'render.modes': ['human']}

    def __init__(self, world_dimensions=(10, 10), headless=True):
        self.logger = logging.Logger("frozen-lake", level=logging.DEBUG)

        self.world_dimensions = world_dimensions

        self.world = np.zeros(self.world_dimensions)

        self.hunter = None
        self.prey = None

        self.movement_scaling_factors = 1 / np.array(self.world_dimensions)

        self.reset()

        self.visualizer = Visualizer(headless=headless)

    def initialize_world(self, new_world=None):

        if new_world is None:
            # self.world = np.zeros(self.state_dimensions)
            self.world = np.random.random_sample(self.world_dimensions)

        elif new_world.ndim == self.world_dimensions:
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

    def get_state_raw(self):
        state_world = self.world

        state_hunters = np.zeros(self.world_dimensions)
        state_hunters[self.hunter[0], self.hunter[1]] = 1

        state_prey = np.zeros(self.world_dimensions)
        state_prey[self.prey[0], self.prey[1]] = 1

        state = np.stack((state_world, state_hunters, state_prey))

        return state

    # def get_state_image(self, resolution=(100, 100), channels_first=False):
    #
    #     fig = Figure(figsize=(5, 5))
    #     canvas = FigureCanvas(fig)
    #     ax = fig.gca()
    #
    #     ax.imshow(self.world, cmap="Blues", alpha=.25)
    #
    #     ax.autoscale(False)
    #
    #     ax.scatter(self.hunter[0], self.hunter[1], c="brown", s=500)
    #
    #     ax.scatter(self.prey[0], self.prey[1], c="grey", s=500)
    #
    #     fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
    #
    #     canvas.draw()  # draw the canvas, cache the renderer
    #
    #     width, height = fig.get_size_inches() * fig.get_dpi()
    #
    #     width, height = int(width), int(height)
    #
    #     img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(
    #         height, width, 3)
    #
    #     img = imresize(img, resolution + (3,))
    #
    #     if channels_first:
    #         img = np.transpose(img, (2, 0, 1))
    #
    #     return img

    def get_state(self, mode='raw'):
        if mode == 'raw':
            return self.get_state_raw()
        else:
            raise NotImplementedError(f"mode={mode} not implemented.")

    def get_state_dimensions(self, mode='raw'):
        if mode == 'raw':
            return self.world_dimensions + (3,)
        else:
            raise NotImplementedError(f"mode={mode} not implemented.")

    def step(self, action):

        if action == 0:
            pass
        elif action == 1:
            self.hunter[0] += 1
        elif action == 2:
            self.hunter[0] -= 1
        elif action == 3:
            self.hunter[1] += 1
        elif action == 4:
            self.hunter[1] -= 1

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

    def render(self, save_file=None):
        self.visualizer.render(self.get_state(), save_file=save_file)


if __name__ == "__main__":
    import tempfile

    env = HuntingGrounds((5, 5), True)

    state = env.get_state()
    print(state.shape)

    with tempfile.NamedTemporaryFile(suffix=".png") as file:
        env.render(file)

        file.seek(0)

        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        img = mpimg.imread(file)
        plt.imshow(img)

        plt.axis('off')

        plt.show()
