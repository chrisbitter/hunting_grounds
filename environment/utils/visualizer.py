import importlib
import matplotlib
import numpy as np

class Visualizer(object):

    def __init__(self, headless=False):

        self.headless = headless

        if not headless:
            matplotlib.use('TkAgg')

        self.plt = importlib.import_module("matplotlib.pyplot")

        self.fig, self.ax = None, None

    def render(self, state, save_file=None):

        if self.headless:

            fig, ax = self.plt.subplots(figsize=(10, 10))

        else:
            if self.ax is None:
                self.fig, self.ax = self.plt.subplots(figsize=(10, 10))
                self.plt.show(block=False)

            fig, ax = self.fig, self.ax

            self.plt.cla()

        world, hunters, prey = state

        ax.imshow(world, cmap="Blues", alpha=.25, origin="lower")

        # ax.set_xlabel("x")
        # ax.set_ylabel("y")

        ax.autoscale(False)

        s = ((ax.get_window_extent().width / (
                1.2 * np.max(world.shape)) * 72. / fig.dpi) ** 2)

        hunters_x, hunters_y = np.argwhere(hunters == 1).T
        ax.scatter(hunters_x, hunters_y, c="brown", s=s)

        prey_x, prey_y = np.argwhere(prey == 1).T
        ax.scatter(prey_x, prey_y, c="grey", s=s)

        self.plt.tight_layout()

        if save_file is not None:
            self.plt.savefig(save_file)

        if self.headless:
            self.plt.close()
        else:
            self.plt.draw()
            self.plt.pause(0.1)


if __name__ == "__main__":

    visualizer = Visualizer(headless=False)

    world_dimensions = (10, 10)

    for _ in range(10):
        world = np.random.random_sample(world_dimensions)
        hunters = np.random.choice([0, 1], world_dimensions, p=[0.9, 0.1])
        prey = np.random.choice([0, 1], world_dimensions, p=[0.9, 0.1])

        state = np.stack((world, hunters, prey))

        visualizer.render(state)
