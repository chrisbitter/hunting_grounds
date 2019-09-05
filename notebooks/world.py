from hunting_grounds.agent.cnn_agent import CnnAgent
from hunting_grounds.environment.hunting_grounds import HuntingGrounds
import matplotlib.pyplot as plt
import time
import numpy as np

world_dimensions = (10, 10)

hunting_grounds = HuntingGrounds(world_dimensions=world_dimensions)

hunting_grounds.reset()

for i in range(5):


    state = hunting_grounds.get_state_raw()

    action = np.random.randint(5)

    hunting_grounds.step(action)

    hunting_grounds.render()


