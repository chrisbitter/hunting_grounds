from agents.cnn_agent import CnnAgent as Agent
from env.hunting_grounds import HuntingGrounds as Environment
import numpy as np
import os
import imageio
from tqdm import tqdm

env = Environment((5, 5))

state_resolution = (100, 100)

agent = Agent(state_resolution, 5)

if False:

    np.random.seed(73)

    epochs = 1000

    experience = []

    for epoch in tqdm(range(epochs)):

        env.reset()

        terminal = False

        step = 0

        state = env.get_state_image(channels_first=True)

        while not terminal:

            if epoch / epochs > np.random.random():
                action = np.random.randint(5)
            else:
                action = agent.predict(state)

            reward, terminal = env.step(action)

            if step >= 20 and not terminal:
                terminal = True
                reward = -1

            next_state = env.get_state_image(channels_first=True)

            agent.add_experience(state, action, reward, next_state, terminal)

            agent.train()

            state = next_state

            step += 1

    agent.save("cnn_agent.pt")

else:
    agent.load("cnn_agent.pt")

import time

for test_id in tqdm(range(5)):

    if not os.path.exists("plots/{}".format(test_id)):
        os.makedirs("plots/{}".format(test_id))

    terminal = False

    env.reset()

    step = 0

    figure_name = "plots/{}/{}.png".format(test_id, time.time())

    env.render(alpha=1, save_file=figure_name)

    print("\n\n")

    while not terminal:

        state = env.get_state_image(channels_first=True)

        action = agent.predict(state)

        reward, terminal = env.step(action)

        if step >= 20 and not terminal:
            terminal = True
            reward = -1

        figure_name = "plots/{}/{}.png".format(test_id, time.time())

        env.render(alpha=(1 - step / 20), save_file=figure_name)

        step += 1

with imageio.get_writer('movie.gif', mode='I') as writer:
    for experiment in os.listdir("plots"):
        for filename in os.listdir("plots/{}".format(experiment)):
            image = imageio.imread("plots/{}/{}".format(experiment, filename))
            writer.append_data(image)
