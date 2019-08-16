from environment.hunting_grounds import HuntingGrounds as Environment
from agent.cnn_agent import CnnAgent as Agent
import numpy as np
import os
import imageio
from tqdm import tqdm
import time
import pandas as pd
import math

train = True
# save_test_images = True
render_interactive = False

world_dimensions = (10, 10)

env = Environment(world_dimensions)

# state_resolution = (100, 100)

epochs = 100_000_000
MAX_STEPS = 20
agent = Agent(world_dimensions, 5)

results_id = time.strftime('%Y%m%d-%H%M%S')

results_folder = "results/" + results_id

if not os.path.exists(results_folder):
    os.makedirs(results_folder)


def run_until_terminal(agent, exploration_probability=0.):
    experiences = []

    state = env.get_state_raw()

    terminal = False

    step = 0

    while not terminal:

        if exploration_probability > np.random.random():
            action = np.random.randint(5)
        else:
            action = agent.predict(state)

        reward, terminal = env.step(action)

        if step >= 20 and not terminal:
            terminal = True
            reward = -1

        next_state = env.get_state_raw()

        experience = [state, action, reward, next_state, terminal]

        experiences.append(experience)

        state = next_state

    return experiences


def train(agent, epochs=1):
    statistics = {"steps": [], "reward": []}

    for epoch in tqdm(range(epochs)):

        env.reset()

        if epoch % 50 == 0:
            experiences = run_until_terminal(agent, 1 - (epoch / epochs))

            reward = sum([e[2] for e in experiences])
            steps = len(experiences)

            statistics["reward"].append(reward)
            statistics["steps"].append(steps)

        experiences = run_until_terminal(agent, 1 - (epoch / epochs))

        for experience in experiences:
            agent.add_experience(*experience)

            agent.train()

    df_stats = pd.DataFrame(statistics)
    df_stats.to_csv(
        os.path.join(results_folder, f"{results_id}-statistics.csv"))



def test(agent, test_runs=1, results_path=None):
    padding_test_runs = int(math.ceil(math.log10(test_runs)))
    padding_steps = int(math.ceil(math.log10(MAX_STEPS)))

    for test_run in tqdm(range(test_runs)):

        terminal = False

        env.reset()

        step = 0

        figure_name = os.path.join(results_path,
                                   f"{test_run:0{padding_test_runs}}-{0:0{padding_steps}}.png")

        env.render(interactive=render_interactive, save_file=figure_name)

        while not terminal:

            state = env.get_state_raw()

            action = agent.predict(state)

            reward, terminal = env.step(action)

            if step >= MAX_STEPS and not terminal:
                terminal = True
                reward = -1

            figure_name = os.path.join(results_path,
                                       f"{test_run:0{padding_test_runs}}-{(step + 1):0{padding_steps}}.png")

            env.render(interactive=render_interactive, save_file=figure_name)

            step += 1

    if os.path.exists(results_path) and len(os.listdir(results_path)):

        frames = []

        # with imageio.get_writer(f'{results_id}.gif', mode='I') as writer:
        for f in sorted([f for f in os.listdir(results_path) if f.endswith(".png")]):

            image = imageio.imread(os.path.join(results_path, f))
            frames.append(image)
            # writer.append_data(image)

        imageio.mimsave(os.path.join(results_path, f'{results_id}.gif'), frames, 'GIF', duration=0.5)


if train:
    train(agent, epochs)

    agent.save("cnn_agent.pt")
else:
    agent.load("cnn_agent.pt")

test(agent, 10, results_folder)
