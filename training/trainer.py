import numpy as np
from tqdm import tqdm
import os


class Trainer:

    def __init__(self):
        ...

    @classmethod
    def run_until_terminal(cls, environment, agent, exploration_probability=0.,
                           max_steps=100):

        experiences = []

        state = environment.get_state()

        terminal = False

        step = 1

        while not terminal:

            if exploration_probability > np.random.random():
                action = np.random.randint(5)
            else:
                action = agent.predict(state)

            reward, terminal = environment.step(action)

            if step >= max_steps and not terminal:
                terminal = True
                reward = -1

            next_state = environment.get_state()

            experience = {"s": state, "a": action, "r": reward,
                          "s_": next_state, "t": terminal}

            experiences.append(experience)

            state = next_state

            step += 1

        return experiences


if __name__ == "__main__":

    import pandas as pd

    from environment.hunting_grounds import HuntingGrounds as Environment
    from agent.cnn_agent import CnnAgent as Agent

    from time import strftime, time

    environment = Environment(headless=False)
    agent = Agent(environment.get_state_dimensions(), 5, memory_size=1_000)

    import matplotlib.pyplot as plt
    import csv

    experiment_id = strftime("%y%m%d-%H%M%S")

    results_folder = f"results/{experiment_id}"
    models_folder = os.path.join(results_folder, "models")

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    epochs = 1000_000
    test_epochs = max(epochs // 50, 1)
    max_reward = -np.inf

    statistics_column_names = ["epoch", "steps_mean", "steps_p25", "steps_p50",
                               "steps_p75", "reward_mean", "reward_p25",
                               "reward_p50", "reward_p75"]

    path_statistics_file = os.path.join(results_folder,
                                        f"{experiment_id}-statistics.csv")

    with open(path_statistics_file, 'w', newline='') as statistics_file:
        statistics_writer = csv.writer(statistics_file)

        statistics_writer.writerow(statistics_column_names)

        for epoch in tqdm(range(epochs)):

            environment.reset()

            if epoch % test_epochs == 0:

                test_rewards = []

                steps = []
                rewards = []

                for idx in range(10):
                    experiences = Trainer.run_until_terminal(environment, agent,
                                                             max_steps=20)

                    steps.append(len(experiences))
                    rewards.append(np.mean([e["r"] for e in experiences]))

                test_statistics = {
                    "epoch": epoch, "steps_mean": np.mean(steps),
                    "steps_p25": np.percentile(steps, 25),
                    "steps_p50": np.percentile(steps, 50),
                    "steps_p75": np.percentile(steps, 75),
                    "reward_mean": np.mean(rewards),
                    "reward_p25": np.percentile(rewards, 25),
                    "reward_p50": np.percentile(rewards, 50),
                    "reward_p75": np.percentile(rewards, 75)
                }

                statistics_row = [test_statistics[name] for name in
                                  statistics_column_names]

                statistics_writer.writerow(statistics_row)

                if test_statistics["reward_mean"] > max_reward:
                    max_reward = test_statistics["reward_mean"]
                    agent.save(os.path.join(models_folder,
                                            f"{experiment_id}-{epoch}.agent"))

            experiences = Trainer.run_until_terminal(environment, agent,
                                                     1 - (epoch / epochs),
                                                     max_steps=20)

            agent.add_experiences(experiences)

            agent.train()

        agent.save(
            os.path.join(models_folder, f"{experiment_id}-{epoch}.agent"))

    statistics = pd.read_csv(path_statistics_file)

    fig, axes = plt.subplots(2, 1)

    # print(statistics.head())

    ax = plt.subplot(211)
    plt.fill_between(statistics.index, statistics["steps_p25"],
                     statistics["steps_p75"], alpha=.2)
    statistics[["steps_mean", "steps_p50"]].plot(ax=ax)

    plt.xlabel("Epoch")
    plt.ylabel("Steps")

    ax = plt.subplot(212)
    plt.fill_between(statistics.index, statistics["reward_p25"],
                     statistics["reward_p75"], alpha=.2)
    statistics[["reward_mean", "reward_p50"]].plot(ax=ax)

    plt.xlabel("Epoch")
    plt.ylabel("Reward")

    plt.savefig(os.path.join(results_folder, f"{experiment_id}.png"))
