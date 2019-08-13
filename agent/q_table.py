import numpy as np
import os


class Q_table(object):

    def __init__(self, input_dim, output_dim):

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.q_table = np.zeros((np.prod(input_dim), np.prod(output_dim)))

    def calculate_q_state(self, hunter_position, prey_position, max_x, max_y):

        agent_state = sum(hunter_position * np.array([max_x, 1]))
        prey_state = np.sum(prey_position * np.array([max_x, 1]))

        q_state = max_x * max_y * agent_state + prey_state

        return q_state

    def predict(self, q_state):

        action = np.argmax(self.q_table[q_state])

        return action

    def train(self, state, action, reward, next_state, terminal):
        q_state = self.calculate_q_state(state)
        next_q_state = self.calculate_q_state(next_state)

        self.q_table[q_state, action] = reward + (1 - terminal) * 0.9 * np.max(
            self.q_table[next_q_state])

    def load(self, path="q_table.npy"):
        if os.path.isfile(path):
            self.q_table = np.load(path)
        else:
            raise ValueError(f"No model found at {path}")

    def save(self, path="q_table.npy"):
        np.save(path, self.q_table)
