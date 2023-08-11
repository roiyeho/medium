from collections import defaultdict 
import numpy as np
import gymnasium as gym

class QLearningAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95
    ):
        self.env = env

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor 

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Initialize an empty dictionary of state-action values
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

    def get_action(self, obs) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_values[obs])
        
    def update(
        self,
        obs,
        action: int,
        reward: float,
        truncated: bool,
        next_obs        
    ):
        """Update the Q-value of the chosen action"""
        future_q_value = (not truncated) * np.max(self.q_values[next_obs])
        td = reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        self.q_values[obs][action] += self.learning_rate * td

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
