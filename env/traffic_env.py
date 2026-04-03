import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import LANES, MAX_CARS, GREEN_SIGNAL_TIME


class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()

        self.observation_space = spaces.Box(
            low=0, high=MAX_CARS, shape=(LANES,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(LANES)

        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.randint(0, 10, size=(LANES,))
        return self.state.astype(np.float32), {}

    def step(self, action):
        # 🔥 safety fix
        if self.state is None:
            raise ValueError("Call reset() before step()")

        # cars leave
        self.state[action] = max(
            0, self.state[action] - GREEN_SIGNAL_TIME
        )

        # new cars arrive
        new_cars = np.random.randint(0, 3, size=(LANES,))
        self.state += new_cars

        self.state = np.clip(self.state, 0, MAX_CARS)

        reward = -np.sum(self.state)

        terminated = False
        truncated = False

        return self.state.astype(np.float32), reward, terminated, truncated, {}