from typing import Optional
import gymnasium
from imageio import config
import numpy as np
from gymnasium import spaces

class GymPendulum(gymnasium.envs.classic_control.PendulumEnv):

    def __init__(self, config, render_mode: Optional[str] = "human", g=10.0):
        self.config = config

        self.max_speed = 100 
        self.max_torque = 100
        self.dt = config.dt
        self.g = config.g
        self.m = config.m 
        self.l = config.L 

        self.render_mode = render_mode 

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the gymnasium api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.state = self.config.x0 
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}
