from gymnasium.envs.classic_control.pendulum import PendulumEnv
import numpy as np
import math
from gymnasium.wrappers import RecordVideo
import time

from bayopt.pendulum.GymPendulum import GymPendulum

def simulate(config, dynamics, U):
    sim = simulate_sim if config.sim_type == "Sim" else simulate_gym
    return sim(config, dynamics, U)

def simulate_sim(config, dynamics, U):

    x_t = config.x0
    X = np.zeros(shape=(config.n_simulation, 2))
    for i in range(config.n_simulation):
        X[i, :] = x_t
        x_t = dynamics(x_t, U, config)
        # x_t[0] = angleDiff(0,x_t[0])
    return X + np.random.normal(scale=config.eps, size=X.shape)

def simulate_gym(config, dynamics, U, render=None):
    X = np.zeros(shape=(config.n_simulation, 2))

    # Mess with gym torques
    envWrapper = GymPendulum(config, render_mode=render)
    observation, info = envWrapper.reset()

    angle = math.atan2(observation[1], observation[0])
    x_t = np.array([angle, observation[2]])
    for i in range(config.n_simulation):
        X[i, :] = x_t
        action = np.array([U(x_t, config)])
        observation, reward, terminated, truncated, info = envWrapper.step(action)
        angle = math.atan2(observation[1], observation[0])
        x_t = np.array([angle, observation[2]])

        if terminated or truncated:
            observation, info = envWrapper.reset()
    envWrapper.close()
    return X + np.random.normal(scale=config.eps, size=X.shape)
