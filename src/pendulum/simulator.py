from gymnasium.envs.classic_control.pendulum import PendulumEnv
import numpy as np
from gymnasium import spaces

from src.pendulum.GymPendulum import GymPendulum


def simulate(config, dynamics, U):
    sim = simulate_sim if config.sim_type == "Sim" else simulate_gym
    return sim(config, dynamics, U)


def simulate_sim(config, dynamics, U):

    x_t = config.x0
    X = np.zeros(shape=(config.n_simulation, 2))
    for i in range(config.n_simulation):
        X[i, :] = x_t
        x_t = dynamics(x_t, U, config)
    return X + np.random.normal(scale=config.eps, size=X.shape)

def simulate_gym(config, dynamics, U):
    X = np.zeros(shape=(config.n_simulation, 2))

    # Mess with gym torques
    envWrapper = GymPendulum(config)

    observation, info = envWrapper.reset()
    x_t = np.array([np.arccos(observation[0]), observation[2]])
    for i in range(config.n_simulation):
        X[i, :] = x_t
        # action = np.array([U(x_t, config)])
        action = np.array([U(x_t, config)])
        print(action)
        observation, reward, terminated, truncated, info = envWrapper.step(action)
        x_t = np.array([np.arccos(observation[0]), observation[2]])
        envWrapper.render()

        if terminated or truncated:
            observation, info = envWrapper.reset()
    envWrapper.close()
    return X + np.random.normal(scale=config.eps, size=X.shape)
