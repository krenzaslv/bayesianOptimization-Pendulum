from configparser import ConfigParser
import numpy as np


class Config:
    def __init__(self, path):
        self.config = ConfigParser()
        self.config.readfp(open(path))

        self.m = self.config.getfloat("Dynamics", "m")
        self.L = self.config.getfloat("Dynamics", "l")
        self.pi = self.config.getfloat("Dynamics", "pi")
        self.g = self.config.getfloat("Dynamics", "g")
        self.dt = self.config.getfloat("Dynamics", "dt")
        self.n_simulation = self.config.getint("Dynamics", "n_simulation")
        self.max_torque = self.config.getfloat("Dynamics", "max_torque")
        self.x0 = np.array(
            [
                self.config.getfloat("Dynamics", "x0"),
                self.config.getfloat("Dynamics", "x0_dot"),
            ]
        )

        self.k1 = self.config.getfloat("Controller", "k1")
        self.k2 = self.config.getfloat("Controller", "k2")
        self.kp = self.config.getfloat("Controller", "kp")
        self.kd = self.config.getfloat("Controller", "kd")
        self.kp_bo = self.config.getfloat("Controller", "kp_bo")
        self.kd_bo = self.config.getfloat("Controller", "kd_bo")
