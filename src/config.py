from configparser import ConfigParser
import numpy as np


class Config:
    def __init__(self, path):
        self.config = ConfigParser()
        self.config.readfp(open(path))
        self.m = self.getFloat("m")
        self.l = self.getFloat("l")
        self.k1 = self.getFloat("k1")
        self.k2 = self.getFloat("k2")
        self.kd = self.getFloat("kd")
        self.kp = self.getFloat("kp")
        self.kd_bo = self.getFloat("kd_bo")
        self.kp_bo = self.getFloat("kp_bo")
        self.pi = self.getFloat("pi")
        self.g = self.getFloat("g")
        self.dt = self.getFloat("dt")
        self.lr_gp = self.getFloat("lr_gp")
        self.n_evaluate = self.getInt("n_evaluate")
        self.lr_aq = self.getFloat("lr_aq")
        self.domain_start = self.getFloat("domain_start")
        self.domain_end = self.getFloat("domain_end")
        self.gamma = self.getFloat("gamma")
        self.max_torque = self.getFloat("max_torque")
        self.epsilon = self.getFloat("epsilon")
        self.n_opt_iterations_gp = self.getInt("n_opt_iterations_gp")
        self.n_opt_iterations_aq = self.getInt("n_opt_iterations_aq")
        self.n_opt_samples = self.getInt("n_opt_samples")
        self.seed = self.getInt("seed")
        self.n_simulation = self.getInt("n_simulation")
        self.x0 = np.array([self.getFloat("x0"), self.getFloat("x0_dot")])

    def getFloat(self, key):
        return self.config.getfloat("Settings", key)

    def getInt(self, key):
        return self.config.getint("Settings", key)
