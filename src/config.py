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
        self.scale_beta = self.getFloat("scale_beta")
        self.n_evaluate = self.getInt("n_evaluate")
        self.lr_aq = self.getFloat("lr_aq")
        self.domain_start_p = self.getFloat("domain_start_p")
        self.domain_end_p = self.getFloat("domain_end_p")
        self.plotting_n_samples = self.getInt("plotting_n_samples")
        self.weight_decay_gp = self.getFloat("weight_decay_gp")
        self.weight_decay_aq = self.getFloat("weight_decay_aq")
        self.domain_start_d = self.getFloat("domain_start_d")
        self.domain_end_d = self.getFloat("domain_end_d")
        self.init_lenghtscale = self.getFloat("init_lenghtscale")
        self.init_variance = self.getFloat("init_variance")
        self.gamma = self.getFloat("gamma")
        self.max_torque = self.getFloat("max_torque")
        self.beta = self.getFloat("beta")
        self.epsilon = self.getFloat("epsilon")
        self.n_opt_iterations_gp = self.getInt("n_opt_iterations_gp")
        self.n_sample_points = self.getInt("n_sample_points")
        self.ucb_set_n = self.getInt("ucb_set_n")
        self.n_opt_iterations_aq = self.getInt("n_opt_iterations_aq")
        self.n_opt_samples = self.getInt("n_opt_samples")
        self.seed = self.getInt("seed")
        self.log_optimizers = self.getBool("log_optimizers")
        self.save_file = self.getString("save_file")
        self.aquisition = self.getString("aquisition")
        self.n_simulation = self.getInt("n_simulation")
        self.plotting = self.getBool("plotting")
        self.ucb_use_set = self.getBool("ucb_use_set")
        self.beta_fixed = self.getBool("beta_fixed")
        self.x0 = np.array([self.getFloat("x0"), self.getFloat("x0_dot")])

    def getFloat(self, key):
        return self.config.getfloat("Settings", key)

    def getBool(self, key):
        return self.config.getboolean("Settings", key)

    def getInt(self, key):
        return self.config.getint("Settings", key)

    def getString(self, key):
        return self.config.get("Settings", key)
