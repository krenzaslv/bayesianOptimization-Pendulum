from configparser import ConfigParser
import numpy as np


class Config:
    def __init__(self, path):
        self.config = ConfigParser()
        self.config.readfp(open(path))
        self.m = self.config.getfloat("Dynamics", "m")
        self.l = self.config.getfloat("Dynamics", "l")
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

        self.scale_beta = self.config.getfloat("Optimization", "scale_beta")
        self.n_evaluate = self.config.getint("Optimization", "n_evaluate")
        self.lr_aq = self.config.getfloat("Optimization", "lr_aq")
        self.lr_gp = self.config.getfloat("Optimization", "lr_gp")
        self.domain_start_p = self.config.getfloat("Optimization", "domain_start_p")
        self.domain_start_d = self.config.getfloat("Optimization", "domain_start_d")
        self.domain_end_p = self.config.getfloat("Optimization", "domain_end_p")
        self.domain_end_d = self.config.getfloat("Optimization", "domain_end_d")
        self.weight_decay_gp = self.config.getfloat("Optimization", "weight_decay_gp")
        self.weight_decay_aq = self.config.getfloat("Optimization", "weight_decay_aq")
        self.init_lenghtscale = self.config.getfloat("Optimization", "init_lenghtscale")
        self.init_variance = self.config.getfloat("Optimization", "init_variance")
        self.gamma = self.config.getfloat("Optimization", "gamma")
        self.beta = self.config.getfloat("Optimization", "beta")
        self.n_opt_iterations_gp = self.config.getint(
            "Optimization", "n_opt_iterations_gp"
        )
        self.n_sample_points = self.config.getint("Optimization", "n_sample_points")
        self.ucb_set_n = self.config.getint("Optimization", "ucb_set_n")
        self.n_opt_iterations_aq = self.config.getint(
            "Optimization", "n_opt_iterations_aq"
        )
        self.n_opt_samples = self.config.getint("Optimization", "n_opt_samples")
        self.aquisition = self.config.get("Optimization", "aquisition")
        self.ucb_use_set = self.config.getboolean("Optimization", "ucb_use_set")
        self.beta_fixed = self.config.getboolean("Optimization", "beta_fixed")

        self.plotting_n_samples = self.config.getint("Plotting", "plotting_n_samples")

        self.seed = self.config.getint("General", "seed")
        self.save_file = self.config.get("General", "save_file")
