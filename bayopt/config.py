from configparser import ConfigParser
import numpy as np


class Config:
    def __init__(self, path):
        self.config = ConfigParser()
        self.config.readfp(open(path))

        self.scale_beta = self.config.getfloat("Optimization", "scale_beta")
        self.domain_start = np.fromstring(self.config.get("Optimization", "domain_start"), sep=',')
        self.domain_end = np.fromstring(self.config.get("Optimization", "domain_end"), sep=',')
        self.init_lenghtscale = np.fromstring(self.config.get(
            "Optimization", "init_lenghtscale"), sep=',')
        self.init_variance = self.config.getfloat("Optimization", "init_variance")
        self.normalize_data = self.config.getboolean("Optimization", "normalize_data")
        self.swarmopt_n_restarts = self.config.getint("Optimization", "swarmopt_n_restarts")
        self.swarmopt_n_iterations = self.config.getint(
            "Optimization", "swarmopt_n_iterations")
        self.swarmopt_p = self.config.getfloat("Optimization", "swarmopt_p")
        self.swarmopt_g = self.config.getfloat("Optimization", "swarmopt_g")
        self.swarmopt_w = self.config.getfloat("Optimization", "swarmopt_w")
        self.beta = self.config.getfloat("Optimization", "beta")
        self.dim_obs = self.config.getint("Optimization", "dim_obs")
        self.dim_params = self.config.getint("Optimization", "dim_params")
        self.aquisition = self.config.get("Optimization", "aquisition")
        self.set_size = self.config.getint("Optimization", "set_size")
        self.n_opt_samples = self.config.getint("Optimization", "n_opt_samples")
        self.set_init = self.config.get("Optimization", "set_init")

        self.acf_optim = self.config.get("Optimization", "acf_optim")

        self.plotting_n_samples = self.config.getint("Plotting", "plotting_n_samples")

        self.log_trajectory = self.config.getboolean("Logger", "log_trajectory")

        self.seed = self.config.getint("General", "seed")
        self.save_file = self.config.get("General", "save_file")
