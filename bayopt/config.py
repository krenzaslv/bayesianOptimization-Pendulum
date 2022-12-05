from configparser import ConfigParser


class Config:
    def __init__(self, path):
        self.config = ConfigParser()
        self.config.readfp(open(path))

        self.scale_beta = self.config.getfloat("Optimization", "scale_beta")
        self.domain_start_p = self.config.getfloat("Optimization", "domain_start_p")
        self.domain_start_d = self.config.getfloat("Optimization", "domain_start_d")
        self.domain_end_p = self.config.getfloat("Optimization", "domain_end_p")
        self.domain_end_d = self.config.getfloat("Optimization", "domain_end_d")
        self.init_lenghtscale = self.config.getfloat("Optimization", "init_lenghtscale")
        self.init_variance = self.config.getfloat("Optimization", "init_variance")
        self.normalize_data = self.config.getboolean("Optimization", "normalize_data")
        self.beta = self.config.getfloat("Optimization", "beta")
        self.dim_obs = self.config.getint("Optimization", "dim_obs")
        self.dim_params = self.config.getint("Optimization", "dim_params")
        self.aquisition = self.config.get("Optimization", "aquisition")
        self.set_size = self.config.getint("Optimization", "set_size")
        self.n_opt_samples = self.config.getint("Optimization", "n_opt_samples")

        self.plotting_n_samples = self.config.getint("Plotting", "plotting_n_samples")

        self.seed = self.config.getint("General", "seed")
        self.save_file = self.config.get("General", "save_file")
