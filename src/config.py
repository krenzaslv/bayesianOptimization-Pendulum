from configparser import ConfigParser

class Config:
    def __init__(self, path):
        self.config = ConfigParser()
        self.config.readfp(open(path))

        self.scale_beta = self.config.getfloat("Optimization", "scale_beta")
        self.lr_aq = self.config.getfloat("Optimization", "lr_aq")
        self.domain_start_p = self.config.getfloat("Optimization", "domain_start_p")
        self.domain_start_d = self.config.getfloat("Optimization", "domain_start_d")
        self.domain_end_p = self.config.getfloat("Optimization", "domain_end_p")
        self.domain_end_d = self.config.getfloat("Optimization", "domain_end_d")
        self.weight_decay_aq = self.config.getfloat("Optimization", "weight_decay_aq")
        self.init_lenghtscale = self.config.getfloat("Optimization", "init_lenghtscale")
        self.init_variance = self.config.getfloat("Optimization", "init_variance")
        self.skip_aready_samples = self.config.getboolean("Optimization","skip_aready_samples")
        self.beta = self.config.getfloat("Optimization", "beta")
        self.dim = self.config.getint("Optimization", "dim")
        self.num_constraints = self.config.getint("Optimization","num_constraints")
        self.n_sample_points = self.config.getint("Optimization", "n_sample_points")
        self.aquisition = self.config.get("Optimization", "aquisition")
        self.ucb_set_n = self.config.getint("Optimization", "ucb_set_n")
        self.n_opt_iterations_aq = self.config.getint(
            "Optimization", "n_opt_iterations_aq"
        )
        self.n_opt_samples = self.config.getint("Optimization", "n_opt_samples")
        self.ucb_use_set = self.config.getboolean("Optimization", "ucb_use_set")

        self.plotting_n_samples = self.config.getint("Plotting", "plotting_n_samples")

        self.seed = self.config.getint("General", "seed")
        self.save_file = self.config.get("General", "save_file")

