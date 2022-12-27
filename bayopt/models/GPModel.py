from botorch.models import ModelListGP
from botorch.models.gp_regression import SingleTaskGP
import torch
import gpytorch
from torch.nn import Sequential, ReLU, Linear


class ExactMultiTaskGP(ModelListGP):
    def __init__(self, config, dim_loss, data, state_dict=None):
        self.c = config
        self.dim_loss = dim_loss
        self.data = data
        self.config = config

        models = self.setUpModels()
        super(ModelListGP, self).__init__(*models)

        if state_dict:
            self.load_state_dict(state_dict)

    def setUpModels(self):
        models = []
        for i in range(self.dim_loss):
            model = ExactGPModel(self.config,
                                 self.data.get_scaled_train_x(), 
                                 self.data.get_scaled_train_y()[:, i].unsqueeze(-1))
            models.append(model)
        return models

    def forward(self, x):
        return super(ModelListGP, self).forward(x)

class ExactGPModel(SingleTaskGP):
    def __init__(self, config, data, state_dict = None):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(config.init_variance)
        )
        likelihood.noise = config.init_variance
        self.config = config
        mean_module = gpytorch.means.ZeroMean()
        covar_module = gpytorch.kernels.MaternKernel(ard_num_dims=config.dim_params)
        covar_module.lengthscale = torch.from_numpy(config.init_lenghtscale)
        super().__init__(data.train_x, data.train_y, likelihood, covar_module, mean_module)

        if state_dict:
            self.load_state_dict(state_dict)


class MLPMean(gpytorch.means.Mean):
    def __init__(self, dim=2):
        super(MLPMean, self).__init__()
        self.mlp = Sequential(
            Linear(dim, 32), ReLU(), Linear(32, 32), ReLU(), Linear(32, 1)
        )
        for layer in self.mlp.children():
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        m = self.mlp(x)
        return m.squeeze(-1)
