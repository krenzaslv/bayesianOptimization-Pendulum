import torch
import gpytorch
from torch.nn import Sequential, ReLU, Linear


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(config.num_constraints)(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
        )
        likelihood.noise = torch.tensor([config.init_variance])
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([config.num_constraints]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([2])),
            batch_shape=torch.Size([config.num_constraints])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, config, train_x=torch.zeros(1, 2), train_y=torch.zeros(1)):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-4)
        )
        likelihood.noise = torch.tensor([config.init_variance])
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        self.config = config
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        self.covar_module.base_kernel.lengthscale = config.init_lenghtscale


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def updateModel(self, train_x, train_y):
        super().set_train_data(train_x, train_y, strict=False)

    


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
