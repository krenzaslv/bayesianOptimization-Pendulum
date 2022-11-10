import torch
import gpytorch
from torch.nn import Sequential, ReLU, Linear


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, config, dim, train_x=None, train_y=None):
        self.dim = dim
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
                num_tasks=dim,
                rank=0,
                has_task_noise=False,
                has_global_noise=True)
        likelihood.noise = torch.tensor([config.init_variance])
        train_x = torch.zeros(1, 2) if train_x == None else train_x
        train_y = torch.zeros(dim) if train_y == None else self.reshapeYToBatchSize(train_y)

        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([dim]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(batch_shape=torch.Size([dim])),
            batch_shape=torch.Size([dim])
        )

    def reshapeYToBatchSize(self, y_train):
        return y_train.transpose(1,1).flatten()


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )

    def updateModel(self, train_x, train_y):
        super().set_train_data(train_x, self.reshapeYToBatchSize(train_y), strict=False)


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
        super().set_train_data(train_x, train_y[:, 0], strict=False)


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
