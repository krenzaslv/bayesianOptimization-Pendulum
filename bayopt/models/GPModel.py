from botorch.models import ModelListGP
import torch
import gpytorch
from torch.nn import Sequential, ReLU, Linear
from gpytorch.models import IndependentModelList


class ExactMultiTaskGP(ModelListGP):
    def __init__(self, config, dim_loss, train_x=None, train_y=None):
        self.c= config
        self.dim_loss = dim_loss
        train_x = torch.zeros(1, config.dim_params) if train_x is None else train_x
        train_y = torch.zeros(1, dim_loss) if train_y is None else train_y
        self.train_x = train_x
        self.train_y = train_y
        self.config = config

        models = self.setUpModels(train_x, train_y)
        super(ModelListGP, self).__init__(*models)

    def setUpModels(self, train_x, train_y):
        models = []
        for i in range(self.dim_loss):
            model = ExactGPModel(self.config, train_x, train_y[:, i])
            models.append(model)
        return models

    def forward(self, x):
        return super(ModelListGP, self).forward(x)


    def eval(self):
        for i in range(self.dim_loss):
            self.models[i].eval()

    def updateModel(self, train_x, train_y):
        # self.setUpModels(self.dim, train_x, train_y)
        for i in range(self.dim_loss):
            self.train_x = train_x
            self.train_y = train_y
            self.models[i].updateModel(train_x, train_y[:, i])


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, config, train_x=torch.zeros(1, 2), train_y=torch.zeros(1)):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(config.init_variance)
        )
        likelihood.noise = config.init_variance

        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        self.config = config
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=8,has_lengthscale=True))
        self.covar_module.base_kernel.lengthscale = torch.from_numpy(config.init_lenghtscale)
        self.train_x = train_x
        self.train_y = train_y

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def updateModel(self, train_x, train_y):
        super().set_train_data(train_x, train_y, strict=False)
        self.train_x = train_x
        self.train_y = train_y


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
