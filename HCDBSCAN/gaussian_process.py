import gpytorch 
import torch
import kernel

class AdvancedGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,n_hyp):
        super(AdvancedGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims = torch.tensor([0])  ) * kernel.MaternKernel2(active_dims = torch.tensor([1]) ) )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)