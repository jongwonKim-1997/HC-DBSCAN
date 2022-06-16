
import torch

import numpy as np
import gpytorch 





from gpytorch.functions import MaternCovariance
from gpytorch.settings import trace_mode
from gpytorch.kernels import Kernel

class MaternKernel2(Kernel):

    has_lengthscale = True

    def __init__(self, nu=2.5, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super(MaternKernel2, self).__init__(**kwargs)
        self.nu = nu

    def forward(self, x1, x2, diag=False, **params):
        x1 = torch.round(x1)
        x2 = torch.round(x2)
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):


            x1_ = (x1).div(self.lengthscale)
            x2_ = (x2).div(self.lengthscale)
            
            distance = self.covar_dist(x1_, x2_, diag=diag, **params)
            exp_component = torch.exp(-np.sqrt(self.nu * 2) * distance)

            if self.nu == 0.5:
                constant_component = 1
            elif self.nu == 1.5:
                constant_component = (np.sqrt(3) * distance).add(1)
            elif self.nu == 2.5:
                constant_component = (np.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
            return constant_component * exp_component
        return MaternCovariance.apply(
            x1, x2, self.lengthscale, self.nu, lambda x1, x2: self.covar_dist(x1, x2, **params)
        )


from gpytorch.functions import RBFCovariance
from gpytorch.settings import trace_mode
from gpytorch.kernels import Kernel


def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()


class RBFKernel2(Kernel):

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        x1 = torch.round(x1)
        x2 = torch.round(x2)
        if (
            x1.requires_grad
            or x2.requires_grad
            or (self.ard_num_dims is not None and self.ard_num_dims > 1)
            or diag
            or params.get("last_dim_is_batch", False)
            or trace_mode.on()
        ):
            x1_ = x1.div(self.lengthscale)
            x2_ = x2.div(self.lengthscale)
            return self.covar_dist(
              x1_, x2_, square_dist=True, diag=diag, dist_postprocess_func=postprocess_rbf, postprocess=True, **params
            )
        return RBFCovariance.apply(
            x1,
            x2,
            self.lengthscale,
            lambda x1, x2: self.covar_dist(
                x1, x2, square_dist=True, diag=False, dist_postprocess_func=postprocess_rbf, postprocess=False, **params
            ),
        )



class AdvancedGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,n_hyp):
        super(AdvancedGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(active_dims = torch.tensor([0])  ) * MaternKernel2(active_dims = torch.tensor([1]) ) )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)