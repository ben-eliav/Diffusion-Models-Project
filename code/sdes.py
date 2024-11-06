import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import abc

# copied from paper -- just to understand why mine doesn't work
class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, N):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.N = N

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  @abc.abstractmethod
  def prior_sampling(self, shape):
    """Generate one sample from the prior distribution, $p_T(x)$."""
    pass

  @abc.abstractmethod
  def prior_logp(self, z):
    """Compute log-density of the prior distribution.

    Useful for computing the log-likelihood via probability flow ODE.

    Args:
      z: latent code
    Returns:
      log probability density
    """
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G

  def reverse(self, score_fn, probability_flow=False):
    """Create the reverse-time SDE/ODE.

    Args:
      score_fn: A time-dependent score-based model that takes x and t and returns the score.
      probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
    """
    N = self.N
    T = self.T
    sde_fn = self.sde
    discretize_fn = self.discretize

    # Build the class for reverse-time SDE.
    class RSDE(self.__class__):
      def __init__(self):
        self.N = N
        self.probability_flow = probability_flow

      @property
      def T(self):
        return T

      def sde(self, x, t):
        """Create the drift and diffusion functions for the reverse SDE/ODE."""
        drift, diffusion = sde_fn(x, t)
        score = score_fn(x, t)
        drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
        # Set the diffusion function to zero for ODEs.
        diffusion = 0. if self.probability_flow else diffusion
        return drift, diffusion

      def discretize(self, x, t):
        """Create discretized iteration rules for the reverse diffusion sampler."""
        f, G = discretize_fn(x, t)
        rev_f = f - G[:, None, None, None] ** 2 * score_fn(x, t) * (0.5 if self.probability_flow else 1.)
        rev_G = torch.zeros_like(G) if self.probability_flow else G
        return rev_f, rev_G

    return RSDE()


class VPSDE_copy(SDE):
  def __init__(self, beta_min=0.1, beta_max=20, N=1000):
    """Construct a Variance Preserving SDE.

    Args:
      beta_min: value of beta(0)
      beta_max: value of beta(1)
      N: number of discretization steps
    """
    super().__init__(N)
    self.beta_0 = beta_min
    self.beta_1 = beta_max
    self.N = N
    self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
    self.alphas = 1. - self.discrete_betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

  @property
  def T(self):
    return 1

  def sde(self, x, t):
    beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
    drift = -0.5 * beta_t[:, None, None, None] * x
    diffusion = torch.sqrt(beta_t)
    return drift, diffusion

  def marginal_prob(self, x, t):
    log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
    std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
    return mean, std

  def prior_sampling(self, shape):
    return torch.randn(*shape)

  def prior_logp(self, z):
    shape = z.shape
    N = np.prod(shape[1:])
    logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
    return logps

  def discretize(self, x, t):
    """DDPM discretization."""
    timestep = (t * (self.N - 1) / self.T).long()
    beta = self.discrete_betas.to(x.device)[timestep]
    alpha = self.alphas.to(x.device)[timestep]
    sqrt_beta = torch.sqrt(beta)
    f = torch.sqrt(alpha)[:, None, None, None] * x - x
    G = sqrt_beta
    return f, G
  

## Variance Preserving SDE (Diffusion as in DDPM paper)
class VPSDE:
    """
    Variance Preserving Stochastic Differential Equation.
    From the paper, we know that this is x(t) = -1/2 beta(t) x(t) dt + sqrt(beta(t)) dW(t)
    """
    def __init__(self, beta_0, beta_T, T):
        self.beta_0 = beta_0
        self.beta_T = beta_T
        self.T = T
        self.betas = torch.linspace(beta_0, beta_T, T)  # discretized
        self.alpha = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.coeff_prev_diffusion = 1 / self.alpha.sqrt()
        self.coeff_noise_diffusion = self.coeff_prev_diffusion * (1 - self.alpha) / (1 - self.alpha_bar).sqrt()

    def beta(self, t):  # continuous beta
        return self.beta_0 + (self.beta_T - self.beta_0) * t / self.T
    
    def sde(self, x, t):
        """
        x: torch.Tensor(B, C, H, W)
        t: torch.Tensor(B)
        """
        return -0.5 * self.beta(t)[:, None, None, None] * x, torch.sqrt(self.beta(t))[:, None, None, None]
    
    def dist(self, x, t):
        """
        x: torch.Tensor(B, C, H, W) - the original x
        t: torch.Tensor(B) \in [0, T] - the time
        Return the mean and variance of x(t) given x(0).
        """
        exponent_integral = (-0.5 * self.beta_0 * t - 0.25 * (self.beta_T - self.beta_0) * t**2)[:, None, None, None]
        mean = x * torch.exp(exponent_integral)
        std = torch.sqrt((1 - torch.exp(2 * exponent_integral)))
        return mean, std
    
    def reverse_sde(self, model, x, t):
        """
        x: torch.Tensor(B, C, H, W) 
        t: torch.Tensor(B) \in [0, 1]
        model: Score model
        Return the drift and diffusion of the reverse SDE, as explained in the paper.
        """
        drift, diffusion = self.sde(x, t)
        return drift - diffusion ** 2 * model(x, t), diffusion
    
    def reverse_pf_ode(self, model, x, t):
        """
        Return the drift of the reverse ODE using Probability Flow sampling.
        """
        drift, diffusion = self.sde(x, t)
        return drift - 0.5 * diffusion ** 2 * model(x, t)
    

## Score model trainer
class SDELoss(nn.Module):
    def __init__(self, sde):
        super().__init__()
        self.sde = sde

    def forward(self, model, x):
        """
        model: Score model
        x: torch.Tensor(B, C, H, W) - train data.
        """
        t = torch.rand(x.shape[0], device=x.device)
        noise = torch.randn_like(x)
        mean, std = self.sde.dist(x, t)
        x_t = mean + std * noise
        score_prediction = model(x_t, t)
        return F.mse_loss(score_prediction * std, -noise)
    

def get_loss_fn(sde, eps=1e-3):

    def loss_fn(model, batch):
        """Compute the loss function.

        Args:
            model: A score model.
            batch: A mini-batch of training data.

        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """
        def score(x, t):
            label = t * 999
            score = model(x, label)
            std = sde.marginal_prob(x, t)[1]
            return -score / std[:, None, None, None]
        
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None, None, None] * z
        score_val = score(perturbed_data, t)

        losses = torch.square(score_val * std[:, None, None, None] + z)
        losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss
    
    return loss_fn
