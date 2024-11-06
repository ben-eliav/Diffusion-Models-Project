import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os
import abc
import matplotlib.pyplot as plt
from score_model import UNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


## Architecture: (UNET)



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
        self.betas = torch.linspace(beta_0, beta_T, T).to(device)  # discretized
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
    

class VPSDE_copy:
    pass


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
        t = torch.rand(x.shape[0], device=device)
        noise = torch.randn_like(x)
        mean, std = self.sde.dist(x, t)
        x_t = mean + std * noise
        score_prediction = model(x_t, t)
        return F.mse_loss(score_prediction * std, -noise)



## training loop
def train(model_config):
    model = UNet(
        ch=model_config["channel"],
        ch_mult=model_config["channel_mult"],
        attn=model_config["attn"],
        num_res_blocks=model_config["num_res_blocks"],
        dropout=model_config["dropout"]
    ).to(model_config["device"])

    sde = VPSDE(model_config["beta_1"], model_config["beta_T"], model_config["T"])
    sde_loss = SDELoss(sde)
    optimizer = optim.Adam(model.parameters(), lr=model_config["lr"])
    model.train()

    train_dataset = MNIST(
        root="./data", train=True, download=True,
        transform=transforms.Compose([
            transforms.Resize(model_config["img_size"]),
            transforms.ToTensor()
        ])
    )
    train_loader = DataLoader(
        train_dataset, batch_size=model_config["batch_size"], shuffle=True
    )

    if not os.path.exists(model_config["save_weight_dir"]):
        os.makedirs(model_config["save_weight_dir"])

    if model_config["training_load_weight"]:
        model.load_state_dict(torch.load(model_config["training_load_weight"]))

    for epoch in range(model_config["epoch"]):
        for x, _ in tqdm(train_loader):
            x = x.to(model_config["device"])
            optimizer.zero_grad()
            loss = sde_loss(model, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model_config["grad_clip"])
            optimizer.step()
        if model_config["show_process"]:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        torch.save(model.state_dict(), model_config["save_weight_dir"] + f"ckpt_{epoch}_.pt")

    
## predictor - corrector framework
class Predictor(abc.ABC):
    def __init__(self, model, sde):
        self.model = model
        self.sde = sde

    def predictor_step(self, prev_x, t):
        pass


class Corrector(abc.ABC):
    def __init__(self, model, sde):
        self.model = model
        self.sde = sde

    def corrector_step(self, prev_x, t):
        pass

class ReverseDiffusionPredictor(Predictor):
    def __init__(self, model, sde):
        super().__init__(model, sde)
    
    def predictor_step(self, prev_x, t):
        drift, diffusion = self.sde.reverse_sde(self.model, prev_x, t)
        return prev_x - drift + diffusion ** 2 * self.model(prev_x, t) + diffusion * torch.randn_like(prev_x).to(prev_x.device)
    
    
class AncestralSamplingPredictor(Predictor):
    def __init__(self, model, sde):
        super().__init__(model, sde)

    def predictor_step(self, x, t):
        sde = self.sde
        timestep = (t * (sde.T - 1)).long()
        beta = sde.betas.to(t.device)[timestep]
        score = self.model(x, t)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

class NullCorrector(Corrector):
    def __init__(self, model, sde):
        super().__init__(model, sde)
        
    def corrector_step(self, x, t):
        return x
    
class NullPredictor(Predictor):
    def __init__(self, model, sde):
        super().__init__(model, sde)
        
    def predictor_step(self, x, t):
        return x
        

class LangevinDynamicsCorrector(Corrector):
    """
    Based on Algorithm 5 in the paper.
    """
    def __init__(self, model, sde, corrector_step_size):
        super().__init__(model, sde)
        self.corrector_step_size = corrector_step_size
    
    def corrector_step(self, prev_x, t):
        score = self.model(prev_x, t)
        noise = torch.randn_like(prev_x).to(prev_x.device)
        score_norm = torch.linalg.norm(score.reshape(score.shape[0], -1), dim=1)
        noise_norm = torch.linalg.norm(noise.reshape(noise.shape[0], -1), dim=1)
        step_size = 2 * self.sde.alpha[(t*self.sde.T).int()] * (self.corrector_step_size * noise_norm / score_norm) ** 2
        return prev_x + step_size[:, None, None, None] * score + torch.sqrt(2 * step_size)[:, None, None, None] * noise

class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr=0.075, n_steps=3):
        super().__init__(sde, score_fn)
        self.snr = snr
        self.n_steps = n_steps

    def corrector_step(self, x, t):
        sde = self.sde
        score_fn = self.model
        n_steps = self.n_steps
        target_snr = self.snr
        timestep = (t * (sde.T - 1)).long()
        alpha = sde.alpha.to(t.device)[timestep]

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise
            

        return x


def predictor_corrector_step(model_config, predictor, corrector, x, t):
    """
    predictor: Predictor
    corrector: Corrector
    x: torch.Tensor(B, C, H, W)
    t: torch.Tensor(B)
    """
    if isinstance(predictor, AncestralSamplingPredictor):
        x = predictor.predictor_step(x, t)[0]
    else:
        x = predictor.predictor_step(x, t)
    for _ in range(model_config['corrector_steps']):
        x = corrector.corrector_step(x, t)
    return x


## sampling
def sample(model_config):
    model = UNet(
        ch=model_config["channel"],
        ch_mult=model_config["channel_mult"],
        attn=model_config["attn"],
        num_res_blocks=model_config["num_res_blocks"],
        dropout=model_config["dropout"]
    ).to(device)
    
    ckpt = torch.load(os.path.join(
        model_config["save_weight_dir"], model_config["test_load_weight"]), map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    sde = VPSDE(model_config["beta_1"], model_config["beta_T"], model_config["T"])
    predictor = NullPredictor(model, sde)
    corrector = LangevinCorrector(model, sde)

    x = torch.randn(model_config["nrow"] ** 2, 1, model_config["img_size"], model_config["img_size"]).to(device)
    noisy_images = x.cpu()
    with torch.no_grad():
        for t in tqdm(reversed(range(model_config["pc_steps"])), total=model_config["pc_steps"]):
            t = torch.tensor([t / model_config["pc_steps"]]).to(device)
            x = predictor_corrector_step(model_config, predictor, corrector, x, t)

    sampled_images = x.cpu()
    sampled_images = torch.clamp(sampled_images, 0, 1)
    noisy_images = torch.clamp(noisy_images, 0, 1)

    save_image(sampled_images, model_config["sampled_dir"] + model_config["sampledImgName"], nrow=model_config["nrow"])


model_config = {
    "state": "train",
    "epoch": 50,
    "batch_size": 64,
    "T": 1000,
    "channel": 32,
    "channel_mult": [1, 2],
    "attn": [],
    "num_res_blocks": 2,
    "dropout": 0.15,
    "lr": 5e-4,
    "multiplier": 2.,
    "beta_1": 1e-4,
    "beta_T": 0.02,
    "img_size": 28,
    "grad_clip": 1.,
    "device": "cuda:0",
    "training_load_weight": None,
    "save_weight_dir": "./Checkpoints/",
    "test_load_weight": "ckpt_49_.pt",
    "sampled_dir": "",
    "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
    "sampledImgName": "SampledNoGuidenceImgs.png",
    "nrow": 8,
    "show_process": True,
    "corrector_steps": 3,
    "corrector_step_size": 0.1,
    "pc_steps": 1000,
}

if __name__ == '__main__':
    if model_config["state"] == "train":
        train(model_config)
    elif model_config["state"] == "sample":
        sample(model_config)
    else:
        raise ValueError("Invalid state. Choose either 'train' or 'sample'.")