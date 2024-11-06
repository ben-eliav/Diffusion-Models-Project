import torch
import abc

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
