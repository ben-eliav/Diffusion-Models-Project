import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm
import os
from score_model import UNet
from sdes import *
from generation import *



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


## sampling
def sample(model_config):
    model = UNet(
        ch=model_config["channel"],
        ch_mult=model_config["channel_mult"],
        attn=model_config["attn"],
        num_res_blocks=model_config["num_res_blocks"],
        dropout=model_config["dropout"]
    ).to(model_config["device"])
    
    ckpt = torch.load(os.path.join(
        model_config["save_weight_dir"], model_config["test_load_weight"]), map_location=model_config["device"], weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()

    sde = VPSDE(model_config["beta_1"], model_config["beta_T"], model_config["T"])
    predictor = NullPredictor(model, sde)
    corrector = LangevinCorrector(model, sde)

    x = torch.randn(model_config["nrow"] ** 2, 1, model_config["img_size"], model_config["img_size"]).to(model_config["device"])
    noisy_images = x.cpu()
    with torch.no_grad():
        for t in tqdm(reversed(range(model_config["pc_steps"])), total=model_config["pc_steps"]):
            t = torch.tensor([t / model_config["pc_steps"]]).to(model_config["device"])
            x = predictor_corrector_step(model_config, predictor, corrector, x, t)

    sampled_images = x.cpu()
    sampled_images = torch.clamp(sampled_images, 0, 1)
    noisy_images = torch.clamp(noisy_images, 0, 1)

    save_image(sampled_images, model_config["sampled_dir"] + model_config["sampledImgName"], nrow=model_config["nrow"])