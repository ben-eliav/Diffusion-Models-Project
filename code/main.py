from training import train, sample

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