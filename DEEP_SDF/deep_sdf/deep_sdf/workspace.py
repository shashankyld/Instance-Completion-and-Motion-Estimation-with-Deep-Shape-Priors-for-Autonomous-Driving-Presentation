import json
import os
import torch
import click
from os.path import join, dirname, abspath


model_params_subdir = "ModelParameters"

def config_decoder(experiment_directory, checkpoint="latest"): # 2000 epochs

    # use the lastest checkpoint
    specs_filename = os.path.join(experiment_directory, "specs.json")
    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs.json"'
        )

    specs = json.load(open(specs_filename))
    arch = __import__("deep_sdf.networks." + specs["NetworkArch"], fromlist=["Decoder"])
    latent_size = specs["CodeLength"]
    decoder = arch.Decoder(latent_size, **specs["NetworkSpecs"])
    decoder = torch.nn.DataParallel(decoder)
    saved_model_state = torch.load(
        os.path.join(experiment_directory, model_params_subdir, checkpoint + ".pth")
        , map_location=torch.device('cpu') # CPU
    )
    decoder.load_state_dict(saved_model_state["model_state_dict"])
    decoder = decoder.module

    decoder.eval()

    return decoder