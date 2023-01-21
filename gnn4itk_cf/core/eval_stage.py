"""
This script:
1. Loads a training config
2. Checks the stage to train
3. Loads the stage module
4. Trains the stage
5. Tests the output
"""

import sys
import os
from pathlib import Path

import yaml
import click
import logging

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule
import torch

from gnn4itk_cf.utils import str_to_class, find_latest_checkpoint

@click.command()
@click.argument("config_file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose mode")

def main(config_file, verbose):
    """
    Main function to train a stage. Separate the main and train_stage functions to allow for testing.
    """
    # set up logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    evaluate(config_file)

def evaluate(config_file):

    # load config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # load stage
    stage = config["stage"]
    model = config["model"]
    stage_module = str_to_class(stage, model)

    if issubclass(stage_module, LightningModule):
        checkpoint_path = find_latest_checkpoint(config["stage_dir"], templates=["best*.ckpt", "*.ckpt"])
        if not checkpoint_path:
            print("No checkpoint found")
            sys.exit(1)
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint_config = torch.load(checkpoint_path)["hyper_parameters"]
        config = checkpoint_config | config

    stage_module.evaluate(config)

if __name__ == "__main__":
    main()