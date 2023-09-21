#!/usr/bin/env python

import logging
import pandas as pd
import wandb
import os
import hydra
from omegaconf import DictConfig


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@hydra.main(config_name='config')
def go(config: DictConfig):

    wandb.login(key=config['wandb']['api_key'])
    run = wandb.init(project=config['wandb']['project'], job_type=config['wandb']['job_type'])

    logger.info("Reading artifact")
    df = pd.read_parquet(config['artifact']['path'])

    # Drop the duplicates
    logger.info("Dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=True)

    logger.info("Fixing missing values")
    # These are missing values that are due to an old version of the data. On new data,
    # because of a change in the web form used to register new songs, the title and the
    # song name are already empty strings
    df['title'].fillna(value='', inplace=True)
    df['song_name'].fillna(value='', inplace=True)
    df['text_feature'] = df['title'] + ' ' + df['song_name']

    filename = "processed_data.csv"
    df.to_csv(filename)

    artifact = wandb.Artifact(
        name=config['artifact']['name'],
        type=config['artifact']['type'],
        description=config['artifact']['description'],
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(filename)


if __name__ == "__main__":
    go()
