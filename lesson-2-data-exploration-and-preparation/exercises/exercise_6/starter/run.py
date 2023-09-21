#!/usr/bin/env python

import logging
import os
import tempfile

import hydra
from omegaconf import DictConfig

import pandas as pd
import wandb
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@hydra.main(config_name='config')
def go(config: DictConfig):

    wandb.login(key=config['wandb']['api_key'])
    run = wandb.init(project=config['wandb']['project'], job_type=config['wandb']['job_type'])

    logger.info("Downloading and reading artifact")
    artifact = run.use_artifact(config['artifact']['input'])
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path, low_memory=False)

    # Split model_dev/test
    logger.info("Splitting data into train and test")
    splits = {}

    ###################################
    # COMPLETE the following line     #
    ###################################

    splits["train"], splits["test"] = train_test_split(
        df,
        test_size=config['parameters']['test_size'],
        random_state=config['parameters']['random_state'],
        stratify=df[config['parameters']['stratify']] if config['parameters']['stratify'] != 'null' else None,
    )

    # Now we save the artifacts. We use a temporary directory so we do not leave
    # any trace behind
    with tempfile.TemporaryDirectory() as tmp_dir:

        for split, df in splits.items():

            # Make the artifact name from the provided root plus the name of the split
            artifact_name = f"{config['artifact']['output_root_name']}_{split}.csv"

            # Get the path on disk within the temp directory
            temp_path = os.path.join(tmp_dir, artifact_name)

            logger.info(f"Uploading the {split} dataset to {artifact_name}")

            # Save then upload to W&B
            df.to_csv(temp_path)

            artifact = wandb.Artifact(
                name=artifact_name,
                type=config['artifact']['type'],
                description=f"{split} split of dataset {config['artifact']['input']}",
            )
            artifact.add_file(temp_path)

            logger.info("Logging artifact")
            run.log_artifact(artifact)

            # This waits for the artifact to be uploaded to W&B. If you
            # do not add this, the temp directory might be removed before
            # W&B had a chance to upload the datasets, and the upload
            # might fail
            artifact.wait()


if __name__ == "__main__":
    go()
