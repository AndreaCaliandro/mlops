import json
import argparse

import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Serialize decision tree configuration
    # model_config = os.path.abspath("random_forest_config.yml")

    # with open(model_config, "w+") as fp:
    #     fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

    # model_config_dict = OmegaConf.to_container(config["random_forest_pipeline"], resolve=True)
    # print(model_config_dict)
    # assert isinstance(model_config_dict, dict)
    # json_object = json.dumps(model_config_dict, indent = 4)
    _ = mlflow.run(
        os.path.join(root_path, "random_forest"),
        "main",
        parameters={
            "hydra_options": config["random_forest"]["hydra_options"]
        },
        build_image=True
    )


if __name__ == "__main__":
    go()
