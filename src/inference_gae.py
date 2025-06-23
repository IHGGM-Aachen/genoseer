import os
import argparse
from utils.parse_config import _parse_config
from utils.graph_dataset_constructor import GraphDatasetConstructor
from models.convolutional_graph_autoencoder import GCNGAE, GCNEncoder
from models.mpnn_graph_autoencoder import MPNNGAE, MPNNEncoder
from torch_geometric.nn.models.autoencoder import InnerProductDecoder
from torch_geometric.loader import DataLoader
import torch
import pandas as pd
import logging


def setup_logging():
    log_dir = "./src/logs/"
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger_handler = logging.FileHandler(os.path.join(log_dir, "model_inference.log"))
    logger_handler.setLevel(logging.INFO)
    logger_handler.setFormatter(formatter)

    inference_logger = logging.getLogger("InferenceLogger")
    inference_logger.setLevel(logging.INFO)
    inference_logger.addHandler(logger_handler)

    return inference_logger


def parse_arguments():
    parser = argparse.ArgumentParser(description="Create protein graph dataset.")
    parser.add_argument(
        "--config_path", type=str, help="Path to the model and training config yaml"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments and config file
    args = parse_arguments()
    config = _parse_config(args.config_path)

    # Setup logger to monitor the training process
    logger = setup_logging()

    # Get the paths of the pdb files to be used for training and testing
    logger.info("Getting list of pdbs paths")
    pdb_paths = [
        os.path.join(config["input_dir_pdbs"], pdb_path)
        for pdb_path in os.listdir(config["input_dir_pdbs"])
        if pdb_path.endswith(".pdb")
    ]

    # Split the paths into training, validation, and testing paths
    training_paths = pdb_paths[: int(0.7 * len(pdb_paths))]
    validation_paths = pdb_paths[int(0.7 * len(pdb_paths)) : int(0.9 * len(pdb_paths))]
    testing_paths = pdb_paths[int(0.9 * len(pdb_paths)) :]

    # Initialize a GraphDatasetConstructor instance
    logger.info("Creating instance of GraphDatasetConstructor")
    constructor = GraphDatasetConstructor(scope=config["scope"])

    # Construct the training dataset
    logger.info("Loading training data.")
    ds_train = constructor.construct_or_load_dataset(
        f"{config['dataset_name']}_train",
        training_paths,
        config["output_path_datasets"],
        config["num_cores"],
        config["chunk_size"],
    )
    logger.info("Training data loaded!")

    # Construct the validation dataset
    logger.info("Loading validation data.")
    ds_valid = constructor.construct_or_load_dataset(
        f"{config['dataset_name']}_valid",
        validation_paths,
        config["output_path_datasets"],
        config["num_cores"],
        config["chunk_size"],
    )
    logger.info("Validation data loaded!")

    # Construct the test dataset
    logger.info("Loading test data.")
    ds_test = constructor.construct_or_load_dataset(
        f"{config['dataset_name']}_test",
        testing_paths,
        config["output_path_datasets"],
        config["num_cores"],
        config["chunk_size"],
    )
    logger.info("Test data loaded!")

    datasets = [ds_train, ds_valid, ds_test]

    # Grab model and training parameters from config
    num_layers = config["num_layers"]
    out_channels = config["out_channels"]

    # Depending on the scope, initialize the respective model (GCNGAE for "residue" or MPNNGAE for "atomic")
    if config["scope"] == "residue":
        num_features = (
            ds_train[0].coords.size(1)
            + ds_train[0].amino_acid_one_hot.size(1)
            + ds_train[0].hbond_acceptors.size(1)
            + ds_train[0].hbond_donors.size(1)
        )
        model = GCNGAE(
            GCNEncoder(num_features, out_channels, num_layers), InnerProductDecoder()
        )
        model.load_state_dict(
            torch.load(
                os.path.join(config["model_output_dir"], f"{config['model_name']}.pt")
            )
        )
        logger.info(f"Loaded model states for {config['model_name']}")

    elif config["scope"] == "atomic":
        num_features = ds_train[0].coords.size(1) + ds_train[0].atom_encoding.size(1)
        model = MPNNGAE(
            MPNNEncoder(num_features, out_channels, num_layers), InnerProductDecoder()
        )
        model.load_state_dict(
            torch.load(
                os.path.join(config["model_output_dir"], f"{config['model_name']}.pt")
            )
        )
        logger.info(f"Loaded model states for {config['model_name']}")

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    df_embeddings = pd.DataFrame()
    for dataset in datasets:
        # Initialize the data loaders for training, validation and testing data
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        latent_spaces = model.inference(data_loader, device)
        df_dataset = pd.DataFrame(
            {
                "structure_name": latent_spaces.keys(),
                "structural_embeddings": latent_spaces.values(),
            }
        )
        df_embeddings = pd.concat([df_embeddings, df_dataset])

    if not os.path.isdir(config["output_path_embeddings"]):
        os.mkdir(config["output_path_embeddings"])

    df_embeddings.to_csv(
        os.path.join(
            config["output_path_embeddings"], f"{config['scope']}_embeddings.csv"
        ),
        index=False,
    )

    logger.info("Embeddings created.")
