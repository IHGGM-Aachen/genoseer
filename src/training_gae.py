import os
import argparse
from utils.parse_config import _parse_config
from utils.graph_dataset_constructor import GraphDatasetConstructor
from models.convolutional_graph_autoencoder import GCNGAE, GCNEncoder
from models.mpnn_graph_autoencoder import MPNNGAE, MPNNEncoder
from torch_geometric.nn.models.autoencoder import InnerProductDecoder
from torch_geometric.loader import DataLoader
import torch
from copy import deepcopy
import logging


def setup_logging():
    log_dir = "./src/logs/"
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger_handler = logging.FileHandler(os.path.join(log_dir, "model_training.log"))
    logger_handler.setLevel(logging.INFO)
    logger_handler.setFormatter(formatter)

    training_logger = logging.getLogger("TrainingLogger")
    training_logger.setLevel(logging.INFO)
    training_logger.addHandler(logger_handler)

    return training_logger


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
    logger.info("Construction of training data started.")
    ds_train = constructor.construct_or_load_dataset(
        f"{config['dataset_name']}_train",
        training_paths,
        config["output_path_datasets"],
        config["num_cores"],
        config["chunk_size"],
    )
    logger.info("Construction of training data done!")

    # Construct the validation dataset
    logger.info("Construction of validation data started.")
    ds_valid = constructor.construct_or_load_dataset(
        f"{config['dataset_name']}_valid",
        validation_paths,
        config["output_path_datasets"],
        config["num_cores"],
        config["chunk_size"],
    )
    logger.info("Construction of validation data done!")

    # Construct the test dataset
    logger.info("Construction of test data started.")
    ds_test = constructor.construct_or_load_dataset(
        f"{config['dataset_name']}_test",
        testing_paths,
        config["output_path_datasets"],
        config["num_cores"],
        config["chunk_size"],
    )

    # Initialize the data loaders for training, validation and testing data
    logger.info("Setting up dataloaders.")
    data_loader_train = DataLoader(
        ds_train, batch_size=config["batch_size"], shuffle=True, drop_last=True
    )
    data_loader_valid = DataLoader(
        ds_valid, batch_size=config["batch_size"], shuffle=False, drop_last=True
    )
    data_loader_test = DataLoader(
        ds_test, batch_size=config["batch_size"], shuffle=False, drop_last=True
    )

    # Grab model and training parameters from config
    num_layers = config["num_layers"]
    out_channels = config["out_channels"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    learning_rate = config["lr"]
    early_stop_counter = 0
    early_stop_patience = config["early_stop_patience"]

    # Depending on the scope, initialize the respective model (GCNGAE for "residue" or MPNNGAE for "atomic")
    if config["scope"] == "residue":
        print(ds_train)
        num_features = (
            ds_train[0].coords.size(1)
            + ds_train[0].amino_acid_one_hot.size(1)
            + ds_train[0].hbond_acceptors.size(1)
            + ds_train[0].hbond_donors.size(1)
        )
        model = GCNGAE(
            GCNEncoder(num_features, out_channels, num_layers), InnerProductDecoder()
        )
        logger.info("Created instance of GCNGAE")
        from models.convolutional_graph_autoencoder import train_validate_test
    elif config["scope"] == "atomic":
        num_features = ds_train[0].coords.size(1) + ds_train[0].atom_encoding.size(1)
        model = MPNNGAE(
            MPNNEncoder(num_features, out_channels, num_layers), InnerProductDecoder()
        )
        logger.info("Created instance of MPNNGAE")
        from models.mpnn_graph_autoencoder import train_validate_test

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Initialize the optimizer for the model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Placeholder for the best validation loss for early stopping
    best_loss = float("inf")

    # Start the training process
    logger.info("Starting to train the model.")
    for epoch in range(1, epochs + 1):
        # Perform training and log the training loss
        train_loss = train_validate_test(
            data_loader_train, model, optimizer, device, mode="train"
        )
        logger.info(f"Epoch: {epoch}, Train Loss: {train_loss}")

        # Perform validation and log the validation loss
        valid_loss, valid_auc, valid_ap = train_validate_test(
            data_loader_valid, model, optimizer, device, mode="validate"
        )
        logger.info(
            f"Epoch: {epoch}, Validation Loss: {valid_loss}, Validation AUC: {valid_auc}, Validation AP: {valid_ap}"
        )

        # Implement early stopping if validation loss doesn't improve
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_wts = deepcopy(model.state_dict())
        else:
            early_stop_counter += 1
            if early_stop_counter == early_stop_patience:
                logger.info(f"Early stopping on epoch {epoch}")
                break

    # Load the best model weights after completion of training
    model.load_state_dict(best_model_wts)

    # Use the best model to perform testing and provide the AUC and AP metrics
    test_auc, test_ap, _ = train_validate_test(
        data_loader_test, model, optimizer, device, mode="test"
    )
    logger.info("Test Metrics - AUC: {:.4f}, AP: {:.4f}".format(test_auc, test_ap))

    # Save the trained model for future use
    if not os.path.isdir(config["model_output_dir"]):
        os.mkdir(config["model_output_dir"])

    torch.save(
        model.state_dict(),
        f"{os.path.join(config['model_output_dir'], config['model_name'])}.pt",
    )
    logger.info("Finished training and saved model.")
