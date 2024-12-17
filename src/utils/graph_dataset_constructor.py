import os
from functools import partial
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import (
    add_aromatic_interactions,
    add_hydrogen_bond_interactions,
    add_distance_threshold,
    add_aromatic_sulphur_interactions,
    add_disulfide_interactions,
    add_cation_pi_interactions,
    add_peptide_bonds,
    add_hydrophobic_interactions,
)
from graphein.protein.features.nodes.amino_acid import (
    amino_acid_one_hot,
    hydrogen_bond_acceptor,
    hydrogen_bond_donor,
)

from graphein.protein.edges.atomic import add_atomic_edges

from graphein.ml.conversion import GraphFormatConvertor

import torch

from torch_geometric.utils import negative_sampling


from typing import *

from graphein.protein.resi_atoms import (
    COVALENT_RADII,
)

from .custom_protein_graph_dataset import CustomProteinGraphDataset


BASE_ATOMS = [
    "C",
    "H",
    "O",
    "N",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "B",
]


class GraphDatasetConstructor:
    """Class responsible for constructing graph datasets with advanced graph information."""

    def __init__(self, scope: str):
        """Initialize the constructor with a given scope.

        Args:
            scope (str): Scope of the dataset, either 'residue' or 'atomic'.
        """
        if scope not in ["residue", "atomic"]:
            raise ValueError("Scope must be either 'residue' or 'atomic'")
        self.scope = scope

    def __compute_advanced_graph_information(self, data):
        """Compute advanced graph information based on the given scope.
        Private method not meant to be accessed directly

        Args:
            data (_type_): The input graph data.

        Returns:
            _type_: The enriched graph data.
        """
        if self.scope == "residue":
            # Perform negative sampling for residue scope
            data.negative_edge_index = negative_sampling(
                data.edge_index, data.num_nodes
            )

        elif self.scope == "atomic":
            # Add negative edge_index for atomic scope
            data.negative_edge_index = negative_sampling(
                data.edge_index, data.num_nodes
            )

            # Add atom encodings
            atom_encodings = []
            for element in data.element_symbol:
                one_hot_encoding = [0] * len(BASE_ATOMS)
                if element in BASE_ATOMS:
                    one_hot_encoding[BASE_ATOMS.index(element)] = 1
                atom_encodings.append(one_hot_encoding)

            # Assign atom encodings
            data.atom_encoding = torch.tensor(atom_encodings)

            # Construct scaled edge features for covalent bonds
            edge_distances = torch.tensor(data.bond_length)
            # Create a mask of NaN values
            mask = torch.isnan(data.bond_length)
            # Replace NaN values with zero
            edge_distances[mask] = 0

            # Minimum value is zero for edges not considered as a covalent bond
            tensor_min = torch.tensor(0)

            # Max distance to be considered as covalent bond is max of Covalent Radii * 2 (each atom has a covalent radius) + 0.56 (Tolerance)
            tensor_max = torch.tensor(max(list(COVALENT_RADII.values())) * 2 + 0.56)

            # Min-Max Scaling of distances
            scaled_tensor = (edge_distances - tensor_min) / (tensor_max - tensor_min)

            # Assign scaled edge distance
            data.scaled_edge_distance = scaled_tensor

        return data

    def construct_or_load_dataset(
        self,
        dataset_name: str,
        input_pdb_paths: list[str],
        output_path: str,
        num_cores: int,
        chunk_size: int,
    ) -> CustomProteinGraphDataset:
        """Construct or load the dataset based on the given scope.

        Args:
            dataset_name (str): The name of the dataset.
            input_pdb_paths (list[str]): Input pathes of pdb files
            output_path (str): Path to save the constructed dataset.
            num_cores (int): Number of cores to use for processing.
            chunk_size (int): Chunk size used to process the graphs in a batch manner. Lower in case you run out of memory.

        Returns:
            _type_: The constructed dataset.
        """

        if self.scope == "residue":
            # Define edge construction functions for residue scope
            edge_const_func = {
                "edge_construction_functions": [
                    partial(
                        add_distance_threshold,
                        threshold=5,
                        long_interaction_threshold=0,
                    ),
                    add_aromatic_interactions,
                    add_hydrogen_bond_interactions,
                    add_hydrophobic_interactions,
                    add_aromatic_sulphur_interactions,
                    add_disulfide_interactions,
                    add_cation_pi_interactions,
                    add_peptide_bonds,
                ]
            }
            # Define node metadata functions for residue scope
            node_meta_func = {
                "node_metadata_functions": [
                    amino_acid_one_hot,
                    hydrogen_bond_acceptor,
                    hydrogen_bond_donor,
                ]
            }
            # Configure the protein graph with the defined functions
            config = ProteinGraphConfig(**{**edge_const_func, **node_meta_func})
            # Define the format convertor for the data
            convertor = GraphFormatConvertor(
                src_format="nx",
                dst_format="pyg",
                verbose=False,
                columns=[
                    "coords",
                    "edge_index",
                    "amino_acid_one_hot",
                    "hbond_acceptors",
                    "hbond_donors",
                    "name",
                    "num_nodes",
                    "negative_edge_index",
                    "kind",
                ],
            )

        elif self.scope == "atomic":

            edge_const_func = {"edge_construction_functions": [add_atomic_edges]}

            config = ProteinGraphConfig(**{**edge_const_func, "granularity": "atom"})
            convertor = GraphFormatConvertor(
                src_format="nx",
                dst_format="pyg",
                verbose="False",
                columns=[
                    "coords",
                    "edge_index",
                    "atom_encoding",
                    "element_symbol",
                    "name",
                    "num_nodes",
                    "negative_edge_index",
                    "bond_length",
                    "kind",
                    "scaled_edge_distance",
                ],
            )

        # Create the dataset with advanced graph information
        ds = CustomProteinGraphDataset(
            chunk_size=chunk_size,
            root=f"{os.path.join(output_path, dataset_name)}",
            paths=input_pdb_paths,
            graph_format_convertor=convertor,
            graph_transformation_funcs=[],
            transform=self.__compute_advanced_graph_information,
            graphein_config=config,
            num_cores=num_cores,
        )

        return ds
