import os
from graphein.protein.graphs import construct_graphs_mp
from graphein.ml.datasets.torch_geometric_dataset import ProteinGraphDataset
from tqdm import tqdm
import typing as ty
import torch


class CustomProteinGraphDataset(ProteinGraphDataset):
    """Custom ProteinGraphDataset class to adjust chunk size in case of too high memory pressure
    """
    def __init__(self, chunk_size: int=64, *args, **kwargs):
        self.chunk_size = chunk_size
        super().__init__(*args, **kwargs)
        
    def process(self):
        """Processes structures from files into PyTorch Geometric Data."""
        # Preprocess PDB files
        if self.pdb_transform:
            self.transform_pdbs()


        # Chunk dataset for parallel processing
        def divide_chunks(l: ty.List[str], n: int = 2) -> ty.Generator:
            for i in range(0, len(l), n):
                yield l[i : i + n]

        chunks: ty.List[int] = list(
            divide_chunks(list(self.examples.keys()), self.chunk_size)
        )

        for chunk in tqdm(chunks):
            pdbs = [self.examples[idx] for idx in chunk]
            # Get chain selections
            if self.chain_selection_map is not None:
                chain_selections = [self.chain_selection_map[idx] for idx in chunk]
            else:
                chain_selections = ["all"] * len(chunk)

            # Create graph objects
            file_names = [f"{self.raw_dir}/{pdb}.pdb" for pdb in pdbs]

            graphs = construct_graphs_mp(
                path_it=file_names,
                config=self.config,
                chain_selections=chain_selections,
                return_dict=False,
            )
            if self.graph_transformation_funcs is not None:
                graphs = [self.transform_graphein_graphs(g) for g in graphs]

            # Convert to PyTorch Geometric Data
            graphs = [self.graph_format_convertor(g) for g in graphs]

            # Assign labels
            if self.graph_label_map:
                labels = [self.graph_label_map[idx] for idx in chunk]
                for i, _ in enumerate(chunk):
                    graphs[i].graph_y = labels[i]
            if self.node_label_map:
                labels = [self.node_label_map[idx] for idx in chunk]
                for i, _ in enumerate(chunk):
                    graphs[i].graph_y = labels[i]

            data_list = graphs

            del graphs

            if self.pre_filter is not None:
                data_list = [g for g in data_list if self.pre_filter(g)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            for i, (pdb, chain) in enumerate(zip(pdbs, chain_selections)):
                if self.chain_selection_map is None:
                    torch.save(
                        data_list[i],
                        os.path.join(self.processed_dir, f"{pdb}.pt"),
                    )
                else:
                    torch.save(
                        data_list[i],
                        os.path.join(self.processed_dir, f"{pdb}_{chain}.pt"),
                    )
