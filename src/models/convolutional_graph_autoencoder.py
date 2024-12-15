
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn.norm.layer_norm import LayerNorm
from torch_geometric.nn import GAE
from torch.optim import Optimizer
from torch_geometric.data import DataLoader
from torch import Tensor
from sklearn.metrics import average_precision_score, roc_auc_score
import torch  
from tqdm import tqdm

from typing import *


EPS = 1e-15  # Small number to avoid division by zero

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels_node, out_channels, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            in_c = in_channels_node if i == 0 else 2 * out_channels
            out_c = out_channels if i == num_layers - 1 else 2 * out_channels
            self.layers.append(GCNConv(in_c, out_c))
            if i != num_layers - 1:  # Only add LayerNorm for non-final layers
                self.layers.append(LayerNorm(out_c))

    def forward(self, x, edge_index, batch):
        """
        Args:
        x (Tensor): Node features.
        edge_index (Tensor): Graph connectivity in COO format with shape [2, num_edges].
        batch (Tensor): Batch vector which assigns each node to a specific example.

        Returns:
        Tuple[Tensor, Tensor]: Updated node features after message passing and pooled graph-level embedding.
        """
        for i, layer in enumerate(self.layers):
            if isinstance(layer, GCNConv):
                x = layer(x, edge_index)  # Apply GCNConv
                if i < len(self.layers) - 1 and isinstance(self.layers[i + 1], LayerNorm):
                    x = x.relu()  # Apply activation function before the LayerNorm
            elif isinstance(layer, LayerNorm):
                x = layer(x)  # Apply LayerNorm
        
        z_pooled = global_mean_pool(x, batch)  # Pool node embeddings to get graph-level embedding
        return x, z_pooled
    
class GCNGAE(GAE):
    def recon_loss(self, z, pos_edge_index, num_nodes=None,
                   neg_edge_index = None):
        """
        Calculate reconstruction loss using binary cross entropy for positive and negative edges.

        Args:
        z (Tensor): Latent space representation of node features.
        pos_edge_index (Tensor): Indices of positive edges.
        num_nodes (int): Number of nodes in the graph.
        neg_edge_index (Tensor, optional): Indices of negative edges. If None, negative sampling will be performed.

        Returns:
        Tensor: Total loss calculated from positive and negative edges.
        """
        pos_loss = -torch.log(self.decoder(z, pos_edge_index, num_nodes) + EPS).mean()  # Positive edge loss
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, num_nodes)  # Negative edge sampling
        neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, num_nodes) + EPS).mean()  # Negative edge loss
        return pos_loss + neg_loss  # Return total loss
    
    def compute_metrics(self, z, pos_edge_index,
                neg_edge_index, num_nodes=None):
            """
            Evaluate the performance using AUC and average precision (AP) metrics.

            Args:
            z (Tensor): Latent space representation of node features.
            pos_edge_index (Tensor): Indices of positive edges.
            neg_edge_index (Tensor): Indices of negative edges.
            num_nodes (int, optional): Number of nodes in the graph.

            Returns:
            Tuple[float, float]: AUC and average precision scores.
            """
            pos_y = z.new_ones(pos_edge_index.size(1))  # Positive labels
            neg_y = z.new_zeros(neg_edge_index.size(1))  # Negative labels
            y = torch.cat([pos_y, neg_y], dim=0)  # Concatenate labels

            pos_pred = torch.sigmoid(self.decoder(z, pos_edge_index, num_nodes))  # Positive predictions
            neg_pred = torch.sigmoid(self.decoder(z, neg_edge_index, num_nodes))  # Negative predictions

            pred = torch.cat([pos_pred, neg_pred], dim=0)  # Concatenate predictions

            y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()  # Convert to numpy

            return roc_auc_score(y, pred), average_precision_score(y, pred)
        
    def inference(self, data_loader: DataLoader, device: str):
        self.eval()
        latent_dict = {}
        with torch.no_grad():
            for data in tqdm(data_loader):
                data = data.to(device) 
                node_features = torch.cat((data.coords, data.amino_acid_one_hot, data.hbond_acceptors, data.hbond_donors), dim=1) 
                batch = torch.zeros(data.num_nodes, dtype=torch.long).to(device)   
                _, latent_space = self.encode(node_features.to(torch.float32), data.edge_index.to(torch.int64), batch)
                latent_dict.update({data.name[0]: latent_space.tolist()})
        return latent_dict
    
        
        
def train_validate_test(data_loader: DataLoader, model: GCNGAE, optimizer: Optimizer, device: str, mode: str = 'train') -> Union[float, Tuple[float, float, float], Tuple[float, float, Tensor]]:    
    """
    This function handles the training, validation and testing of the MPNNGAE model.

    Args:
        data_loader (DataLoader): A DataLoader object that provides batches of graph data.
        model (MPNNGAE): The specific model to train or evaluate.
        optimizer (Optimizer): The optimizer to use for updating the parameters during training.
        device (str): The device (CPU or GPU) to use for computation.
        mode (str, optional): Indicates whether the function should train, validate or test the model. Defaults to 'train'.

    Returns:
        float: If in 'train' mode, returns the average training loss.
        Tuple[float, float, float]: If in 'validate' mode, returns the average validation loss, average AUC score, and average AP score.
        Tuple[float, float, Tensor]: If in 'test' mode, returns the average AUC score, average AP score, and the encoded latent space of the node features.
    """

    total_loss = 0      # To keep track of total loss
    # Initialize sum of AUC scores  
    auc_sum = 0  
    ap_sum = 0  # Initialize sum of AP scores  
      
    if mode == 'train':     # If mode is 'train', set the model to training mode  
        model.train()    
    else:    # For 'validate' and 'test' mode, set the model to evaluation mode  
        model.eval()      
      
    for data in tqdm(data_loader):  # Loop over each batch of data in the data loader  
        data = data.to(device)  # Move data to the selected device  
        
        optimizer.zero_grad()  # Zero out any stored gradients 

         # Concatenate coordinates and one-hot encoded amino acids to form node features  
        node_features = torch.cat((data.coords, data.amino_acid_one_hot, data.hbond_acceptors, data.hbond_donors), dim=1)    
        # Get the batch of data  
        batch = data.batch  # Get the batch of data  
        z, z_pooled = model.encode(node_features.to(torch.float32), data.edge_index.to(torch.int64), batch)   
        # Compute the reconstruction loss  
        loss = model.recon_loss(z, data.edge_index.to(torch.int64), data.num_nodes)      
          
        # If in 'train' mode, perform backpropagation and update model parameters  
        if mode == 'train':    
            loss.backward()      
            optimizer.step()      
            
        total_loss += loss.item()  # Accumulate loss  
        if mode != 'train':    # If in 'validate' or 'test' mode, compute the ROC AUC score & the AP score  
            auc, ap = model.compute_metrics(z, data.edge_index.to(torch.int64), data.negative_edge_index.to(torch.int64), data.num_nodes)      
            # Accumulate AUC scores  
            auc_sum += auc      
            # Accumulate AP scores  
            ap_sum += ap      
              
    if mode == 'train':   # If in 'train' mode, return average loss  
        return float(total_loss / len(data_loader))      
    elif mode == 'validate':   # If in 'validate' mode, return average loss, average AUC score, and average AP score  
        return float(total_loss / len(data_loader)), auc_sum / len(data_loader), ap_sum / len(data_loader)    
    else:  # If in 'test' mode, return average AUC score, average AP score, and encoded latent space  
        return auc_sum / len(data_loader), ap_sum / len(data_loader), z_pooled  