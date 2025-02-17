from typing import Optional

import torch
import torch.nn as nn
import numpy as np

class DeepSets(nn.Module):
    """
    This class applies a DeepSets-based distillation of a list of embeddings into a single embedding.
    
    Parameters
    ----------
    input_size : int
        The dimension of each embedding vector.
    phi_hidden_size : int, optional
        The hidden size for the phi network, by default 128.
    rho_hidden_size : int, optional
        The hidden size for the rho network, by default 128.
    output_size : int, optional
        The dimension of the distilled embedding, by default same as input_size.
    device : str, optional
        Device to run the model on, e.g., 'cpu' or 'cuda', by default 'cpu'.
    """
    def __init__(
        self,
        input_size: int,
        phi_hidden_size: int = 128,
        rho_hidden_size: int = 128,
        output_size: Optional[int] = None,
        device: str = 'cpu'
    ):
        super(DeepSets, self).__init__()
        self.device = device
        
        self.phi_network = nn.Sequential(
            nn.Linear(input_size, phi_hidden_size),
            nn.ReLU(),
            nn.Linear(phi_hidden_size, phi_hidden_size),
            nn.ReLU()
        ).to(device)
        
        self.rho_network = nn.Sequential(
            nn.Linear(phi_hidden_size, rho_hidden_size),
            nn.ReLU(),
            nn.Linear(rho_hidden_size, output_size if output_size else input_size)
        ).to(device)

    def forward(self, embeddings_list: np.ndarray) -> np.ndarray:
        """
        Apply the DeepSets algorithm to distill a list of embeddings into a single embedding.
        
        Parameters
        ----------
        embeddings_list : np.ndarray
            A numpy array of shape (N, D) where N is the number of embeddings and D is the 
            embedding dimension.
        
        Returns
        -------
        np.ndarray
            A single distilled embedding of shape (output_size,).
        """
        with torch.no_grad():
            x = torch.tensor(embeddings_list, dtype=torch.float32, device=self.device)
            x_phi = self.phi_network(x)
            x_sum = torch.sum(x_phi, dim=0, keepdim=True)
            out = self.rho_network(x_sum)
            return out.cpu().numpy().flatten()
