"""
Multi-Scale Graph Neural Network for Particle-Based Simulations.

This module implements a multi-scale GNN following the Encode-Process-Decode pattern
with cross-scale message passing operations. The architecture supports:

1. Multi-scale graph hierarchy: Grid (scale 0) → Mesh Level 1 → Mesh Level 2
2. Cross-scale message passing: Grid2Mesh, Mesh2Mesh, Mesh2Grid
3. Edge updates: Both nodes and edges are updated during message passing
4. Residual connections: Following original GNS InteractionNetwork pattern

Architecture:
- Encoder: Grid node MLP + edge MLPs for all edge types
- Processor: Multiple rounds of mesh2mesh message passing
- Decoder: Mesh2grid message passing + prediction head

All message passing blocks follow the InteractionNetwork pattern from original GNS,
ensuring both node and edge features are updated with residual connections.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any, List
from torch_geometric.nn import MessagePassing

def build_mlp(
        input_size: int,
        hidden_layer_sizes: List[int],
        output_size: int = None,
        output_activation: nn.Module = nn.Identity,
        activation: nn.Module = nn.ReLU) -> nn.Module:
  """Build a MultiLayer Perceptron.

  Args:
    input_size: Size of input layer.
    layer_sizes: An array of input size for each hidden layer.
    output_size: Size of the output layer.
    output_activation: Activation function for the output layer.
    activation: Activation function for the hidden layers.

  Returns:
    mlp: An MLP sequential container.
  """
  # Size of each layer
  layer_sizes = [input_size] + hidden_layer_sizes
  if output_size:
    layer_sizes.append(output_size)

  # Number of layers
  nlayers = len(layer_sizes) - 1

  # Create a list of activation functions and
  # set the last element to output activation function
  act = [activation for i in range(nlayers)]
  act[-1] = output_activation

  # Create a torch sequential container
  mlp = nn.Sequential()
  for i in range(nlayers):
    mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i],
                                             layer_sizes[i + 1]))
    mlp.add_module("Act-" + str(i), act[i]())

  return mlp

class G2MBlock(MessagePassing):
    """Grid to Mesh Message Passing Block (following InteractionNetwork pattern).
    
    Updates both mesh nodes and grid2mesh edges, consistent with original GNS.
    """
    def __init__(self, nnode_in: int, nnode_out: int, nedge_in: int, nedge_out: int, nmlp_layers: int, latent_dim: int):
        super().__init__(aggr='add')
        # Node MLP: combines aggregated edge messages with node features
        self.node_fn = nn.Sequential(*[
            build_mlp(nnode_in + nedge_out, [latent_dim for _ in range(nmlp_layers)], nnode_out),
            nn.LayerNorm(nnode_out)
        ])
        # Edge MLP: combines source (grid) and target (mesh) node features with edge features
        self.edge_fn = nn.Sequential(*[
            build_mlp(nnode_in + nnode_in + nedge_in, [latent_dim for _ in range(nmlp_layers)], nedge_out),
            nn.LayerNorm(nedge_out)
        ])
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor):
        """Forward pass for grid2mesh message passing.
        
        Args:
            x: Node features [num_nodes, nnode_in]
            edge_index: Grid2mesh edge indices [2, num_edges] (source=grid, target=mesh)
            edge_features: Edge features [num_edges, nedge_in]
        """
        x_residual = x
        edge_features_residual = edge_features
        
        x, edge_features = self.propagate(edge_index=edge_index, x=x, edge_features=edge_features)
        
        return x + x_residual, edge_features + edge_features_residual
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        """Construct message from grid node j to mesh node i."""
        # x_i: mesh node features, x_j: grid node features
        edge_features = torch.cat([x_i, x_j, edge_features], dim=-1)
        edge_features = self.edge_fn(edge_features)
        return edge_features
    
    def update(self, x_updated: torch.Tensor, x: torch.Tensor, edge_features: torch.Tensor):
        """Update node features using aggregated messages."""
        x_updated = torch.cat([x_updated, x], dim=-1)
        x_updated = self.node_fn(x_updated)
        return x_updated, edge_features


class M2MBlock(MessagePassing):
    """Mesh to Mesh Message Passing Block (following InteractionNetwork pattern).
    
    Updates both mesh nodes and mesh2mesh edges, consistent with original GNS.
    This is the core processor block.
    """
    def __init__(self, nnode_in: int, nnode_out: int, nedge_in: int, nedge_out: int, nmlp_layers: int, latent_dim: int):
        super().__init__(aggr='add')
        # Node MLP: combines aggregated edge messages with node features
        self.node_fn = nn.Sequential(*[
            build_mlp(nnode_in + nedge_out, [latent_dim for _ in range(nmlp_layers)], nnode_out),
            nn.LayerNorm(nnode_out)
        ])
        # Edge MLP: combines source and target mesh node features with edge features
        self.edge_fn = nn.Sequential(*[
            build_mlp(nnode_in + nnode_in + nedge_in, [latent_dim for _ in range(nmlp_layers)], nedge_out),
            nn.LayerNorm(nedge_out)
        ])
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor):
        """Forward pass for mesh2mesh message passing.
        
        Args:
            x: Mesh node features [num_mesh_nodes, nnode_in]
            edge_index: Mesh2mesh edge indices [2, num_edges]
            edge_features: Edge features [num_edges, nedge_in]
        """
        x_residual = x
        edge_features_residual = edge_features
        
        x, edge_features = self.propagate(edge_index=edge_index, x=x, edge_features=edge_features)
        
        return x + x_residual, edge_features + edge_features_residual
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        """Construct message from mesh node j to mesh node i."""
        edge_features = torch.cat([x_i, x_j, edge_features], dim=-1)
        edge_features = self.edge_fn(edge_features)
        return edge_features
    
    def update(self, x_updated: torch.Tensor, x: torch.Tensor, edge_features: torch.Tensor):
        """Update mesh node features using aggregated messages."""
        x_updated = torch.cat([x_updated, x], dim=-1)
        x_updated = self.node_fn(x_updated)
        return x_updated, edge_features


class M2GBlock(MessagePassing):
    """Mesh to Grid Message Passing Block (following InteractionNetwork pattern).
    
    Updates both grid nodes and mesh2grid edges, consistent with original GNS.
    This is the decoder block.
    """
    def __init__(self, nnode_in: int, nnode_out: int, nedge_in: int, nedge_out: int, nmlp_layers: int, latent_dim: int):
        super().__init__(aggr='add')
        # Node MLP: combines aggregated edge messages with node features
        self.node_fn = nn.Sequential(*[
            build_mlp(nnode_in + nedge_out, [latent_dim for _ in range(nmlp_layers)], nnode_out),
            nn.LayerNorm(nnode_out)
        ])
        # Edge MLP: combines source (mesh) and target (grid) node features with edge features
        self.edge_fn = nn.Sequential(*[
            build_mlp(nnode_in + nnode_in + nedge_in, [latent_dim for _ in range(nmlp_layers)], nedge_out),
            nn.LayerNorm(nedge_out)
        ])
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_features: torch.Tensor):
        """Forward pass for mesh2grid message passing.
        
        Args:
            x: Grid node features [num_grid_nodes, nnode_in]
            edge_index: Mesh2grid edge indices [2, num_edges] (source=mesh, target=grid)
            edge_features: Edge features [num_edges, nedge_in]
        """
        x_residual = x
        edge_features_residual = edge_features
        
        x, edge_features = self.propagate(edge_index=edge_index, x=x, edge_features=edge_features)
        
        return x + x_residual, edge_features + edge_features_residual
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        """Construct message from mesh node j to grid node i."""
        # x_i: grid node features, x_j: mesh node features
        edge_features = torch.cat([x_i, x_j, edge_features], dim=-1)
        edge_features = self.edge_fn(edge_features)
        return edge_features
    
    def update(self, x_updated: torch.Tensor, x: torch.Tensor, edge_features: torch.Tensor):
        """Update grid node features using aggregated messages."""
        x_updated = torch.cat([x_updated, x], dim=-1)
        x_updated = self.node_fn(x_updated)
        return x_updated, edge_features


class MultiScaleGNN(nn.Module):
    """Multi-Scale Graph Neural Network following Encode-Process-Decode pattern.
    
    Implements a multi-scale GNN that processes particle-based simulation data through
    a hierarchical graph structure with cross-scale message passing.
    
    Args:
        nnode_in_features: Number of input node features (e.g., position, velocity)
        nnode_out_features: Number of output node features (e.g., accelerations)
        nedge_in_features: Number of input edge features (e.g., relative displacement, distance)
        nedge_out_features: Number of output edge features (latent edge dimension)
        latent_dim: Latent dimension for all hidden layers and node embeddings
        nmessage_passing_steps: Number of mesh2mesh message passing steps
        nmlp_layers: Number of hidden layers in MLPs
        num_scales: Number of scales in the hierarchy (typically 3: grid + 2 mesh levels)
        share_weights_across_scales: Whether to share weights across scales (unused)
    """
    def __init__(self,
                 nnode_in_features: int,
                 nnode_out_features: int,
                 nedge_in_features: int,
                 nedge_out_features: int,
                 latent_dim: int,
                 nmessage_passing_steps: int,
                 nmlp_layers: int,
                 num_scales: int,
                 share_weights_across_scales: bool = False):
        super().__init__()
        self.num_scales = num_scales
        self.latent_dim = latent_dim
        self.nmessage_passing_steps = nmessage_passing_steps
        
        # 1. Encoder: Grid node and edge encoders
        self.grid_node_encoder = nn.Sequential(*[
            build_mlp(nnode_in_features, [latent_dim for _ in range(nmlp_layers)], latent_dim),
            nn.LayerNorm(latent_dim)
        ])
        
        # Edge encoders for each edge type
        self.g2m_edge_encoder = nn.Sequential(*[
            build_mlp(nedge_in_features, [latent_dim for _ in range(nmlp_layers)], nedge_out_features),
            nn.LayerNorm(nedge_out_features)
        ])
        self.m2m_edge_encoder = nn.Sequential(*[
            build_mlp(nedge_in_features, [latent_dim for _ in range(nmlp_layers)], nedge_out_features),
            nn.LayerNorm(nedge_out_features)
        ])
        self.m2g_edge_encoder = nn.Sequential(*[
            build_mlp(nedge_in_features, [latent_dim for _ in range(nmlp_layers)], nedge_out_features),
            nn.LayerNorm(nedge_out_features)
        ])
        
        # 2. Encoder: Grid2Mesh GNN block
        self.g2m_block = G2MBlock(latent_dim, latent_dim, 
                                  nedge_out_features, nedge_out_features, 
                                  nmlp_layers, latent_dim)
        
        # 3. Processor: Multiple rounds of Mesh2Mesh GNN
        self.m2m_blocks = nn.ModuleList([
            M2MBlock(latent_dim, latent_dim, nedge_out_features, nedge_out_features, nmlp_layers, latent_dim)
            for _ in range(nmessage_passing_steps)
        ])
        
        # 4. Decoder: Mesh2Grid GNN block
        self.m2g_block = M2GBlock(latent_dim, latent_dim, nedge_out_features, nedge_out_features, nmlp_layers, latent_dim)
        
        # 5. Prediction head on grid particles
        self.prediction_head = build_mlp(latent_dim, [latent_dim for _ in range(nmlp_layers)], nnode_out_features)
    
    def forward(self,
                x: torch.Tensor,  # Grid node features [num_grid_nodes, nnode_in_features]
                g2m_edge_index: torch.Tensor,
                g2m_edge_features: torch.Tensor,
                m2m_edge_index: torch.Tensor,
                m2m_edge_features: torch.Tensor,
                m2g_edge_index: torch.Tensor,
                m2g_edge_features: torch.Tensor,
                graph_hierarchy: Dict[int, Dict[str, Any]]) -> torch.Tensor:
        """Forward pass following Encode-Process-Decode pattern.
        
        Args:
            x: Grid node features
            g2m_edge_index: Grid2mesh edge indices [2, num_g2m_edges]
            g2m_edge_features: Grid2mesh edge features [num_g2m_edges, nedge_in_features]
            m2m_edge_index: Mesh2mesh edge indices [2, num_m2m_edges]
            m2m_edge_features: Mesh2mesh edge features [num_m2m_edges, nedge_in_features]
            m2g_edge_index: Mesh2grid edge indices [2, num_m2g_edges]
            m2g_edge_features: Mesh2grid edge features [num_m2g_edges, nedge_in_features]
            graph_hierarchy: Multi-scale graph hierarchy with sampling indices (unused here)
            
        Returns:
            Grid node outputs [num_grid_nodes, nnode_out_features]
        """
        # 1. ENCODER: Embed grid nodes and all edge types
        grid_latent = self.grid_node_encoder(x)  # [num_grid_nodes, latent_dim]
        g2m_edge_latent = self.g2m_edge_encoder(g2m_edge_features)  # [num_g2m_edges, nedge_out_features]
        m2m_edge_latent = self.m2m_edge_encoder(m2m_edge_features)  # [num_m2m_edges, nedge_out_features]
        m2g_edge_latent = self.m2g_edge_encoder(m2g_edge_features)  # [num_m2g_edges, nedge_out_features]
        
        # 2. ENCODER: Grid2Mesh message passing over full grid (targets are mesh indices in edge_index)
        grid_latent, g2m_edge_latent = self.g2m_block(
            grid_latent, g2m_edge_index, g2m_edge_latent
        )
        
        # 3. PROCESSOR: Mesh2Mesh message passing over full grid (edges connect mesh nodes)
        for m2m_block in self.m2m_blocks:
            grid_latent, m2m_edge_latent = m2m_block(
                grid_latent, m2m_edge_index, m2m_edge_latent
            )
        
        # 4. DECODER: Mesh2Grid message passing over full grid (sources are mesh indices)
        grid_latent, m2g_edge_latent = self.m2g_block(
            grid_latent, m2g_edge_index, m2g_edge_latent
        )
        
        # 5. Prediction head on all grid particles
        outputs = self.prediction_head(grid_latent)  # [num_grid_nodes, nnode_out_features]
        return outputs

