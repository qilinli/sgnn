"""
Multi-Scale Simulator for Particle-Based Simulations.

This module implements a multi-scale simulator that extends the LearnedSimulator
pattern to work with multi-scale graph structures. It integrates with MultiScaleGraph
to build hierarchical graph connectivity and uses MultiScaleGNN for predictions.

The architecture follows the same pattern as LearnedSimulator but operates on
multi-scale graphs with cross-scale message passing.
"""

import torch
import torch.nn as nn
import numpy as np
from gns.multi_scale.multi_scale_gnn import MultiScaleGNN
from gns.multi_scale.multi_scale_graph import MultiScaleGraph, MultiScaleConfig
from typing import Dict, Tuple, Optional, Any


class MultiScaleSimulator(nn.Module):
    """Multi-scale simulator for particle-based simulations.
    
    Extends the LearnedSimulator pattern to work with multi-scale graph structures.
    Uses MultiScaleGraph for hierarchical connectivity and MultiScaleGNN for predictions.
    
    Args:
        kinematic_dimensions: Dimensionality of the problem (2D or 3D)
        nnode_in: Number of input node features (computed from velocity sequence + wall distances + particle types)
        nedge_in: Number of input edge features (relative displacement + distance = 3 for 2D)
        nedge_out: Number of output edge features (latent edge dimension)
        latent_dim: Size of latent dimension for all hidden layers
        nmessage_passing_steps: Number of mesh2mesh message passing steps
        nmlp_layers: Number of hidden layers in MLPs
        normalization_stats: Dictionary with statistics for acceleration and velocity normalization
        nparticle_types: Number of different particle types
        particle_type_embedding_size: Embedding size for particle types
        num_scales: Number of scales in the hierarchy (typically 3: grid + 2 mesh levels)
        window_size: Sampling window size for mesh levels
        radius_multiplier: Multiplier for all connectivity radius calculations
        device: Runtime device (cuda or cpu)
        
    Note:
        grid_spacing (0.5) is fixed in MultiScaleConfig and cannot be configured.
        num_scales, window_size, and radius_multiplier are configurable.
    """
    
    def __init__(self,
                 kinematic_dimensions: int,
                 nnode_in: int,
                 nedge_in: int,
                 nedge_out: int,
                 latent_dim: int,
                 nmessage_passing_steps: int,
                 nmlp_layers: int,
                 normalization_stats: Dict,
                 nparticle_types: int,
                 particle_type_embedding_size: int,
                 num_scales: int = 3,
                 window_size: int = 3,
                 radius_multiplier: float = 2.0,
                 device: str = "cpu"):
        super(MultiScaleSimulator, self).__init__()
        
        # Store configuration parameters
        self._kinematic_dimensions = kinematic_dimensions
        self._normalization_stats = normalization_stats
        self._nparticle_types = nparticle_types
        self._num_scales = num_scales
        self._window_size = window_size
        self._device = device
        
        # Particle type embedding (same as LearnedSimulator)
        self._particle_type_embedding = nn.Embedding(
            nparticle_types, particle_type_embedding_size)
        
        # Multi-scale graph configuration
        # num_scales, window_size, and radius_multiplier are configurable
        self._multi_scale_config = MultiScaleConfig(
            num_scales=num_scales,
            window_size=window_size,
            radius_multiplier=radius_multiplier
        )
        
        # Multi-scale GNN (replaces EncodeProcessDecode)
        self._multi_scale_gnn = MultiScaleGNN(
            nnode_in_features=nnode_in,
            nnode_out_features=kinematic_dimensions + 1,  # accelerations + auxiliary output (strain or stress)
            nedge_in_features=nedge_in,
            nedge_out_features=nedge_out,
            latent_dim=latent_dim,
            nmessage_passing_steps=nmessage_passing_steps,
            nmlp_layers=nmlp_layers,
            num_scales=num_scales
        )
        
        # Static multi-scale graph data (built during data processing)
        self._static_graph_data = None
    
    def forward(self):
        """Forward hook runs on class instantiation"""
        pass
    
    def set_static_graph(self, graph_data: Dict[str, Any]):
        """Set the static multi-scale graph for this simulation case.
        
        Args:
            graph_data: Dictionary containing:
                - 'graph_hierarchy': Dict with scale data (sampling_indices, spacing, num_particles)
                - 'grid2mesh_edges': Edge indices for grid to mesh connections
                - 'mesh2mesh_edges': Edge indices for mesh to mesh connections  
                - 'mesh2grid_edges': Edge indices for mesh to grid connections
        """
        self._static_graph_data = graph_data
        
    def _validate_static_graph(self):
        """Validate that static graph data is available and properly formatted."""
        if self._static_graph_data is None:
            raise ValueError("Static graph data not set. Call set_static_graph() first.")
        
        required_keys = ['graph_hierarchy', 'grid2mesh_edges', 'mesh2mesh_edges', 'mesh2grid_edges']
        for key in required_keys:
            if key not in self._static_graph_data:
                raise ValueError(f"Missing required graph data key: {key}")
    
    def _encoder_preprocessor(self,
                            position_sequence: torch.Tensor,
                            nparticles_per_example: torch.Tensor,
                            particle_types: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Extract features using static multi-scale graph structure.
        
        Args:
            position_sequence: A sequence of particle positions. Shape is (nparticles, 6, dim)
            nparticles_per_example: Number of particles per example
            particle_types: Particle types with shape (nparticles)
            
        Returns:
            Tuple containing:
            - node_features: Node features (nparticles, nnode_in)
            - edge_indices: Dictionary with g2m, m2m, m2g edge indices (from static graph)
            - edge_features: Dictionary with g2m, m2m, m2g edge features
        """
        # Validate static graph is available
        self._validate_static_graph()
        
        # Ensure input tensor is contiguous for reliable operations
        position_sequence = position_sequence.contiguous()
        
        nparticles = position_sequence.shape[0]
        most_recent_position = position_sequence[:, -1].contiguous()  # (n_nodes, 2)
        velocity_sequence = self._time_diff(position_sequence).contiguous()
        
        # Build node features
        node_features = self._build_node_features(
            velocity_sequence, most_recent_position, particle_types, nparticles)
        
        # Get static graph structure and move to correct device
        g2m_edge_index = self._static_graph_data['grid2mesh_edges'].to(node_features.device)
        m2m_edge_index = self._static_graph_data['mesh2mesh_edges'].to(node_features.device)
        m2g_edge_index = self._static_graph_data['mesh2grid_edges'].to(node_features.device)
        
        # Build edge features for each edge type
        edge_indices = {
            'g2m': g2m_edge_index,
            'm2m': m2m_edge_index,
            'm2g': m2g_edge_index
        }
        
        edge_features = self._build_edge_features(
            g2m_edge_index, m2m_edge_index, m2g_edge_index, most_recent_position)
        
        return node_features, edge_indices, edge_features
    
    def _build_node_features(self, velocity_sequence, most_recent_position, particle_types, nparticles):
        """Build node features from velocity sequence, positions, and particle types."""
        node_features = []
        
        # Normalized velocity sequence, merging spatial and time axis
        velocity_stats = self._normalization_stats["velocity"]
        normalized_velocity_sequence = (
            (velocity_sequence - velocity_stats['mean']) / velocity_stats['std']
        ).contiguous()
        flat_velocity_sequence = normalized_velocity_sequence.reshape(nparticles, -1)
        node_features.append(flat_velocity_sequence)
        
        # Wall distance feature: distance to left wall at x = -2
        # This helps the GNN understand particle proximity to boundaries
        # Distance = 0 means touching wall, distance = 1.0 means far from wall
        # Distance in between means interaction with the wall
        # Use the actual grid radius for normalization (grid_spacing * radius_multiplier)
        grid_radius = self._multi_scale_config.grid_spacing * self._multi_scale_config.radius_multiplier
        wall_distances = torch.clamp(most_recent_position[:, 0:1] + 2.0, 
                                     min=0.0, max=grid_radius) / grid_radius
        node_features.append(wall_distances)
        
        # Particle type embeddings
        if self._nparticle_types > 1:
            particle_type_embeddings = self._particle_type_embedding(particle_types)
            node_features.append(particle_type_embeddings)
        
        # Combine node features
        return torch.cat(node_features, dim=-1)
    
    def _build_edge_features(self, g2m_edge_index, m2m_edge_index, m2g_edge_index, most_recent_position):
        """Build edge features for all edge types using appropriate radius normalization."""
        edge_features = {}
        
        # Get radii for different edge types
        graph_hierarchy = self._static_graph_data['graph_hierarchy']
        
        # Grid radius for g2m and m2g edges
        grid_radius = self._multi_scale_config.grid_spacing * self._multi_scale_config.radius_multiplier
        
        # Coarsest mesh radius for m2m edges (last scale)
        coarsest_scale = self._multi_scale_config.num_scales - 1
        if coarsest_scale in graph_hierarchy:
            coarsest_spacing = graph_hierarchy[coarsest_scale]['spacing']
            coarsest_radius = coarsest_spacing * self._multi_scale_config.radius_multiplier
        else:
            coarsest_radius = grid_radius  # Fallback to grid radius
        
        # Helper function to build edge features for a single edge type
        def build_single_edge_features(edge_index, radius):
            if edge_index.shape[1] == 0:
                return torch.empty((0, 3), device=most_recent_position.device)
            
            senders, receivers = edge_index[0], edge_index[1]
            relative_displacements = (
                most_recent_position[senders, :] - most_recent_position[receivers, :]
            ) / radius
            relative_distances = torch.norm(relative_displacements, dim=-1, keepdim=True)
            return torch.cat([relative_displacements, relative_distances], dim=-1)
        
        # Build all edge features with appropriate radius normalization
        edge_features['g2m'] = build_single_edge_features(g2m_edge_index, grid_radius)
        edge_features['m2m'] = build_single_edge_features(m2m_edge_index, coarsest_radius)
        edge_features['m2g'] = build_single_edge_features(m2g_edge_index, grid_radius)
        
        return edge_features
    

    def _decoder_postprocessor(self,
                              normalized_acceleration: torch.Tensor,
                              position_sequence: torch.Tensor) -> torch.Tensor:
        """Compute new position based on acceleration and current position.
        
        Args:
            normalized_acceleration: Normalized acceleration (nparticles, dim)
            position_sequence: Position sequence of shape (nparticles, sequence_length, dim)
            
        Returns:
            torch.Tensor: New position of the particles
        """
        # Extract real acceleration values from normalized values
        acceleration_stats = self._normalization_stats["acceleration"]
        acceleration = (
            normalized_acceleration * acceleration_stats['std']
        ) + acceleration_stats['mean']

        # Use Euler integrator to go from acceleration to position
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = most_recent_position - position_sequence[:, -2]

        new_velocity = most_recent_velocity + acceleration
        new_position = most_recent_position + new_velocity
        return new_position
    
    def predict_positions(self,
                         current_positions: torch.Tensor,
                         nparticles_per_example: torch.Tensor,
                         particle_types: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict next positions using multi-scale GNN.
        
        Args:
            current_positions: Current particle positions (nparticles, 6, dim)
            nparticles_per_example: Number of particles per example
            particle_types: Particle types with shape (nparticles)
            
        Returns:
            Tuple of (next_positions, predicted_strain):
            - next_positions: Next position of particles
            - predicted_strain: Predicted strain values
        """
        # Extract features using static multi-scale graph
        node_features, edge_indices, edge_features = self._encoder_preprocessor(
            current_positions, nparticles_per_example, particle_types)
        
        # Get graph hierarchy for MultiScaleGNN
        graph_hierarchy = self._static_graph_data['graph_hierarchy']
        
        # Call MultiScaleGNN
        pred = self._multi_scale_gnn(
            node_features,
            edge_indices['g2m'],
            edge_features['g2m'],
            edge_indices['m2m'],
            edge_features['m2m'],
            edge_indices['m2g'],
            edge_features['m2g'],
            graph_hierarchy
        )
        
        # Extract predictions
        predicted_normalized_acceleration = pred[:, :self._kinematic_dimensions]
        predicted_strain = pred[:, -1]  # Last dimension is strain/stress
        
        # Convert to positions using physics integration
        next_positions = self._decoder_postprocessor(
            predicted_normalized_acceleration, current_positions)
        
        return next_positions, predicted_strain
    
    def predict_accelerations(self,
                             next_positions: torch.Tensor,
                             position_sequence_noise: torch.Tensor,
                             position_sequence: torch.Tensor,
                             nparticles_per_example: torch.Tensor,
                             particle_types: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Produce normalized and predicted acceleration targets for training."""
        # Add noise to the input position sequence
        noisy_position_sequence = position_sequence + position_sequence_noise

        # Perform forward pass with noisy position sequence
        node_features, edge_indices, edge_features = self._encoder_preprocessor(
            noisy_position_sequence, nparticles_per_example, particle_types)
        
        graph_hierarchy = self._static_graph_data['graph_hierarchy']
        pred = self._multi_scale_gnn(
            node_features,
            edge_indices['g2m'],
            edge_features['g2m'],
            edge_indices['m2m'],
            edge_features['m2m'],
            edge_indices['m2g'],
            edge_features['m2g'],
            graph_hierarchy
        )
        
        predicted_normalized_acceleration = pred[:, :self._kinematic_dimensions]
        predicted_strain = pred[:, -1]

        # Calculate target acceleration
        next_position_adjusted = next_positions + position_sequence_noise[:, -1]
        target_normalized_acceleration = self._inverse_decoder_postprocessor(
            next_position_adjusted, noisy_position_sequence)

        return predicted_normalized_acceleration, target_normalized_acceleration, predicted_strain
    
    def _inverse_decoder_postprocessor(self,
                                     next_position: torch.Tensor,
                                     position_sequence: torch.Tensor) -> torch.Tensor:
        """Inverse of _decoder_postprocessor for training."""
        previous_position = position_sequence[:, -1]
        previous_velocity = previous_position - position_sequence[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity

        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration = (
            acceleration - acceleration_stats['mean']) / acceleration_stats['std']
        return normalized_acceleration
    
    def save(self, path: str = 'multi_scale_model.pt'):
        """Save model state
        
        Args:
            path: Model path
        """
        torch.save(self.state_dict(), path)
    
    def load(self, path: str):
        """Load model state from file
        
        Args:
            path: Model path
        """
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    def get_static_graph_data(self) -> Optional[Dict[str, Any]]:
        """Get the current static graph data (if set)
        
        Returns:
            Dictionary with graph data or None if not set
        """
        return self._static_graph_data
    
    def _time_diff(self, position_sequence: torch.Tensor) -> torch.Tensor:
        """Finite difference between two input position sequence"""
        return (position_sequence[:, 1:] - position_sequence[:, :-1]).contiguous()
