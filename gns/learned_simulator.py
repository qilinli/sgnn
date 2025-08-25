import torch
import torch.nn as nn
import numpy as np
from gns import graph_network
from torch_geometric.nn import radius_graph
from typing import Dict


class LearnedSimulator(nn.Module):
  """Learned simulator from https://arxiv.org/pdf/2002.09405.pdf."""

  def __init__(
          self,
          particle_dimensions: int,
          nnode_in: int,
          nedge_in: int,
          latent_dim: int,
          nmessage_passing_steps: int,
          nmlp_layers: int,
          mlp_hidden_dim: int,
          connectivity_radius: float,
          normalization_stats: Dict,
          nparticle_types: int,
          particle_type_embedding_size,
          device="cpu"):
    """Initializes the model.

    Args:
      particle_dimensions: Dimensionality of the problem.
      nnode_in: Number of node inputs.
      nedge_in: Number of edge inputs.
      latent_dim: Size of latent dimension (128)
      nmessage_passing_steps: Number of message passing steps.
      nmlp_layers: Number of hidden layers in the MLP (typically of size 2).
      connectivity_radius: Scalar with the radius of connectivity.
      normalization_stats: Dictionary with statistics with keys "acceleration"
        and "velocity", containing a named tuple for each with mean and std
        fields, matching the dimensionality of the problem.
      nparticle_types: Number of different particle types.
      particle_type_embedding_size: Embedding size for the particle type.
      device: Runtime device (cuda or cpu).

    """
    super(LearnedSimulator, self).__init__()
    self._connectivity_radius = connectivity_radius
    self._normalization_stats = normalization_stats
    self._nparticle_types = nparticle_types
    self._particle_dimensions = particle_dimensions

    # Particle type embedding has shape (9, 16)
    self._particle_type_embedding = nn.Embedding(
        nparticle_types, particle_type_embedding_size)

    # Initialize the EncodeProcessDecode
    self._encode_process_decode = graph_network.EncodeProcessDecode(
        nnode_in_features=nnode_in,
        nnode_out_features=particle_dimensions + 1, # qilin, with one auxiliary output
        nedge_in_features=nedge_in,
        latent_dim=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers,
        mlp_hidden_dim=mlp_hidden_dim)

    self._device = device

  def forward(self):
    """Forward hook runs on class instantiation"""
    pass

  def _compute_graph_connectivity(
          self,
          positions: torch.tensor,
          nparticles_per_example: torch.tensor,
          radius: float,
          add_self_edges: bool = True):
    """Generate graph edges to all particles within a threshold radius based on position proximity

    Args:
      positions: Particle positions with shape (nparticles, dim).
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      radius: Threshold to construct edges to all particles within the radius.
      add_self_edges: Boolean flag to include self edge (default: True)
    """
    # Validate inputs
    if len(positions.shape) != 2:
        raise ValueError(f"Expected 2D positions tensor, got shape {positions.shape}")
    
    if not isinstance(nparticles_per_example, torch.Tensor):
        nparticles_per_example = torch.tensor(nparticles_per_example)
    
    # Ensure nparticles_per_example is 1D
    if len(nparticles_per_example.shape) > 1:
        nparticles_per_example = nparticles_per_example.flatten()
    
    # Validate that total particles matches
    total_particles = nparticles_per_example.sum().item()
    if total_particles != positions.shape[0]:
        print(f"⚠️  Warning: Total particles mismatch: {total_particles} vs {positions.shape[0]}")
        print(f"   nparticles_per_example: {nparticles_per_example}")
        print(f"   positions shape: {positions.shape}")
    
    # Specify examples id for particles
    batch_ids = torch.cat(
        [torch.LongTensor([i for _ in range(n)])
         for i, n in enumerate(nparticles_per_example)]).to(self._device)
    
    # Debug information for graph construction
    if hasattr(self, '_debug_graph') and self._debug_graph:
        print(f"   Batch IDs shape: {batch_ids.shape}")
        print(f"   Batch IDs range: [{batch_ids.min()}, {batch_ids.max()}]")
        print(f"   Unique batch IDs: {torch.unique(batch_ids).tolist()}")
    
    # radius_graph accepts r < radius not r <= radius
    # A torch tensor list of source and target nodes with shape (2, nedges)
    edge_index = radius_graph(
        positions, r=radius, batch=batch_ids, loop=add_self_edges, max_num_neighbors=20)

    # The flow direction when using in combination with message passing is
    # "source_to_target"
    receivers = edge_index[0, :]
    senders = edge_index[1, :]

    return receivers, senders

  def _test_graph_connectivity(self, positions: torch.tensor, nparticles_per_example: torch.tensor, 
                              radius: float, num_test_nodes: int = 5):
    """
    Test graph connectivity by examining random nodes and their neighbors.
    
    Note: This method counts both incoming and outgoing edges for each particle.
    Since radius_graph creates bidirectional edges, the total connections per particle
    will be approximately 2 × (number of neighbors within radius).
    
    Args:
      positions: Particle positions with shape (nparticles, dim).
      nparticles_per_example: Number of particles per example.
      radius: Connectivity radius used for graph construction.
      num_test_nodes: Number of random nodes to test.
    """
    print(f"\n🔍 Graph Connectivity Test (radius={radius:.3f})")
    print("=" * 50)
    
    # Validate input tensor dimensions
    if len(positions.shape) != 2:
        print(f"⚠️  Warning: Expected 2D positions tensor, got shape {positions.shape}")
        print(f"   Reshaping to 2D for graph testing...")
        if len(positions.shape) == 3:
            positions = positions[:, -1, :]  # Take last timestep
        else:
            print(f"   Error: Cannot handle positions with {len(positions.shape)} dimensions")
            return
    
    # Get graph connectivity
    receivers, senders = self._compute_graph_connectivity(
        positions, nparticles_per_example, radius, add_self_edges=True)
    
    # Convert to edge list for easier analysis
    edges = torch.stack([senders, receivers], dim=0)
    
    # Get unique nodes and their edge counts
    unique_nodes = torch.unique(positions, dim=0)
    nparticles = positions.shape[0]
    
    print(f"Total particles: {nparticles}")
    print(f"Total edges: {edges.shape[1]}")
    print(f"Average edges per particle: {edges.shape[1] / nparticles:.2f}")
    print(f"Note: Each edge is bidirectional, so connections per particle ≈ 2 × neighbors within radius")
    
    # Test random nodes
    import random
    test_indices = random.sample(range(nparticles), min(num_test_nodes, nparticles))
    
    for i, node_idx in enumerate(test_indices):
        # Find all edges where this node is the receiver
        incoming_edges = (receivers == node_idx).sum().item()
        
        # Find all edges where this node is the sender
        outgoing_edges = (senders == node_idx).sum().item()
        
        # Total connections for this node
        total_connections = incoming_edges + outgoing_edges
        
        # Get node position
        node_pos = positions[node_idx]
        
        # Count neighbors within radius (including self if self_edges=True)
        distances = torch.norm(positions - node_pos, dim=1)
        neighbors_within_radius = (distances <= radius).sum().item()
        
        print(f"Node {i+1} (idx={node_idx}):")
        print(f"  Position: [{node_pos[0]:.3f}, {node_pos[1]:.3f}]")
        print(f"  Incoming edges: {incoming_edges}")
        print(f"  Outgoing edges: {outgoing_edges}")
        print(f"  Total connections: {total_connections}")
        print(f"  Particles within radius: {neighbors_within_radius}")
        print(f"  Expected max neighbors: 20 (max_num_neighbors limit)")
        
        # Check if connections match expectations
        # Note: total_connections includes both incoming and outgoing edges
        # For bidirectional graphs, this will be approximately 2x the number of neighbors
        expected_connections = neighbors_within_radius * 2  # Bidirectional edges
        if abs(total_connections - expected_connections) <= 1:  # Allow for small differences due to self-edges
            print(f"  ✅ Connections match expectation: {total_connections} ≈ {expected_connections} (2 × {neighbors_within_radius} neighbors)")
        else:
            print(f"  ⚠️  Unexpected connection count: {total_connections} vs expected ~{expected_connections}")
        
        print()
    
    # Analyze edge distribution
    edge_counts = torch.zeros(nparticles, dtype=torch.long)
    for i in range(nparticles):
        edge_counts[i] = (receivers == i).sum() + (senders == i).sum()
    
    print(f"Edge distribution statistics:")
    print(f"  Min connections: {edge_counts.min().item()}")
    print(f"  Max connections: {edge_counts.max().item()}")
    print(f"  Mean connections: {edge_counts.float().mean().item():.2f}")
    print(f"  Std connections: {edge_counts.float().std().item():.2f}")
    print(f"  Expected range: ~2 to ~40 (2 × 1 to 2 × 20 neighbors, considering max_num_neighbors=20)")
    
    # Check for isolated nodes
    isolated_nodes = (edge_counts == 0).sum().item()
    if isolated_nodes > 0:
        print(f"  ⚠️  Warning: {isolated_nodes} isolated nodes found!")
    else:
        print(f"  ✅ No isolated nodes found")
    
    print("=" * 50)

  def _encoder_preprocessor(
          self,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor):
    """Extracts important features from the position sequence. Returns a tuple
    of node_features (nparticles, 12), edge_index (nparticles, nparticles), and
    edge_features (nedges, 3).

    Args:
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).
    """
    # Ensure input tensor is contiguous for reliable operations
    position_sequence = position_sequence.contiguous()
    
    # Validate input dimensions
    if len(position_sequence.shape) != 3:
        raise ValueError(f"Expected position_sequence to have 3 dimensions, got {len(position_sequence.shape)}")
    if position_sequence.shape[1] < 2:
        raise ValueError(f"Expected at least 2 timesteps, got {position_sequence.shape[1]}")
    
    nparticles = position_sequence.shape[0]
    most_recent_position = position_sequence[:, -1].contiguous()  # (n_nodes, 2)
    velocity_sequence = time_diff(position_sequence).contiguous()

    # Get connectivity of the graph with shape of (nparticles, 2)
    senders, receivers = self._compute_graph_connectivity(
        most_recent_position, nparticles_per_example, self._connectivity_radius)
    
    # Test graph connectivity if debug flag is enabled
    if hasattr(self, '_debug_graph') and self._debug_graph:
        self._test_graph_connectivity(
            most_recent_position, nparticles_per_example, self._connectivity_radius)
            
    node_features = []

    # Normalized velocity sequence, merging spatial an time axis.
    velocity_stats = self._normalization_stats["velocity"]
    normalized_velocity_sequence = (
        (velocity_sequence - velocity_stats['mean']) / velocity_stats['std']
    ).contiguous()
    flat_velocity_sequence = normalized_velocity_sequence.reshape(
        nparticles, -1)
    node_features.append(flat_velocity_sequence)

    # Wall distance feature for left wall at x = -2
    # Distance to wall: x - (-2) = x + 2, clipped to connectivity radius
    wall_distances = torch.clamp(most_recent_position[:, 0:1] + 2.0, 
                                 min=0.0, max=self._connectivity_radius)
    node_features.append(wall_distances)
    
    # Particle type
    if self._nparticle_types > 1:
      particle_type_embeddings = self._particle_type_embedding(
          particle_types)
      node_features.append(particle_type_embeddings)
    # Final node_features shape (nparticles, 11) for 2D
    # (input_sequence_length - 1) * dim + 1 wall distance
    
    # Collect edge features.
    edge_features = []

    # Relative displacement and distances normalized to radius
    # with shape (nedges, 2)
    normalized_relative_displacements = (
        most_recent_position[senders, :] -
        most_recent_position[receivers, :]
    ) / self._connectivity_radius
    
    # Add relative displacement between two particles as an edge feature
    # with shape (nedges, ndim)
    edge_features.append(normalized_relative_displacements)

    # Add relative distance between 2 particles with shape (nparticles, 1)
    # Edge features has a final shape of (nedges, ndim + 1)
    normalized_relative_distances = torch.norm(
        normalized_relative_displacements, dim=-1, keepdim=True)
    edge_features.append(normalized_relative_distances)

    return (torch.cat(node_features, dim=-1),
            torch.stack([senders, receivers]),
            torch.cat(edge_features, dim=-1))

  def test_graph_connectivity_once(self, positions: torch.tensor, nparticles_per_example: torch.tensor):
    """
    Test graph connectivity once for debugging purposes.
    This method can be called independently to test graph construction.
    
    Note: This method counts both incoming and outgoing edges for each particle.
    Since radius_graph creates bidirectional edges, the total connections per particle
    will be approximately 2 × (number of neighbors within radius).
    
    Args:
      positions: Particle positions with shape (nparticles, dim).
      nparticles_per_example: Number of particles per example.
    """
    print(f"\n🔍 One-time Graph Connectivity Test")
    print("=" * 50)
    
    # Validate input tensor dimensions
    if len(positions.shape) != 2:
        print(f"⚠️  Warning: Expected 2D positions tensor, got shape {positions.shape}")
        print(f"   Reshaping to 2D for graph testing...")
        if len(positions.shape) == 3:
            positions = positions[:, -1, :]  # Take last timestep
            print(f"   Reshaped to: {positions.shape}")
        else:
            print(f"   Error: Cannot handle positions with {len(positions.shape)} dimensions")
            return
    
    # Get graph connectivity
    receivers, senders = self._compute_graph_connectivity(
        positions, nparticles_per_example, self._connectivity_radius, add_self_edges=True)
    
    # Basic statistics
    nparticles = positions.shape[0]
    total_edges = receivers.shape[0]
    
    print(f"Total particles: {nparticles}")
    print(f"Total edges: {total_edges}")
    print(f"Average edges per particle: {total_edges / nparticles:.2f}")
    print(f"Connectivity radius: {self._connectivity_radius}")
    print(f"Note: Each edge is bidirectional, so connections per particle ≈ 2 × neighbors within radius")
    
    # Check edge distribution
    edge_counts = torch.zeros(nparticles, dtype=torch.long)
    for i in range(nparticles):
        edge_counts[i] = (receivers == i).sum() + (senders == i).sum()
    
    print(f"\nEdge distribution:")
    print(f"  Min connections: {edge_counts.min().item()}")
    print(f"  Max connections: {edge_counts.max().item()}")
    print(f"  Mean connections: {edge_counts.float().mean().item():.2f}")
    print(f"  Std connections: {edge_counts.float().std().item():.2f}")
    print(f"  Expected range: ~2 to ~40 (2 × 1 to 2 × 20 neighbors, considering max_num_neighbors=20)")
    
    # Check for isolated nodes
    isolated_nodes = (edge_counts == 0).sum().item()
    if isolated_nodes > 0:
        print(f"  ⚠️  Warning: {isolated_nodes} isolated nodes found!")
    else:
        print(f"  ✅ No isolated nodes found")
    
    print("=" * 50)


  def _decoder_postprocessor(
          self,
          normalized_acceleration: torch.tensor,
          position_sequence: torch.tensor) -> torch.tensor:
    """ Compute new position based on acceleration and current position.
    The model produces the output in normalized space so we apply inverse
    normalization.

    Args:
      normalized_acceleration: Normalized acceleration (nparticles, dim).
      position_sequence: Position sequence of shape (nparticles, dim).

    Returns:
      torch.tensor: New position of the particles.

    """
    # Extract real acceleration values from normalized values
    acceleration_stats = self._normalization_stats["acceleration"]
    acceleration = (
        normalized_acceleration * acceleration_stats['std']
    ) + acceleration_stats['mean']

    # Use an Euler integrator to go from acceleration to position, assuming
    # a dt=1 corresponding to the size of the finite difference.
    most_recent_position = position_sequence[:, -1]
    most_recent_velocity = most_recent_position - position_sequence[:, -2]

    # TODO: Fix dt
    new_velocity = most_recent_velocity + acceleration  # * dt = 1
    new_position = most_recent_position + new_velocity  # * dt = 1
    return new_position

  def predict_positions(
          self,
          current_positions: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor) -> torch.tensor:
    """Predict position based on acceleration.

    Args:
      current_positions: Current particle positions (nparticles, dim).
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).

    Returns:
      next_positions (torch.tensor): Next position of particles.
    """
    node_features, edge_index, edge_features = self._encoder_preprocessor(
        current_positions, nparticles_per_example, particle_types)
    pred = self._encode_process_decode(
        node_features, edge_index, edge_features)
    predicted_normalized_acceleration = pred[:, :self._particle_dimensions] # qilin, the first 2 or 3 dims are accelrations
    predicted_strain = pred[:, -1]  # the last dimension are strains
    next_positions = self._decoder_postprocessor(
        predicted_normalized_acceleration, current_positions)
    
    return next_positions, predicted_strain

  def predict_accelerations(
          self,
          next_positions: torch.tensor,
          position_sequence_noise: torch.tensor,
          position_sequence: torch.tensor,
          nparticles_per_example: torch.tensor,
          particle_types: torch.tensor):
    """Produces normalized and predicted acceleration targets.

    Args:
      next_positions: Tensor of shape (nparticles_in_batch, dim) with the
        positions the model should output given the inputs.
      position_sequence_noise: Tensor of the same shape as `position_sequence`
        with the noise to apply to each particle.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.
      nparticles_per_example: Number of particles per example. Default is 2
        examples per batch.
      particle_types: Particle types with shape (nparticles).

    Returns:
      Tensors of shape (nparticles_in_batch, dim) with the predicted and target
        normalized accelerations.

    """

    # Add noise to the input position sequence.
    noisy_position_sequence = position_sequence + position_sequence_noise

    # Perform the forward pass with the noisy position sequence.
    node_features, edge_index, edge_features = self._encoder_preprocessor(
        noisy_position_sequence, nparticles_per_example, particle_types)
    pred = self._encode_process_decode(
        node_features, edge_index, edge_features)
    predicted_normalized_acceleration = pred[:, :self._particle_dimensions] # qilin, the first 2 or 3 dims are accelrations
    predicted_strain = pred[:, -1]  # the rest dimension are strains

    # Calculate the target acceleration, using an `adjusted_next_position `that
    # is shifted by the noise in the last input position.
    next_position_adjusted = next_positions + position_sequence_noise[:, -1]
    target_normalized_acceleration = self._inverse_decoder_postprocessor(
        next_position_adjusted, noisy_position_sequence)
    # As a result the inverted Euler update in the `_inverse_decoder` produces:
    # * A target acceleration that does not explicitly correct for the noise in
    #   the input positions, as the `next_position_adjusted` is different
    #   from the true `next_position`.
    # * A target acceleration that exactly corrects noise in the input velocity
    #   since the target next velocity calculated by the inverse Euler update
    #   as `next_position_adjusted - noisy_position_sequence[:,-1]`
    #   matches the ground truth next velocity (noise cancels out).

    return predicted_normalized_acceleration, target_normalized_acceleration, predicted_strain

  def _inverse_decoder_postprocessor(
          self,
          next_position: torch.tensor,
          position_sequence: torch.tensor):
    """Inverse of `_decoder_postprocessor`.

    Args:
      next_position: Tensor of shape (nparticles_in_batch, dim) with the
        positions the model should output given the inputs.
      position_sequence: A sequence of particle positions. Shape is
        (nparticles, 6, dim). Includes current + last 5 positions.

    Returns:
      normalized_acceleration (torch.tensor): Normalized acceleration.

    """
    previous_position = position_sequence[:, -1]
    previous_velocity = previous_position - position_sequence[:, -2]
    next_velocity = next_position - previous_position
    acceleration = next_velocity - previous_velocity

    acceleration_stats = self._normalization_stats["acceleration"]
    normalized_acceleration = (
        acceleration - acceleration_stats['mean']) / acceleration_stats['std']
    return normalized_acceleration

  def save(
          self,
          path: str = 'model.pt'):
    """Save model state

    Args:
      path: Model path
    """
    torch.save(self.state_dict(), path)

  def load(
          self,
          path: str):
    """Load model state from file

    Args:
      path: Model path
    """
    self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))


def time_diff(
        position_sequence: torch.tensor) -> torch.tensor:
  """Finite difference between two input position sequence

  Args:
    position_sequence: Input position sequence & shape(nparticles, 6 steps, dim)

  Returns:
    torch.tensor: Velocity sequence
  """
  return (position_sequence[:, 1:] - position_sequence[:, :-1]).contiguous()
