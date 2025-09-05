"""
Multi-scale graph hierarchy for multi-scale GNN.

This module creates hierarchical structures where each level is a subset
of the previous level, using efficient grid-based edge creation.
Scale 0: base grid, Scale 1,2: sampled mesh levels.
"""

from typing import Dict, Tuple, Any
import torch
from torch_geometric.nn import radius_graph


class MultiScaleConfig:
    """Configuration for multi-scale mesh parameters."""
    
    def __init__(self, 
                 num_scales: int = 3, # total scales: 0=grid, 1=mesh1, 2=mesh2
                 window_size: int = 3,
                 radius_multiplier: float = 2.0):
        """
        Initialize multi-scale configuration.
        
        Args:       
            num_scales: Total number of scales (0=grid, 1=mesh1, 2=mesh2, etc.)
            window_size: Sampling window size for coarser levels
            radius_multiplier: Multiplier for all connectivity radius calculations
        """
        self.num_scales = num_scales    # total scales: 0=grid, 1=mesh1, 2=mesh2
        self.window_size = window_size  # sampling window size for coarser scales
        self.grid_spacing = 0.5         # mm - original particle grid spacing
        self.radius_multiplier = radius_multiplier  # multiplier for all connectivity radius calculations
        self.max_neighbors = 20         # maximum number of neighbors for all radius graphs

    
class MultiScaleGraph:
    def __init__(self, config: MultiScaleConfig):
        self.config = config
        
        # Core data storage
        self.grid_positions = None          # Original grid positions (N, 2)
        self.graph_hierarchy = {}           # scale data for each scale (scale 0=grid, scale 1,2=mesh)
        
    def create_all_edges(self, grid_positions: torch.Tensor) -> Dict[str, Any]:
        """
        Create all edge types (grid2mesh, mesh2mesh, mesh2grid) in one unified call.
        
        This is the main interface method that creates the complete multi-scale
        graph structure including all edge types with unified global grid indices.
        
        Hierarchy:
        - Scale 0: Base grid (all particles)
        - Scale 1: First mesh level (sampled from grid)
        - Scale 2: Second mesh level (sampled from scale 1)
        
        Args:
            grid_positions: (N, 2) particle positions at finest level (grid)
            
        Returns:
            Dictionary containing all edge types and connectivity information
        """
        # 1. Build hierarchy if not already built
        if not self.graph_hierarchy:
            self.build_hierarchy(grid_positions)
        
        # 2. Create grid-mesh connectivity (grid2mesh and mesh2grid edges)
        grid2mesh_edge_index, mesh2grid_edge_index = self._create_grid_mesh_connectivity(grid_positions)
        

        # 3. Create mesh2mesh edges for actual mesh scales (scale 1, 2, ...)
        # Note: scale 0 is the grid, so we start from scale 1
        all_mesh2mesh_edges = []
        for scale in range(1, self.config.num_scales):
            scale_edges = self._create_mesh2mesh_edges(scale)
            if scale_edges.shape[1] > 0:
                all_mesh2mesh_edges.append(scale_edges)
        
        # Combine all mesh2mesh edges from all levels
        if all_mesh2mesh_edges:
            mesh2mesh_edge_index = torch.cat(all_mesh2mesh_edges, dim=1)
        else:
            mesh2mesh_edge_index = torch.empty((2, 0), dtype=torch.long, device=grid_positions.device)
        
        # 4. Return unified structure
        return {
            'graph_hierarchy': self.graph_hierarchy,
            
            'grid2mesh_edges': grid2mesh_edge_index,
            'mesh2mesh_edges': mesh2mesh_edge_index,
            'mesh2grid_edges': mesh2grid_edge_index  
        }
    
    def build_hierarchy(self, 
                       grid_positions: torch.Tensor) -> Dict[str, Any]:
        """
        Build the complete multi-scale hierarchy from grid to coarser meshes.
        
        Args:
            grid_positions: (N, 2) particle positions at finest scale
            
        Returns:
            Complete hierarchy information for all scales (scale 0=grid, scale 1,2=mesh)
        """
        # Store grid data
        self.grid_positions = grid_positions
        
        # Handle scale 0 (original grid) separately
        self.graph_hierarchy[0] = {
            'sampling_indices': torch.arange(len(grid_positions), dtype=torch.long),  # All grid indices
            'spacing': self.config.grid_spacing,
            'num_particles': len(grid_positions)
        }
        
        # Build coarser mesh scales (starting from scale 1)
        current_positions = grid_positions
        current_spacing = self.config.grid_spacing
        
        for scale in range(1, self.config.num_scales):
            # Sample particles to create coarser scale
            coarser_positions, coarser_spacing, global_indices = self._sample_coarser_scale(
                current_positions, current_spacing, scale)
            
            # Store mesh level data
            self.graph_hierarchy[scale] = {
                'sampling_indices': global_indices,  # Global grid indices
                'spacing': coarser_spacing,
                'num_particles': len(coarser_positions)
            }
            
            # Update for next iteration
            current_positions = coarser_positions
            current_spacing = coarser_spacing
        
        return self.graph_hierarchy
    
    def _sample_coarser_scale(self, 
                                current_positions: torch.Tensor, 
                                current_spacing: float, 
                                scale: int) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Create a coarse level using geometric sampling based on spatial coordinates.
        
        For 2D Taylor impact, we sample every window_size-th particle in both x and y directions
        to maintain the grid structure and ensure proper spatial coverage.
        
        Args:
            current_positions: Positions from current level (N, 2) with [x, y] coordinates
            current_spacing: Spacing from current level
            scale: Current scale being created (1, 2, ...)
            
        Returns:
            Tuple of (coarse_positions, coarse_spacing, sampling_indices)
        """
        # Calculate new spacing
        new_spacing = current_spacing * self.config.window_size
        
        # Geometric sampling: sample every window_size-th particle in both x and y directions
        # This maintains the grid structure and spatial coverage
        
        # Get x and y coordinates
        x_coords = current_positions[:, 0]
        y_coords = current_positions[:, 1]
        
        # Find unique x and y values (sorted)
        unique_x = torch.sort(torch.unique(x_coords))[0]
        unique_y = torch.sort(torch.unique(y_coords))[0]
        
        # Sample every window_size-th coordinate in each direction
        sampled_x = unique_x[::self.config.window_size]
        sampled_y = unique_y[::self.config.window_size]
        
        # Create sampling mask: select particles that are at sampled coordinates
        x_mask = torch.isin(x_coords, sampled_x)
        y_mask = torch.isin(y_coords, sampled_y)
        sampling_mask = x_mask & y_mask
        
        # Get sampling indices
        sampling_indices = torch.where(sampling_mask)[0]
        
        # Get sampled positions
        coarse_positions = current_positions[sampling_indices]
        
        # Convert local indices to global grid indices
        # Scale 1+: sample from previous scale, need to chain the indices
        parent_global_indices = self.graph_hierarchy[scale - 1]['sampling_indices']
        global_sampling_indices = parent_global_indices[sampling_indices]
        
        return coarse_positions, new_spacing, global_sampling_indices
    
    def _create_grid_mesh_connectivity(self, 
                                       grid_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create connectivity between grid and first mesh level using radius-based filtering.
        
        Since mesh nodes are sampled from grid nodes, we create a radius graph on the grid
        and then filter edges to identify grid-mesh connections.
        
        Creates:
        1. grid2mesh: Any grid node → Mesh nodes (for aggregation)
        2. mesh2grid: Mesh nodes → Any grid node (for propagation)
        
        Note: Scale 0 is the grid, scale 1 is the first mesh level sampled from the grid.
        
        Returns:
            Tuple of (grid2mesh_edges, mesh2grid_edges)
        """
        # Get first mesh level (Level 1 is the first mesh level, sampled from grid)
        first_mesh_level = 1
        if first_mesh_level not in self.graph_hierarchy:
            raise ValueError(f"First mesh level {first_mesh_level} not found")
        
        mesh_data = self.graph_hierarchy[first_mesh_level]
        mesh_indices = mesh_data['sampling_indices']
        
        # Calculate connection radius based on grid spacing
        connection_radius = self.config.radius_multiplier * self.config.grid_spacing
        
        # Create radius graph on grid only (most efficient)
        grid_edge_index = radius_graph(
            grid_positions, 
            r=connection_radius, 
            loop=False,
            max_num_neighbors=self.config.max_neighbors
        )
        
        # Extract grid-mesh edges efficiently using vectorized operations
        # Since mesh nodes are sampled from grid, we filter edges where:
        # 1. grid2mesh: source is any grid node, target is a mesh node (sampled from grid)
        # 2. mesh2grid: source is a mesh node, target is any grid node
        
        # Filter edges where target is a mesh node (grid2mesh)
        target_is_mesh = torch.isin(grid_edge_index[1], mesh_indices)
        grid2mesh_edge_index = grid_edge_index[:, target_is_mesh]
        
        # Filter edges where source is a mesh node (mesh2grid)
        source_is_mesh = torch.isin(grid_edge_index[0], mesh_indices)
        mesh2grid_edge_index = grid_edge_index[:, source_is_mesh]
        
        return grid2mesh_edge_index, mesh2grid_edge_index
    
    
    def _create_mesh2mesh_edges(self, scale: int) -> torch.Tensor:
        """
        Create edges within a specific mesh scale using radius-based connectivity.
        
        Args:
            scale: Mesh scale index
            
        Returns:
            Edge index tensor with global grid indices
        """
        if scale not in self.graph_hierarchy:
            raise ValueError(f"Scale {scale} not found")
        
        scale_data = self.graph_hierarchy[scale]
        scale_spacing = scale_data['spacing']
        
        # Use uniform radius multiplier for connectivity
        scale_radius = scale_spacing * self.config.radius_multiplier
        
        # All scales are now mesh levels - get positions from sampling indices
        sampling_indices = scale_data['sampling_indices']
        positions = self.grid_positions[sampling_indices]
        
        # Create edges using mesh positions
        edge_index = radius_graph(
            positions, 
            r=scale_radius, 
            loop=False,
            max_num_neighbors=self.config.max_neighbors
        )
        
        # Convert local mesh indices to global grid indices
        source_global = sampling_indices[edge_index[0]]
        target_global = sampling_indices[edge_index[1]]
        mesh2mesh_edge_index = torch.stack([source_global, target_global])
        
        return mesh2mesh_edge_index
    

