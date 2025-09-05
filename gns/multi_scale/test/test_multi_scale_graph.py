#!/usr/bin/env python3
"""
Unit tests for MultiScaleGraph functionality.

This module tests:
1. MultiScaleConfig initialization and validation
2. MultiScaleGraph hierarchy building
3. Edge generation for all edge types
4. Edge validation and consistency
5. Error handling and edge cases
"""

import unittest
import torch
import numpy as np
import tempfile
import os

# Add parent directory to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from gns.multi_scale.multi_scale_graph import MultiScaleGraph, MultiScaleConfig


class TestMultiScaleConfig(unittest.TestCase):
    """Test MultiScaleConfig class functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MultiScaleConfig()
        
        self.assertEqual(config.num_scales, 3)
        self.assertEqual(config.window_size, 3)
        self.assertEqual(config.grid_spacing, 0.5)
        self.assertEqual(config.radius_multiplier, 2)
        self.assertEqual(config.max_neighbors, 20)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MultiScaleConfig(
            num_scales=5,
            window_size=4
        )
        
        self.assertEqual(config.num_scales, 5)
        self.assertEqual(config.window_size, 4)
        self.assertEqual(config.grid_spacing, 0.5)  # Default
        self.assertEqual(config.radius_multiplier, 2)  # Default
        self.assertEqual(config.max_neighbors, 20)  # Default
    
    def test_config_attributes(self):
        """Test that all config attributes are accessible."""
        config = MultiScaleConfig()
        
        # Test attribute access
        self.assertTrue(hasattr(config, 'num_scales'))
        self.assertTrue(hasattr(config, 'window_size'))
        self.assertTrue(hasattr(config, 'grid_spacing'))
        self.assertTrue(hasattr(config, 'radius_multiplier'))
        self.assertTrue(hasattr(config, 'max_neighbors'))


class TestMultiScaleGraph(unittest.TestCase):
    """Test MultiScaleGraph class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MultiScaleConfig(
            num_scales=3,
            window_size=2
        )
        self.graph = MultiScaleGraph(self.config)
        
        # Create simple 4x4 grid for testing
        x_coords = torch.arange(4, dtype=torch.float32) * 1.0
        y_coords = torch.arange(4, dtype=torch.float32) * 1.0
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
        self.grid_positions = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
        
        # Expected hierarchy:
        # Scale 0: 16 particles (4x4 grid)
        # Scale 1: 4 particles (2x2 sampled)
        # Scale 2: 1 particle (1x1 sampled)
    
    def test_initialization(self):
        """Test MultiScaleGraph initialization."""
        self.assertIsInstance(self.graph.config, MultiScaleConfig)
        self.assertIsNone(self.graph.grid_positions)
        self.assertEqual(self.graph.graph_hierarchy, {})
    
    def test_build_hierarchy_basic(self):
        """Test basic hierarchy building."""
        hierarchy = self.graph.build_hierarchy(self.grid_positions)
        
        # Check that hierarchy was built
        self.assertIn(0, hierarchy)
        self.assertIn(1, hierarchy)
        self.assertIn(2, hierarchy)
        
        # Check scale 0 (grid)
        scale_0 = hierarchy[0]
        self.assertEqual(scale_0['num_particles'], 16)
        self.assertEqual(scale_0['spacing'], 0.5)
        self.assertEqual(len(scale_0['sampling_indices']), 16)
        
        # Check scale 1 (first mesh)
        scale_1 = hierarchy[1]
        self.assertEqual(scale_1['num_particles'], 4)
        self.assertEqual(scale_1['spacing'], 1.0)  # 0.5 * 2
        self.assertEqual(len(scale_1['sampling_indices']), 4)
        
        # Check scale 2 (second mesh)
        scale_2 = hierarchy[2]
        self.assertEqual(scale_2['num_particles'], 1)
        self.assertEqual(scale_2['spacing'], 2.0)  # 1.0 * 2
        self.assertEqual(len(scale_2['sampling_indices']), 1)
    
    def test_hierarchy_sampling_indices(self):
        """Test that sampling indices are properly chained."""
        hierarchy = self.graph.build_hierarchy(self.grid_positions)
        
        # Scale 0 should have all indices
        scale_0_indices = hierarchy[0]['sampling_indices']
        expected_scale_0 = torch.arange(16, dtype=torch.long)
        self.assertTrue(torch.equal(scale_0_indices, expected_scale_0))
        
        # Scale 1 should sample from scale 0
        scale_1_indices = hierarchy[1]['sampling_indices']
        # With window_size=2, should sample every 2nd particle
        # This gives indices [0, 2, 8, 10] for a 4x4 grid
        expected_scale_1 = torch.tensor([0, 2, 8, 10], dtype=torch.long)
        self.assertTrue(torch.equal(scale_1_indices, expected_scale_1))
        
        # Scale 2 should sample from scale 1
        scale_2_indices = hierarchy[2]['sampling_indices']
        # With window_size=2, should sample every 2nd particle from scale 1
        # This gives index [0] (first particle from scale 1)
        expected_scale_2 = torch.tensor([0], dtype=torch.long)
        self.assertTrue(torch.equal(scale_2_indices, expected_scale_2))
    
    def test_create_all_edges(self):
        """Test complete edge creation."""
        edges = self.graph.create_all_edges(self.grid_positions)
        
        # Check that all edge types are present
        self.assertIn('graph_hierarchy', edges)
        self.assertIn('grid2mesh_edges', edges)
        self.assertIn('mesh2mesh_edges', edges)
        self.assertIn('mesh2grid_edges', edges)
        
        # Check edge counts
        self.assertGreater(edges['grid2mesh_edges'].shape[1], 0)
        self.assertGreater(edges['mesh2mesh_edges'].shape[1], 0)
        self.assertGreater(edges['mesh2grid_edges'].shape[1], 0)
        
        # Check edge shapes
        self.assertEqual(edges['grid2mesh_edges'].shape[0], 2)
        self.assertEqual(edges['mesh2mesh_edges'].shape[0], 2)
        self.assertEqual(edges['mesh2grid_edges'].shape[0], 2)
    
    def test_edge_validation(self):
        """Test that all edges have valid indices."""
        edges = self.graph.create_all_edges(self.grid_positions)
        
        # Check grid2mesh edges
        grid2mesh = edges['grid2mesh_edges']
        self.assertTrue((grid2mesh[0] >= 0).all())  # Sources >= 0
        self.assertTrue((grid2mesh[0] < 16).all())  # Sources < total particles
        
        # Check mesh2grid edges
        mesh2grid = edges['mesh2grid_edges']
        self.assertTrue((mesh2grid[1] >= 0).all())  # Targets >= 0
        self.assertTrue((mesh2grid[1] < 16).all())  # Targets < total particles
        
        # Check mesh2mesh edges
        mesh2mesh = edges['mesh2mesh_edges']
        self.assertTrue((mesh2mesh[0] >= 0).all())  # Sources >= 0
        self.assertTrue((mesh2mesh[1] >= 0).all())  # Targets >= 0
        self.assertTrue((mesh2mesh[0] < 16).all())  # Sources < total particles
        self.assertTrue((mesh2mesh[1] < 16).all())  # Targets < total particles
    
    def test_edge_symmetry(self):
        """Test that grid2mesh and mesh2grid edges are symmetric."""
        edges = self.graph.create_all_edges(self.grid_positions)
        
        grid2mesh = edges['grid2mesh_edges']
        mesh2grid = edges['mesh2grid_edges']
        
        # Convert to sets for comparison
        grid2mesh_pairs = set()
        for i in range(grid2mesh.shape[1]):
            src, tgt = grid2mesh[0, i].item(), grid2mesh[1, i].item()
            grid2mesh_pairs.add((src, tgt))
        
        mesh2grid_pairs = set()
        for i in range(mesh2grid.shape[1]):
            src, tgt = mesh2grid[0, i].item(), mesh2grid[1, i].item()
            mesh2grid_pairs.add((src, tgt))
        
        # Check symmetry: for each grid->mesh edge, there should be a corresponding mesh->grid edge
        symmetric_count = 0
        for src, tgt in grid2mesh_pairs:
            if (tgt, src) in mesh2grid_pairs:
                symmetric_count += 1
        
        # Should have perfect symmetry
        self.assertEqual(symmetric_count, len(grid2mesh_pairs))
    
    def test_no_self_loops(self):
        """Test that no edges connect a particle to itself."""
        edges = self.graph.create_all_edges(self.grid_positions)
        
        # Check mesh2mesh edges for self-loops
        mesh2mesh = edges['mesh2mesh_edges']
        self.assertTrue((mesh2mesh[0] != mesh2mesh[1]).all())
        
        # Check grid2mesh edges for self-loops
        grid2mesh = edges['grid2mesh_edges']
        self.assertTrue((grid2mesh[0] != grid2mesh[1]).all())
        
        # Check mesh2grid edges for self-loops
        mesh2grid = edges['mesh2grid_edges']
        self.assertTrue((mesh2grid[0] != mesh2grid[1]).all())
    
    def test_no_duplicate_edges(self):
        """Test that no duplicate edges exist."""
        edges = self.graph.create_all_edges(self.grid_positions)
        
        # Check mesh2mesh edges
        mesh2mesh = edges['mesh2mesh_edges']
        edge_pairs = mesh2mesh.T
        unique_pairs, counts = torch.unique(edge_pairs, dim=0, return_counts=True)
        self.assertTrue((counts == 1).all())  # All edges should appear exactly once
        
        # Check grid2mesh edges
        grid2mesh = edges['grid2mesh_edges']
        edge_pairs = grid2mesh.T
        unique_pairs, counts = torch.unique(edge_pairs, dim=0, return_counts=True)
        self.assertTrue((counts == 1).all())
        
        # Check mesh2grid edges
        mesh2grid = edges['mesh2grid_edges']
        edge_pairs = mesh2grid.T
        unique_pairs, counts = torch.unique(edge_pairs, dim=0, return_counts=True)
        self.assertTrue((counts == 1).all())


class TestMultiScaleGraphEdgeCases(unittest.TestCase):
    """Test MultiScaleGraph edge cases and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MultiScaleConfig(num_scales=2, window_size=2)
        self.graph = MultiScaleGraph(self.config)
    
    def test_single_particle(self):
        """Test with single particle (edge case)."""
        positions = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        
        # Should not raise error
        edges = self.graph.create_all_edges(positions)
        
        # Check hierarchy
        hierarchy = edges['graph_hierarchy']
        self.assertEqual(hierarchy[0]['num_particles'], 1)
        self.assertEqual(hierarchy[1]['num_particles'], 1)
        
        # Check edges
        self.assertEqual(edges['grid2mesh_edges'].shape[1], 0)  # No grid2mesh edges
        self.assertEqual(edges['mesh2mesh_edges'].shape[1], 0)  # No mesh2mesh edges
        self.assertEqual(edges['mesh2grid_edges'].shape[1], 0)  # No mesh2grid edges
    
    def test_two_particles(self):
        """Test with two particles (edge case)."""
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
        
        edges = self.graph.create_all_edges(positions)
        
        # Check hierarchy
        hierarchy = edges['graph_hierarchy']
        self.assertEqual(hierarchy[0]['num_particles'], 2)
        self.assertEqual(hierarchy[1]['num_particles'], 1)  # window_size=2, so sample 1st
    
    def test_invalid_config(self):
        """Test with invalid configuration."""
        # Test with num_scales < 2
        with self.assertRaises(ValueError):
            config = MultiScaleConfig(num_scales=1)
            graph = MultiScaleGraph(config)
            graph.create_all_edges(torch.randn(10, 2))
    
    def test_empty_positions(self):
        """Test with empty positions tensor."""
        positions = torch.empty(0, 2, dtype=torch.float32)
        
        with self.assertRaises(ValueError):
            self.graph.create_all_edges(positions)


class TestMultiScaleGraphDifferentConfigs(unittest.TestCase):
    """Test MultiScaleGraph with different configurations."""
    
    def test_large_window_size(self):
        """Test with large window size."""
        config = MultiScaleConfig(num_scales=3, window_size=4)
        graph = MultiScaleGraph(config)
        
        # Create 8x8 grid
        x_coords = torch.arange(8, dtype=torch.float32) * 1.0
        y_coords = torch.arange(8, dtype=torch.float32) * 1.0
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
        positions = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
        
        edges = graph.create_all_edges(positions)
        hierarchy = edges['graph_hierarchy']
        
        # Scale 0: 64 particles
        self.assertEqual(hierarchy[0]['num_particles'], 64)
        # Scale 1: 16 particles (64 / 4^2)
        self.assertEqual(hierarchy[1]['num_particles'], 16)
        # Scale 2: 4 particles (16 / 4^2)
        self.assertEqual(hierarchy[2]['num_particles'], 4)
    
    def test_many_scales(self):
        """Test with many scales."""
        config = MultiScaleConfig(num_scales=5, window_size=2)
        graph = MultiScaleGraph(config)
        
        # Create 16x16 grid
        x_coords = torch.arange(16, dtype=torch.float32) * 1.0
        y_coords = torch.arange(16, dtype=torch.float32) * 1.0
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
        positions = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
        
        edges = graph.create_all_edges(positions)
        hierarchy = edges['graph_hierarchy']
        
        # Check all scales exist
        for scale in range(5):
            self.assertIn(scale, hierarchy)
        
        # Check particle counts decrease exponentially
        self.assertEqual(hierarchy[0]['num_particles'], 256)  # 16^2
        self.assertEqual(hierarchy[1]['num_particles'], 64)   # 256 / 2^2
        self.assertEqual(hierarchy[2]['num_particles'], 16)   # 64 / 2^2
        self.assertEqual(hierarchy[3]['num_particles'], 4)    # 16 / 2^2
        self.assertEqual(hierarchy[4]['num_particles'], 1)    # 4 / 2^2


class TestMultiScaleGraphConsistency(unittest.TestCase):
    """Test MultiScaleGraph consistency across multiple calls."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = MultiScaleConfig(num_scales=3, window_size=2)
        self.graph = MultiScaleGraph(self.config)
        
        # Create test positions
        x_coords = torch.arange(4, dtype=torch.float32) * 1.0
        y_coords = torch.arange(4, dtype=torch.float32) * 1.0
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')
        self.positions = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
    
    def test_multiple_calls_consistency(self):
        """Test that multiple calls produce consistent results."""
        edges1 = self.graph.create_all_edges(self.positions)
        edges2 = self.graph.create_all_edges(self.positions)
        
        # Check that results are identical
        self.assertTrue(torch.equal(edges1['grid2mesh_edges'], edges2['grid2mesh_edges']))
        self.assertTrue(torch.equal(edges1['mesh2mesh_edges'], edges2['mesh2mesh_edges']))
        self.assertTrue(torch.equal(edges1['mesh2grid_edges'], edges2['mesh2grid_edges']))
    
    def test_hierarchy_persistence(self):
        """Test that hierarchy persists between calls."""
        # First call
        edges1 = self.graph.create_all_edges(self.positions)
        hierarchy1 = edges1['graph_hierarchy']
        
        # Second call
        edges2 = self.graph.create_all_edges(self.positions)
        hierarchy2 = edges2['graph_hierarchy']
        
        # Check that hierarchy is the same object
        self.assertIs(hierarchy1, hierarchy2)
        
        # Check that hierarchy content is identical
        for scale in hierarchy1:
            self.assertTrue(torch.equal(
                hierarchy1[scale]['sampling_indices'],
                hierarchy2[scale]['sampling_indices']
            ))
            self.assertEqual(hierarchy1[scale]['spacing'], hierarchy2[scale]['spacing'])
            self.assertEqual(hierarchy1[scale]['num_particles'], hierarchy2[scale]['num_particles'])


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
