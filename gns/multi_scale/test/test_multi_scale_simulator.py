"""
Test suite for MultiScaleSimulator.

This module contains comprehensive tests for the MultiScaleSimulator class,
including initialization, static graph setup, feature extraction, and prediction.
"""

import sys
import os
import torch
import unittest
import numpy as np

# Add the parent directory to the path to import gns modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from gns.multi_scale.multi_scale_simulator import MultiScaleSimulator
from gns.multi_scale.data_loader_multi_scale import build_static_multi_scale_graph


class TestMultiScaleSimulator(unittest.TestCase):
    """Test cases for MultiScaleSimulator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Test configuration
        self.kinematic_dimensions = 2
        self.nnode_in = 27  # 10 velocity features (5 timesteps × 2D) + 1 wall distance + 16 particle type
        self.nedge_in = 3   # 2 displacement + 1 distance
        self.nedge_out = 32
        self.latent_dim = 64
        self.nmessage_passing_steps = 3
        self.nmlp_layers = 2
        self.connectivity_radius = 1.0
        self.nparticle_types = 2
        self.particle_type_embedding_size = 16
        self.num_scales = 3
        self.window_size = 3
        
        # Normalization stats (typical values for 2D simulation)
        # Based on metadata.json structure: vel_mean/vel_std are 2D vectors (x, y components)
        self.normalization_stats = {
            'velocity': {
                'mean': torch.zeros(2),  # 2D vector for x, y components
                'std': torch.ones(2)
            },
            'acceleration': {
                'mean': torch.zeros(2),  # 2D vector for x, y components
                'std': torch.ones(2)
            }
        }
        
        # Create simulator
        self.simulator = MultiScaleSimulator(
            kinematic_dimensions=self.kinematic_dimensions,
            nnode_in=self.nnode_in,
            nedge_in=self.nedge_in,
            nedge_out=self.nedge_out,
            latent_dim=self.latent_dim,
            nmessage_passing_steps=self.nmessage_passing_steps,
            nmlp_layers=self.nmlp_layers,
            normalization_stats=self.normalization_stats,
            nparticle_types=self.nparticle_types,
            particle_type_embedding_size=self.particle_type_embedding_size,
            num_scales=self.num_scales,
            window_size=self.window_size,
            device="cpu"
        )
        
        # Create test data
        self.num_particles = 25
        self.sequence_length = 6
        
        # Create initial positions (2D grid)
        x = torch.linspace(-2, 2, 5)
        y = torch.linspace(-2, 2, 5)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        self.initial_positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        # Create position sequence (6 timesteps)
        self.position_sequence = torch.randn(
            self.num_particles, self.sequence_length, self.kinematic_dimensions
        )
        
        # Create particle types
        self.particle_types = torch.randint(0, self.nparticle_types, (self.num_particles,))
        
        # Create nparticles_per_example
        self.nparticles_per_example = torch.tensor([self.num_particles])
        
        # Build static graph
        self.graph_data = build_static_multi_scale_graph(
            self.initial_positions, 
            num_scales=self.num_scales, 
            window_size=self.window_size
        )
        
        # Set static graph in simulator
        self.simulator.set_static_graph(self.graph_data)
    
    def test_initialization(self):
        """Test simulator initialization."""
        self.assertIsInstance(self.simulator, MultiScaleSimulator)
        self.assertEqual(self.simulator._kinematic_dimensions, self.kinematic_dimensions)
        self.assertEqual(self.simulator._nparticle_types, self.nparticle_types)
        self.assertEqual(self.simulator._num_scales, self.num_scales)
        self.assertEqual(self.simulator._window_size, self.window_size)
    
    def test_static_graph_setup(self):
        """Test static graph setup and validation."""
        # Test that graph data is set
        graph_data = self.simulator.get_static_graph_data()
        self.assertIsNotNone(graph_data)
        
        # Test required keys are present
        required_keys = ['graph_hierarchy', 'grid2mesh_edges', 'mesh2mesh_edges', 'mesh2grid_edges']
        for key in required_keys:
            self.assertIn(key, graph_data)
        
        # Test graph hierarchy structure
        hierarchy = graph_data['graph_hierarchy']
        self.assertIn(0, hierarchy)  # Grid level
        self.assertIn(1, hierarchy)  # First mesh level
        self.assertIn(2, hierarchy)  # Second mesh level
        
        # Test edge shapes
        self.assertEqual(self.graph_data['grid2mesh_edges'].shape[0], 2)
        self.assertEqual(self.graph_data['mesh2mesh_edges'].shape[0], 2)
        self.assertEqual(self.graph_data['mesh2grid_edges'].shape[0], 2)
    
    def test_node_feature_building(self):
        """Test node feature building."""
        # Test with single particle type
        simulator_single_type = MultiScaleSimulator(
            kinematic_dimensions=self.kinematic_dimensions,
            nnode_in=11,  # 10 velocity + 1 wall distance (no particle type embedding)
            nedge_in=self.nedge_in,
            nedge_out=self.nedge_out,
            latent_dim=self.latent_dim,
            nmessage_passing_steps=self.nmessage_passing_steps,
            nmlp_layers=self.nmlp_layers,
            normalization_stats=self.normalization_stats,
            nparticle_types=1,  # Single particle type
            particle_type_embedding_size=self.particle_type_embedding_size,
            num_scales=self.num_scales,
            window_size=self.window_size,
            device="cpu"
        )
        simulator_single_type.set_static_graph(self.graph_data)
        
        # Test node feature building
        velocity_sequence = torch.randn(self.num_particles, 5, self.kinematic_dimensions)
        most_recent_position = torch.randn(self.num_particles, self.kinematic_dimensions)
        particle_types = torch.zeros(self.num_particles, dtype=torch.long)
        
        node_features = simulator_single_type._build_node_features(
            velocity_sequence, most_recent_position, particle_types, self.num_particles
        )
        
        # Check shape: 10 velocity features + 1 wall distance = 11
        expected_features = 10 + 1  # velocity + wall distance
        self.assertEqual(node_features.shape, (self.num_particles, expected_features))
        
        # Test with multiple particle types
        node_features_multi = self.simulator._build_node_features(
            velocity_sequence, most_recent_position, self.particle_types, self.num_particles
        )
        
        # Check shape: 10 velocity + 1 wall distance + 16 particle type = 27
        expected_features_multi = 10 + 1 + self.particle_type_embedding_size
        self.assertEqual(node_features_multi.shape, (self.num_particles, expected_features_multi))
    
    def test_edge_feature_building(self):
        """Test edge feature building."""
        # Get edge indices from graph data
        g2m_edges = self.graph_data['grid2mesh_edges']
        m2m_edges = self.graph_data['mesh2mesh_edges']
        m2g_edges = self.graph_data['mesh2grid_edges']
        
        most_recent_position = torch.randn(self.num_particles, self.kinematic_dimensions)
        
        # Build edge features
        edge_features = self.simulator._build_edge_features(
            g2m_edges, m2m_edges, m2g_edges, most_recent_position
        )
        
        # Check that all edge types are present
        self.assertIn('g2m', edge_features)
        self.assertIn('m2m', edge_features)
        self.assertIn('m2g', edge_features)
        
        # Check edge feature shapes
        for edge_type in ['g2m', 'm2m', 'm2g']:
            if edge_features[edge_type].shape[0] > 0:
                self.assertEqual(edge_features[edge_type].shape[1], 3)  # 2 displacement + 1 distance
    
    def test_encoder_preprocessor(self):
        """Test encoder preprocessor."""
        # Test encoder preprocessor
        node_features, edge_indices, edge_features = self.simulator._encoder_preprocessor(
            self.position_sequence, self.nparticles_per_example, self.particle_types
        )
        
        # Check node features shape
        expected_node_features = 10 + 1 + self.particle_type_embedding_size  # velocity + wall + particle type
        self.assertEqual(node_features.shape, (self.num_particles, expected_node_features))
        
        # Check edge indices
        self.assertIn('g2m', edge_indices)
        self.assertIn('m2m', edge_indices)
        self.assertIn('m2g', edge_indices)
        
        # Check edge features
        self.assertIn('g2m', edge_features)
        self.assertIn('m2m', edge_features)
        self.assertIn('m2g', edge_features)
    
    def test_predict_positions(self):
        """Test position prediction."""
        # Test position prediction
        next_positions, predicted_strain = self.simulator.predict_positions(
            self.position_sequence, self.nparticles_per_example, self.particle_types
        )
        
        # Check output shapes
        self.assertEqual(next_positions.shape, (self.num_particles, self.kinematic_dimensions))
        self.assertEqual(predicted_strain.shape, (self.num_particles,))
        
        # Check that positions are reasonable (not NaN or infinite)
        self.assertTrue(torch.isfinite(next_positions).all())
        self.assertTrue(torch.isfinite(predicted_strain).all())
    
    def test_predict_accelerations(self):
        """Test acceleration prediction for training."""
        # Create target positions
        next_positions = torch.randn(self.num_particles, self.kinematic_dimensions)
        
        # Create noise
        position_sequence_noise = torch.randn_like(self.position_sequence) * 0.01
        
        # Test acceleration prediction
        pred_acc, target_acc, pred_strain = self.simulator.predict_accelerations(
            next_positions, position_sequence_noise, self.position_sequence,
            self.nparticles_per_example, self.particle_types
        )
        
        # Check output shapes
        self.assertEqual(pred_acc.shape, (self.num_particles, self.kinematic_dimensions))
        self.assertEqual(target_acc.shape, (self.num_particles, self.kinematic_dimensions))
        self.assertEqual(pred_strain.shape, (self.num_particles,))
        
        # Check that outputs are reasonable
        self.assertTrue(torch.isfinite(pred_acc).all())
        self.assertTrue(torch.isfinite(target_acc).all())
        self.assertTrue(torch.isfinite(pred_strain).all())
    
    def test_decoder_postprocessor(self):
        """Test decoder postprocessor."""
        # Create normalized acceleration
        normalized_acceleration = torch.randn(self.num_particles, self.kinematic_dimensions)
        
        # Test decoder postprocessor
        new_positions = self.simulator._decoder_postprocessor(
            normalized_acceleration, self.position_sequence
        )
        
        # Check output shape
        self.assertEqual(new_positions.shape, (self.num_particles, self.kinematic_dimensions))
        
        # Check that positions are reasonable
        self.assertTrue(torch.isfinite(new_positions).all())
    
    def test_inverse_decoder_postprocessor(self):
        """Test inverse decoder postprocessor."""
        # Create target next position
        next_position = torch.randn(self.num_particles, self.kinematic_dimensions)
        
        # Test inverse decoder postprocessor
        normalized_acceleration = self.simulator._inverse_decoder_postprocessor(
            next_position, self.position_sequence
        )
        
        # Check output shape
        self.assertEqual(normalized_acceleration.shape, (self.num_particles, self.kinematic_dimensions))
        
        # Check that acceleration is reasonable
        self.assertTrue(torch.isfinite(normalized_acceleration).all())
    
    def test_time_diff(self):
        """Test time difference calculation."""
        # Test time difference
        velocity_sequence = self.simulator._time_diff(self.position_sequence)
        
        # Check output shape
        expected_shape = (self.num_particles, self.sequence_length - 1, self.kinematic_dimensions)
        self.assertEqual(velocity_sequence.shape, expected_shape)
        
        # Check that velocity is reasonable
        self.assertTrue(torch.isfinite(velocity_sequence).all())
    
    def test_save_load(self):
        """Test model saving and loading."""
        # Save model
        save_path = 'test_multi_scale_simulator.pt'
        self.simulator.save(save_path)
        
        # Check that file exists
        self.assertTrue(os.path.exists(save_path))
        
        # Create new simulator and load
        new_simulator = MultiScaleSimulator(
            kinematic_dimensions=self.kinematic_dimensions,
            nnode_in=self.nnode_in,
            nedge_in=self.nedge_in,
            nedge_out=self.nedge_out,
            latent_dim=self.latent_dim,
            nmessage_passing_steps=self.nmessage_passing_steps,
            nmlp_layers=self.nmlp_layers,
            normalization_stats=self.normalization_stats,
            nparticle_types=self.nparticle_types,
            particle_type_embedding_size=self.particle_type_embedding_size,
            num_scales=self.num_scales,
            window_size=self.window_size,
            device="cpu"
        )
        
        # Load model
        new_simulator.load(save_path)
        
        # Test that loaded model works
        new_simulator.set_static_graph(self.graph_data)
        next_positions, predicted_strain = new_simulator.predict_positions(
            self.position_sequence, self.nparticles_per_example, self.particle_types
        )
        
        # Check that prediction works
        self.assertEqual(next_positions.shape, (self.num_particles, self.kinematic_dimensions))
        self.assertEqual(predicted_strain.shape, (self.num_particles,))
        
        # Clean up
        if os.path.exists(save_path):
            os.remove(save_path)
    
    def test_error_handling(self):
        """Test error handling."""
        # Test without static graph
        simulator_no_graph = MultiScaleSimulator(
            kinematic_dimensions=self.kinematic_dimensions,
            nnode_in=self.nnode_in,
            nedge_in=self.nedge_in,
            nedge_out=self.nedge_out,
            latent_dim=self.latent_dim,
            nmessage_passing_steps=self.nmessage_passing_steps,
            nmlp_layers=self.nmlp_layers,
            normalization_stats=self.normalization_stats,
            nparticle_types=self.nparticle_types,
            particle_type_embedding_size=self.particle_type_embedding_size,
            num_scales=self.num_scales,
            window_size=self.window_size,
            device="cpu"
        )
        
        # Test that prediction fails without static graph
        with self.assertRaises(ValueError):
            simulator_no_graph.predict_positions(
                self.position_sequence, self.nparticles_per_example, self.particle_types
            )


def run_tests():
    """Run all tests."""
    unittest.main(verbosity=2)


if __name__ == '__main__':
    run_tests()
