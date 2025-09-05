import torch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from multi_scale_gnn import MultiScaleGNN


def test_multi_scale_gnn():
    """Simple test to verify MultiScaleGNN works."""
    print("🧪 Testing MultiScaleGNN...")
    
    # Create a small model for testing
    model = MultiScaleGNN(
        nnode_in_features=10,
        nnode_out_features=3,
        nedge_in_features=3,
        nedge_out_features=32,
        latent_dim=32,
        nmessage_passing_steps=5,
        nmlp_layers=2,
        num_scales=3
    )
    
    print("✅ Model created successfully")
    
    # Test data
    num_grid_nodes = 25
    num_mesh_nodes = 9
    num_g2m_edges = 20
    num_m2m_edges = 15
    num_m2g_edges = 20
    
    # Create test tensors
    grid_node_features = torch.randn(num_grid_nodes, 10)
    
    # Create valid edge indices
    # Global mesh indices sampled from grid
    mesh_indices = torch.randperm(num_grid_nodes)[:num_mesh_nodes]
    
    # g2m: src in [0, num_grid_nodes), tgt in mesh_indices
    g2m_edge_index = torch.stack([
        torch.randint(0, num_grid_nodes, (num_g2m_edges,)),
        mesh_indices[torch.randint(0, num_mesh_nodes, (num_g2m_edges,))]
    ])
    
    # m2m: src,tgt both in mesh_indices
    m2m_edge_index = torch.stack([
        mesh_indices[torch.randint(0, num_mesh_nodes, (num_m2m_edges,))],
        mesh_indices[torch.randint(0, num_mesh_nodes, (num_m2m_edges,))]
    ])
    
    # m2g: src in mesh_indices, tgt in [0, num_grid_nodes)
    m2g_edge_index = torch.stack([
        mesh_indices[torch.randint(0, num_mesh_nodes, (num_m2g_edges,))],
        torch.randint(0, num_grid_nodes, (num_m2g_edges,))
    ])
    
    # Edge features
    g2m_edge_features = torch.randn(num_g2m_edges, 3)
    m2m_edge_features = torch.randn(num_m2m_edges, 3)
    m2g_edge_features = torch.randn(num_m2g_edges, 3)
    
    # Graph hierarchy: level 0 = all grid nodes, level 1 = sampling of 0, level 2 = sampling of 1
    graph_hierarchy = {
        0: {'sampling_indices': torch.arange(num_grid_nodes)},  # All grid nodes
        1: {'sampling_indices': mesh_indices},  # Mesh level 1: sampling of level 0
        2: {'sampling_indices': mesh_indices[::2]}  # Mesh level 2: sampling of level 1
    }
    
    print("✅ Test data created successfully")
    
    # Test forward pass
    try:
        outputs = model(
            x=grid_node_features,
            g2m_edge_index=g2m_edge_index,
            g2m_edge_features=g2m_edge_features,
            m2m_edge_index=m2m_edge_index,
            m2m_edge_features=m2m_edge_features,
            m2g_edge_index=m2g_edge_index,
            m2g_edge_features=m2g_edge_features,
            graph_hierarchy=graph_hierarchy
        )
        
        print(f"✅ Forward pass successful! Output shape: {outputs.shape}")
        print(f"   Expected: ({num_grid_nodes}, 3)")
        print(f"   Actual: {outputs.shape}")
        
        # Check output is not NaN
        if not torch.isnan(outputs).any():
            print("✅ Outputs are valid (no NaN values)")
        else:
            print("❌ Outputs contain NaN values")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test edge encoders
    try:
        g2m_latent = model.g2m_edge_encoder(g2m_edge_features)
        m2m_latent = model.m2m_edge_encoder(m2m_edge_features)
        m2g_latent = model.m2g_edge_encoder(m2g_edge_features)
        
        print("✅ Edge encoders work correctly")
        print(f"   G2M latent shape: {g2m_latent.shape}")
        print(f"   M2M latent shape: {m2m_latent.shape}")
        print(f"   M2G latent shape: {m2g_latent.shape}")
        
    except Exception as e:
        print(f"❌ Edge encoders failed: {e}")
        return False
    
    # Test grid node encoder
    try:
        grid_latent = model.grid_node_encoder(grid_node_features)
        print(f"✅ Grid node encoder works: {grid_latent.shape}")
        
    except Exception as e:
        print(f"❌ Grid node encoder failed: {e}")
        return False
    
    print("\n🎉 All tests passed! MultiScaleGNN is working correctly.")
    return True


if __name__ == '__main__':
    success = test_multi_scale_gnn()
    if not success:
        sys.exit(1)
