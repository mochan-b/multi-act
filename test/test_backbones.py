#!/usr/bin/env python3
"""
Pytest-based unit tests for backbone implementations.

Tests the new MobileNet and EfficientNet backbones that are designed
for image history processing to reduce memory usage.

Run with: pytest test/test_backbones.py -v
"""

import pytest
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'detr'))

from detr.models.backbone import build_backbone, MobileNetBackbone, EfficientNetBackbone


class MockArgs:
    """Mock args class for testing backbone builds."""
    def __init__(self):
        self.backbone = 'resnet18'
        self.lr_backbone = 0.0001
        self.masks = False
        self.dilation = False
        self.position_embedding = 'sine'
        self.hidden_dim = 256


@pytest.fixture
def mock_args():
    """Fixture for mock arguments."""
    return MockArgs()


@pytest.fixture
def dummy_image():
    """Fixture for dummy image tensor."""
    return torch.randn(2, 3, 224, 224)


class TestBackbones:
    """Test suite for backbone implementations."""
    
    def test_resnet18_backbone(self, mock_args, dummy_image):
        """Test ResNet18 backbone (baseline)."""
        backbone = build_backbone(mock_args, 'resnet18')
        assert backbone.num_channels == 512
        
        output = backbone(dummy_image)
        features, pos = output
        assert len(features) == 1  # Single output layer
        assert features[0].shape == (2, 512, 7, 7)
    
    def test_mobilenet_v2_backbone(self, mock_args, dummy_image):
        """Test MobileNetV2 backbone (lightweight option)."""
        backbone = build_backbone(mock_args, 'mobilenet_v2')
        assert backbone.num_channels == 1280
        
        output = backbone(dummy_image)
        features, pos = output
        assert len(features) == 1  # Single output layer
        assert features[0].shape == (2, 1280, 7, 7)
    
    def test_mobilenet_v3_small_backbone(self, mock_args, dummy_image):
        """Test MobileNetV3 Small backbone (smallest option)."""
        backbone = build_backbone(mock_args, 'mobilenet_v3_small')
        assert backbone.num_channels == 576
        
        output = backbone(dummy_image)
        features, pos = output
        assert len(features) == 1
        assert features[0].shape == (2, 576, 7, 7)
    
    def test_mobilenet_v3_large_backbone(self, mock_args, dummy_image):
        """Test MobileNetV3 Large backbone."""
        backbone = build_backbone(mock_args, 'mobilenet_v3_large')
        assert backbone.num_channels == 960
        
        output = backbone(dummy_image)
        features, pos = output
        assert len(features) == 1
        assert features[0].shape[0] == 2
        assert features[0].shape[1] == 960
    
    def test_efficientnet_b0_backbone(self, mock_args, dummy_image):
        """Test EfficientNetB0 backbone (efficient option)."""
        backbone = build_backbone(mock_args, 'efficientnet_b0')
        assert backbone.num_channels == 1280
        
        output = backbone(dummy_image)
        features, pos = output
        assert len(features) == 1
        assert features[0].shape == (2, 1280, 7, 7)
    
    def test_efficientnet_b1_backbone(self, mock_args, dummy_image):
        """Test EfficientNetB1 backbone."""
        backbone = build_backbone(mock_args, 'efficientnet_b1')
        assert backbone.num_channels == 1280
        
        output = backbone(dummy_image)
        features, pos = output
        assert len(features) == 1
        assert features[0].shape == (2, 1280, 7, 7)
    
    def test_invalid_backbone_name(self, mock_args):
        """Test that invalid backbone names raise appropriate errors."""
        with pytest.raises(ValueError):
            build_backbone(mock_args, 'invalid_backbone')
        
        with pytest.raises(ValueError):
            MobileNetBackbone('invalid_mobilenet', True, False)
        
        with pytest.raises(ValueError):
            EfficientNetBackbone('invalid_efficientnet', True, False)
    
    def test_memory_efficiency_comparison(self, mock_args):
        """Test that lightweight backbones have fewer parameters than ResNet."""
        resnet_backbone = build_backbone(mock_args, 'resnet18')
        mobilenet_backbone = build_backbone(mock_args, 'mobilenet_v2')
        efficientnet_backbone = build_backbone(mock_args, 'efficientnet_b0')
        
        # Count parameters
        resnet_params = sum(p.numel() for p in resnet_backbone.parameters())
        mobilenet_params = sum(p.numel() for p in mobilenet_backbone.parameters())
        efficientnet_params = sum(p.numel() for p in efficientnet_backbone.parameters())
        
        # MobileNet should have fewer parameters than ResNet
        assert mobilenet_params < resnet_params, f"MobileNet ({mobilenet_params}) should have fewer params than ResNet ({resnet_params})"
        
        # EfficientNet should be more efficient than ResNet
        assert efficientnet_params < resnet_params, f"EfficientNet ({efficientnet_params}) should have fewer params than ResNet ({resnet_params})"
        
        # Print comparison for debugging
        print(f"\nParameter counts:")
        print(f"  ResNet18: {resnet_params:,}")
        print(f"  MobileNetV2: {mobilenet_params:,}")
        print(f"  EfficientNetB0: {efficientnet_params:,}")
        print(f"  Memory reduction - MobileNet: {(1 - mobilenet_params/resnet_params)*100:.1f}%")
        print(f"  Memory reduction - EfficientNet: {(1 - efficientnet_params/resnet_params)*100:.1f}%")
    
    @pytest.mark.parametrize("backbone_name,expected_channels", [
        ('resnet18', 512),
        ('mobilenet_v2', 1280),
        ('mobilenet_v3_small', 576),
        ('efficientnet_b0', 1280)
    ])
    def test_backbone_consistency(self, mock_args, dummy_image, backbone_name, expected_channels):
        """Test that all backbones produce consistent output shapes."""
        backbone = build_backbone(mock_args, backbone_name)
        output = backbone(dummy_image)
        features, pos = output
        
        # All should produce single feature map
        assert len(features) == 1, f"{backbone_name} should produce single feature map"
        
        # Check channel count
        assert features[0].shape[1] == expected_channels, f"{backbone_name} channel mismatch"
        
        # Check spatial dimensions (should be 7x7 for 224x224 input)
        assert features[0].shape[2:] == (7, 7), f"{backbone_name} spatial dimension mismatch"
        
        # Check batch dimension
        assert features[0].shape[0] == 2, f"{backbone_name} batch dimension mismatch"


@pytest.mark.slow
class TestBackbonePerformance:
    """Performance tests for backbones (marked as slow)."""
    
    def test_inference_speed_comparison(self, mock_args):
        """Compare inference speed of different backbones."""
        import time
        
        dummy_image = torch.randn(1, 3, 224, 224)
        backbones = {
            'resnet18': build_backbone(mock_args, 'resnet18'),
            'mobilenet_v2': build_backbone(mock_args, 'mobilenet_v2'),
            'efficientnet_b0': build_backbone(mock_args, 'efficientnet_b0')
        }
        
        results = {}
        for name, backbone in backbones.items():
            # Warmup
            for _ in range(5):
                _ = backbone(dummy_image)
            
            # Time inference
            start_time = time.time()
            for _ in range(10):
                _ = backbone(dummy_image)
            end_time = time.time()
            
            results[name] = (end_time - start_time) / 10
        
        print(f"\nInference times (per image):")
        for name, time_per_img in results.items():
            print(f"  {name}: {time_per_img*1000:.2f}ms")
        
        # MobileNet should be faster than ResNet
        assert results['mobilenet_v2'] <= results['resnet18'] * 1.5, "MobileNet should be reasonably fast"