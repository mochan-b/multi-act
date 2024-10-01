import torch

from detr.models.position_encoding import PositionEmbeddingSine

def test_position_embedding_sine():
    # Create a PositionEmbeddingSine object
    pos_emb = PositionEmbeddingSine(512, normalize=True)

    # Create a random tensor to test the position embedding
    input = torch.randn(16, 512, 15, 30)

    # # Get the position embedding
    pos = pos_emb(input)

    # # Check the shape of the position embedding
    assert pos.shape == (1, 512, 15, 30)