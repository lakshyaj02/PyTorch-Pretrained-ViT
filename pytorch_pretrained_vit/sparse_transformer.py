"""
Adapted from https://github.com/lukemelas/simple-bert
"""
 
import numpy as np
from torch import nn
from torch import Tensor 
from torch.nn import functional as F

from pytorch_block_sparse import (
    BlockSparseLinear,
    BlockSparseMatrix,
    BlockSparseMatrixEmulator,
    PseudoBlockSparseLinear,
)


def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)



class BlockSparseMultiHeadedSelfAttention(nn.Module):
    """Multi-Headed Dot Product Attention"""
    def __init__(self, dim, num_heads, dropout, block_shape=(8,8), density=0.1, block_mask=None):
        super().__init__()
        self.block_shape = block_shape
        self.density = density
        self.block_mask = block_mask
        self.proj_q = BlockSparseLinear(dim, dim, True, self.density, block_shape=self.block_shape)
        self.proj_k = BlockSparseLinear(dim, dim, True, self.density, block_shape=self.block_shape)
        self.proj_v = BlockSparseLinear(dim, dim, True, self.density, block_shape=self.block_shape)
        self.drop = nn.Dropout(dropout)
        self.n_heads = num_heads
        self.scores = None # for visualization

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(n_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h

    
class BlockSparsePositionWiseFeedForward(nn.Module):
    """FeedForward Neural Networks for each position"""
    def __init__(self, dim, ff_dim, block_shape=(8,8), density=0.1, block_mask=None):
        super().__init__()
        self.density = density
        self.block_shape = block_shape
        self.block_mask = block_mask
        self.fc1 = BlockSparseLinear(dim, ff_dim, True, self.density, block_shape=self.block_shape)
        self.fc2 = BlockSparseLinear(ff_dim, dim, True, self.density, block_shape=self.block_shape)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(F.gelu(self.fc1(x)))


class BlockSparseBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout, block_shape=(8,8), density=0.1, block_mask=None):
        super().__init__()
        self.density = density
        self.block_shape = block_shape
        self.attn = BlockSparseMultiHeadedSelfAttention(dim, num_heads, dropout, block_shape, density, block_mask)
        self.proj = BlockSparseLinear(dim, dim, True, self.density, block_shape=self.block_shape)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = BlockSparsePositionWiseFeedForward(dim, ff_dim, block_shape, density, block_mask)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.drop(self.proj(self.attn(self.norm1(x), mask)))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x


class BlockSparseTransformer(nn.Module):
    """Transformer with Self-Attentive Blocks"""
    def __init__(self, num_layers, dim, num_heads, ff_dim, dropout, block_shape=(8,8), density=0.1, block_mask=None):
        super().__init__()
        self.blocks = nn.ModuleList([
            BlockSparseBlock(dim, num_heads, ff_dim, dropout, block_shape, density, block_mask) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for block in self.blocks:
            x = block(x, mask)
        return x
