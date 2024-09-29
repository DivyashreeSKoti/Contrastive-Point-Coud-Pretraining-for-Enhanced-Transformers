# Partial code in this file is developed with help of chatGPT, specifically for debugging.
# Code base reference
# Ludwig, D. (n.d.). Set transformer MNIST. GitHub.
# https://github.com/DLii-Research/tf-settransformer/
# Lee, J. (n.d.). Set Transformer. GitHub. https://github.com/juho-lee/set_transformer/blob/master/modules.py 

#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
# import pytorch_lightning as pl
import torch.optim as optim

class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CustomMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, query, key, value):
        # Apply linear projections to query, key, and value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Compute scaled dot-product attention
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))

        # Apply softmax activation
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # Apply attention weights to value
        attention_output = torch.matmul(attention_probs, v)

        attention_output = self._merge_heads(attention_output)
        attention_output = self.out_proj(attention_output)
        return attention_output, attention_probs

    def _split_heads(self, x):
        batch_size, sequence_length, _ = x.size()
        x = x.view(batch_size, sequence_length, self.num_heads, -1)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size * self.num_heads, sequence_length, -1)
        return x

    def _merge_heads(self, x):
        batch_size, _, _ = x.size()
        x = x.view(batch_size, self.num_heads, -1, self.embed_dim // self.num_heads)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, -1, self.embed_dim)
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim=None,
        ff_activation="relu",
        use_layernorm=True,
        pre_layernorm=False,
        is_final_block=False
    ):
        super(MultiHeadAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = embed_dim if ff_dim is None else ff_dim
        self.ff_activation = ff_activation
        self.use_layernorm = use_layernorm
        self.pre_layernorm = pre_layernorm
        self.is_final_block = is_final_block

        # Custom Multihead Attention layer
        self.att = CustomMultiheadAttention(
            embed_dim, num_heads)

        # Feed-forward layer
        self.ffn = nn.Sequential(
            nn.Linear(self.embed_dim, self.ff_dim),
            nn.ReLU(),
            nn.Linear(self.ff_dim, self.embed_dim)
        )

        self.layernorm_x = nn.LayerNorm(self.embed_dim, eps=1e-3)
        self.layernorm_y = nn.LayerNorm(self.embed_dim, eps=1e-3)
        self.layernorm_attn = nn.LayerNorm(self.embed_dim, eps=1e-3)
        if self.is_final_block:
            self.layernorm_final = nn.LayerNorm(self.embed_dim, eps=1e-3)

    def forward(self, inputs, return_attention_scores=False):
        x, y = inputs[0], inputs[1]
        if self.use_layernorm and self.pre_layernorm:
            return self.call_pre_layernorm(x, y, return_attention_scores)
        return self.call_post_attention_layer(x, y, return_attention_scores)
    
    def call_pre_layernorm(self, x, y, return_attention_scores):
        x_norm = self.layernorm_attn(x)
        y_norm = self.layernorm_attn(y) if y is not x else x_norm

        # Multi-head attention
        attn_output, attn_probs = self.att(x_norm, y_norm, y_norm)
        attn_output = attn_output.view(x.size())
        attn_output = x + attn_output

        # ff-projection
        out = self.layernorm_attn(attn_output)
        out = attn_output + self.ffn(out)

        if self.is_final_block:
            out = self.layernorm_final(out)

        return out
    
    def call_post_attention_layer(self, x, y, return_attention_scores=False):           
        # Multi-head attention
        attn_scores = None
        attn_output, attn_probs = self.att(x, y, y)
        attn_output = attn_output.view(x.size())
        attn = x + attn_output

        if self.use_layernorm:
            attn = self.layernorm_attn(attn)

        out = attn + self.ffn(attn)
        if self.is_final_block:
            out = self.layernorm_final(out)

        if return_attention_scores:
            return out, attn_scores
        return out


class InducedSetAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_induce,
        ff_dim=None,
        ff_activation="relu",
        use_layernorm=True,
        pre_layernorm=False,
        is_final_block=False,
    ):
        super(InducedSetAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_induce = num_induce

        self.mab1 = MultiHeadAttentionBlock(
            embed_dim, num_heads, ff_dim, ff_activation, use_layernorm = use_layernorm, pre_layernorm = pre_layernorm,
            is_final_block=False)
        self.mab2 = MultiHeadAttentionBlock(
            embed_dim, num_heads, ff_dim, ff_activation, use_layernorm = use_layernorm, pre_layernorm = pre_layernorm,
            is_final_block=is_final_block)

        self.inducing_points = nn.Parameter(torch.Tensor(1, num_induce, embed_dim))
        nn.init.xavier_uniform_(self.inducing_points)

    def forward(self, x, return_attention_scores=False):
        batch_size = x.size(0)
        i = self.inducing_points.repeat(batch_size, 1, 1)
        # print('i',i.shape)
        # print('x',x.shape)
        h = self.mab1((i, x), return_attention_scores=False)
        # print('h',h.shape)
        # print('x',x.shape)
        result = self.mab2((x, h), return_attention_scores=return_attention_scores)
        # print('result',result.shape)
        return result

class PoolingByMultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_seeds,
        embed_dim,
        num_heads,
        ff_dim=None,
        ff_activation="gelu",
        use_layernorm=True,
        pre_layernorm=False,
        is_final_block=False,
        **kwargs
    ):
        super(PoolingByMultiHeadAttention, self).__init__(**kwargs)
        self.num_seeds = num_seeds
        self.embed_dim = embed_dim

        self.mab = MultiHeadAttentionBlock(
            embed_dim, num_heads, ff_dim, ff_activation, use_layernorm=use_layernorm,pre_layernorm=pre_layernorm,
            is_final_block=is_final_block)

        self.seed_vectors = nn.Parameter(
            torch.randn(1, self.num_seeds, self.embed_dim),
            requires_grad=True
        )

    def forward(self, z):
        batch_size = z.size(0)
        seeds = self.seed_vectors.repeat(batch_size, 1, 1)
        return self.mab((seeds, z))

    
# Define the PyTorch model
class PyTorchModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_induce, stack, ff_activation, dropout, use_layernorm, pre_layernorm, is_final_block, num_classes):
        super(PyTorchModel, self).__init__()
        self.embed_dim = embed_dim
        self.input = torch.nn.Identity()
        
        self.dense = nn.Linear(3, embed_dim)
        self.pre_layernorm = pre_layernorm
        self.induced_set_attention_blocks = nn.ModuleList([
            InducedSetAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_induce=num_induce,
                ff_activation=ff_activation,
                use_layernorm=use_layernorm,
                pre_layernorm = pre_layernorm,
                is_final_block = is_final_block)
            for _ in range(stack)
        ])

        self.dropout = nn.Dropout(dropout)

        self.pooling_attention = PoolingByMultiHeadAttention(
            num_seeds=1,
            embed_dim=embed_dim,
            num_heads=1,
            ff_activation=ff_activation,
            use_layernorm=use_layernorm,
            pre_layernorm = pre_layernorm,
            is_final_block=True
        )

        self.final_dropout = nn.Dropout(dropout)
        self.reshape = nn.Flatten()
        self.final_dense = nn.Linear(embed_dim, num_classes)
        
        # Define the loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x,num_classes=None,device=None, get_embeddings = False):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        x = x.to(self.dense.weight.dtype)  # Convert x to the same dtype as the weight matrix
        y=x
        y = self.dense(y)
        for block in self.induced_set_attention_blocks:
            y = block(y)

        y = self.dropout(y)
        y = self.pooling_attention(y)
        y = self.dropout(y)
        y = self.reshape(y)

        if get_embeddings:
            y_embeddings = y
        y = self.final_dense(y)
        # Get embeddings along with final prediction
        if get_embeddings: 
            return y, y_embeddings
        
        return y