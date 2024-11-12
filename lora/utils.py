import torch
import torch.nn as nn
import mlx_lm

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank_size):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        in_features, out_features = original_layer.weight.shape
        self.name = "LoRALayer"
        # Define low-rank matrices A and B
        self.A = nn.Parameter(torch.randn(out_features, rank_size))
        self.B = nn.Parameter(torch.randn(rank_size, in_features))
        
    def forward(self, x):
        # Compute the low-rank approximation: W_approx = A @ B
        W_approx = self.A @ self.B
        # Add W_approx to the original weights for an adaptive approach (or replace completely)
        adapted_weights = W_approx + self.original_layer.weight
        return nn.functional.linear(x, adapted_weights, self.original_layer.bias)
    

class LoRAMLP(nn.Module):
    def __init__(self, mlp_layer, rank_size):
        super(LoRAMLP, self).__init__()
        self.layers = nn.ModuleDict({
            'gate_proj': LoRALayer(mlp_layer.gate_proj, rank_size),
            'down_proj': LoRALayer(mlp_layer.down_proj, rank_size),
            'up_proj': LoRALayer(mlp_layer.up_proj, rank_size),
        })


    def forward(self, x):
        # Implement the forward pass for the LoRAMLP
        x = self.layers['gate_proj'](x)
        x = self.layers['down_proj'](x)
        x = self.layers['up_proj'](x)
        return x
    
def replace_mlp_with_lora(model, rank_size, layers_to_replace='all'):
    """
    Replace MLP submodules in a nested Transformer-based model with LoRALinear layers.

    Parameters:
    - model: The input pretrained model (nn.Module).
    - rank_size: Integer defining the rank of the low-rank approximation.
    - layers_to_replace: 'all' for replacing all MLP layers, or list of specific layer indices.
    
    Returns:
    - Modified model with LoRALinear modules.
    """
    for layer_idx, layer in enumerate(model.layers):
        if layer_idx in layers_to_replace or layers_to_replace == 'all':
            setattr(layer, 'mlp', LoRAMLP(layer.mlp, rank_size))

    torch.compile(model) # NOTE: not sure if we should keep this
    
    return model