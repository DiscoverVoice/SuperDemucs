
import torch
from scipy.stats import norm

class WeightRemover:
    def __init__(self, model, p=0.9):
        self.source_model = model
        self.p = p

    def propagate(self, ref_layer, target_layer, input_tensor):
        def hook(layer, input, output):
            current_weight, current_bias = target_layer.weight, target_layer.bias
            original_outputs = ref_layer(input[0])

            if torch.sum(current_weight != 0) > torch.numel(current_weight) * self.p:
                self.remove(layer, output)
        handle = target_layer.register_forward_hook(hook)
        out = target_layer(input_tensor)
        handle.remove()
        return out

    def remove(self, layer, output):
        current_weight, current_bias = layer.weight.clone(), layer.bias.clone()
        shape = current_weight.shape

        mean = torch.mean(current_weight, dim=1, keepdim=True)
        std = torch.std(current_weight, dim=1, keepdim=True)
        z_scores = (current_weight - mean) / std

        lower_z, upper_z = norm.ppf(0.45), norm.ppf(0.55)
        mask = torch.logical_and(z_scores >= lower_z, z_scores < upper_z)

        current_weight[mask] = 0
        all_zeros = ~mask.any(dim=1)
        current_bias[all_zeros] = 0
        set_parameters(layer, current_weight, current_bias)