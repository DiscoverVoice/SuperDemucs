import torch
from scipy.stats import norm


class WeightRemover:
    def __init__(self, model, device="cuda:0", p=0.8):
        self.model = model.to(device)
        self.device = device
        self.p = p
        self.results = {"layer": [], "input": [], "output": []}

    def hook(self, layer, input, output):
        self.results["layer"].append(layer)
        self.results["input"].append(input[0].to('cpu'))
        self.results["output"].append(output[0].to('cpu'))

    def register_hooks(self):
        handle_list = []
        for layer in self.model.modules():
            if isinstance(layer, torch.nn.Linear):
                handle = layer.register_forward_hook(self.hook)
                handle_list.append(handle)
        return handle_list

    def remove_hooks(self, handle_list):
        for handle in handle_list:
            handle.remove()

    def remove_weights(self, layer):
        current_weight = layer.weight.clone()
        if layer.bias is not None:
            current_bias = layer.bias.clone()
        else:
            current_bias = None

        mean = torch.mean(current_weight, dim=1, keepdim=True)
        std = torch.std(current_weight, dim=1, keepdim=True)
        z_scores = (current_weight - mean) / std

        lower_z, upper_z = norm.ppf(0.45), norm.ppf(0.55)
        mask = torch.logical_and(z_scores >= lower_z, z_scores < upper_z)

        current_weight[mask] = 0
        all_zeros = ~mask.any(dim=1)
        if current_bias is not None:
            current_bias[all_zeros] = 0
        self.set_parameters(layer, current_weight, current_bias)

    def set_parameters(self, layer, weight, bias):
        layer.weight.data = weight
        if bias is not None:
            layer.bias.data = bias

    def process(self, input_tensor):
        self.results = {"layer": [], "input": [], "output": []}
        handle_list = self.register_hooks()
        output = self.model(input_tensor.to(self.device))
        self.remove_hooks(handle_list)
        return output

    def apply_removal(self):
        for idx, layer in enumerate(self.results["layer"]):
            current_weight = layer.weight
            if torch.sum(current_weight != 0) > torch.numel(current_weight) * self.p:
                print(f"before {torch.sum(layer.weight != 0)}")
                self.results["output"][idx] = self.results["output"][idx].to(self.device)
                self.remove_weights(layer)
                self.results["output"][idx] = self.results["output"][idx].to('cpu')
                print(f"after {torch.sum(layer.weight != 0)}")
