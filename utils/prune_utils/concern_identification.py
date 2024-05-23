import torch
from scipy.stats import norm


class ConcernIdentification:
    def __init__(self, ref_model, model, device='cuda:0', p=0.7):
        self.ref_model = ref_model.to(device)
        self.model = model.to(device)
        self.device = device
        self.p = p
        self.original_results = {"layer": [], "input": [], "output": []}
        self.current_results = {"layer": [], "input": [], "output": []}

    def original_hook(self, layer, input, output):
        self.original_results["layer"].append(layer)
        self.original_results["input"].append(input[0].to('cpu'))
        self.original_results["output"].append(output[0].to('cpu'))

    def current_hook(self, layer, input, output):
        self.current_results["layer"].append(layer)
        self.current_results["input"].append(input[0].to('cpu'))
        self.current_results["output"].append(output[0].to('cpu'))

    def register_hooks(self, model, hook):
        handle_list = []
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                handle = layer.register_forward_hook(hook)
                handle_list.append(handle)
        return handle_list

    def remove_hooks(self, handle_list):
        for handle in handle_list:
            handle.remove()

    def prune(self, ref_model, model, original_output, output):
        current_weight = model.weight.clone()
        if model.bias is not None:
            current_bias = model.bias.clone()
        else:
            current_bias = None
        original_weight = ref_model.weight.clone()
        
        if ref_model.bias is not None:
            original_bias = ref_model.bias.clone()
        else:
            original_bias = None
        shape = current_weight.shape

        output_loss = output - original_output
        if len(output_loss.shape) > len(shape):
            output_loss = output_loss[:, 0, :]
            
        positive_loss_mask = (
            torch.all(output_loss > 0, dim=0).unsqueeze(1).expand(-1, shape[1])
        )

        original_weight_std = safe_std(original_weight, dim=1, keepdim=True)
        current_weight_std = safe_std(
            current_weight,
            epsilon=original_weight_std,
            unbiased=True,
            dim=1,
            keepdim=True,
        )

        padded_positive = torch.where(
            current_weight > 0, current_weight, torch.tensor(float("nan"))
        )
        padded_negative = torch.where(
            current_weight < 0, current_weight, torch.tensor(float("nan"))
        )
        positive_mean = torch.nanmean(padded_positive, dim=1, keepdim=True)
        negative_mean = torch.nanmean(padded_negative, dim=1, keepdim=True)

        positive_std = safe_std(
            current_weight,
            epsilon=current_weight_std,
            unbiased=True,
            dim=1,
            keepdim=True,
        )
        negative_std = safe_std(
            current_weight,
            epsilon=current_weight_std,
            unbiased=True,
            dim=1,
            keepdim=True,
        )

        positive_scores = (padded_positive - positive_mean) / positive_std
        negative_scores = (padded_negative - negative_mean) / negative_std

        positive_median = torch.nanmedian(padded_positive, dim=1, keepdim=True)
        negative_median = torch.nanmedian(padded_negative, dim=1, keepdim=True)
        lower_z, upper_z = norm.ppf(0.1), norm.ppf(0.3)

        positive_remove_mask = torch.where(
            positive_mean < positive_median.values,
            positive_scores <= lower_z,
            torch.logical_and(positive_scores >= lower_z, positive_scores < upper_z),
        )

        negative_remove_mask = torch.where(
            negative_mean < negative_median.values,
            torch.logical_and(negative_scores < -lower_z, negative_scores >= -upper_z),
            negative_scores >= -upper_z,
        )

        remove_mask = torch.where(
            ~positive_loss_mask, positive_remove_mask, negative_remove_mask
        )

        current_weight[remove_mask] = 0

        all_zeros = ~remove_mask.any(dim=1)
        if current_bias is not None:
            current_bias[all_zeros] = 0
        self.set_parameters(model, current_weight, current_bias)

    def set_parameters(self, layer, weight, bias):
        layer.weight.data = weight
        if bias is not None:
            layer.bias.data = bias

    def process(self, input_tensor):
        self.original_results = {"layer": [], "input": [], "output": []}
        self.current_results = {"layer": [], "input": [], "output": []}

        handle_list = self.register_hooks(self.model, self.current_hook)
        self.model(input_tensor.to(self.device))
        self.remove_hooks(handle_list)
        handle_list = self.register_hooks(self.ref_model, self.original_hook)
        self.ref_model(input_tensor.to(self.device))
        self.remove_hooks(handle_list)

    def apply_prune(self):
        for idx, layer in enumerate(self.current_results["layer"]):
            current_weight = layer.weight
            if torch.sum(current_weight != 0) > torch.numel(current_weight) * self.p:
                print(f"before {torch.sum(layer.weight != 0)}")

                self.original_results["output"][idx] = self.original_results["output"][idx].to(self.device)
                self.current_results["output"][idx] = self.current_results["output"][idx].to(self.device)
                self.prune(self.original_results["layer"][idx], layer, self.original_results["output"][idx],
                           self.current_results["output"][idx])
                self.original_results["output"][idx] = self.original_results["output"][idx].to('cpu')
                self.current_results["output"][idx] = self.current_results["output"][idx].to('cpu')
                print(f"after {torch.sum(layer.weight != 0)}")

    def set_parameters(self, layer, weight, bias):
        layer.weight.data = weight
        layer.bias.data = bias

def safe_std(tensor, epsilon=None, unbiased=False, dim=None, keepdim=True):
    if tensor.numel():
        return nanstd(tensor, dim=dim, unbiased=unbiased, keepdim=keepdim)
    else:
        return torch.tensor(epsilon, dtype=tensor.dtype)


def nanstd(tensor, unbiased=False, dim=None, keepdim=True):
    mask = torch.isnan(tensor)
    n_obs = mask.logical_not().sum(dim=dim, keepdim=keepdim)
    mean = torch.nanmean(tensor, dim=dim, keepdim=keepdim)

    centered = tensor - mean
    centered = centered.masked_fill(mask, 0)
    sum_sq = torch.sum(centered ** 2, dim=dim, keepdim=keepdim)

    unbiased_factor = torch.where(n_obs > 1, n_obs - 1, n_obs)
    var = sum_sq / unbiased_factor

    std = torch.sqrt(var)
    if not keepdim:
        std = std.squeeze(dim)
    return std