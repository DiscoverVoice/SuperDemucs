import torch
from fractions import Fraction
from scipy.stats import norm
from einops import rearrange
from torch.nn import functional as F
from demucs.apply import tensor_chunk, TensorChunk


def set_parameters(module, weight, bias):
    module.weight = torch.nn.Parameter(weight)
    module.bias = torch.nn.Parameter(bias)

class WeightRemover:
    def __init__(self, model, p=0.9):
        self.source_model = model
        self.p = p

    def remove_edge(self, module, output):
        """
        Positive hook
        Attributes:
            module (Layer): custom layer
            output (torch.Tensor): output tensor of the original layer
        """
        # Get the shapes and original parameters (weights and biases) of the layer
        current_weight, current_bias = (
            module.weight.clone(),
            module.bias.clone(),
        )  # updating parameters
        shape = current_weight.shape

        mean = torch.mean(current_weight, dim=1, keepdim=True)
        std = torch.std(current_weight, dim=1, keepdim=True)
        z_scores = (current_weight - mean) / std

        lower_z, upper_z = norm.ppf(0.45), norm.ppf(0.55)

        mask = torch.logical_and(z_scores >= lower_z, z_scores < upper_z)

        current_weight[mask] = 0
        all_zeros = ~mask.any(dim=1)
        current_bias[all_zeros] = 0
        set_parameters(module, current_weight, current_bias)

    def propagate(self, model, mix):
        length = mix.shape[-1]
        length_pre_pad = None
        if model.use_train_segment:
            if model.training:
                model.segment = Fraction(mix.shape[-1], model.samplerate)
            else:
                training_length = int(model.segment * model.samplerate)
                if mix.shape[-1] < training_length:
                    length_pre_pad = mix.shape[-1]
                    mix = F.pad(mix, (0, training_length - length_pre_pad))
        z = model._spec(mix)
        mag = model._magnitude(z).to(mix.device)
        x = mag

        B, C, Fq, T = x.shape

        # unlike previous Demucs, we always normalize because it is easier.
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)
        # x will be the freq. branch input.

        # Prepare the time branch input.
        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # okay, this is a giant mess I know...
        saved = []  # skip connections, freq.
        saved_t = []  # skip connections, time.
        lengths = []  # saved lengths to properly remove padding, freq branch.
        lengths_t = []  # saved lengths for time branch.
        for idx, encode in enumerate(model.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(model.tencoder):
                # we have not yet merged branches.
                lengths_t.append(xt.shape[-1])
                tenc = model.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    # save for skip connection
                    saved_t.append(xt)
                else:
                    # tenc contains just the first conv., so that now time and freq.
                    # branches have the same shape and can be merged.
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and model.freq_emb is not None:
                # add frequency embedding to allow for non equivariant convolutions
                # over the frequency axis.
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = model.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + model.freq_emb_scale * emb

            saved.append(x)
        if model.crosstransformer:
            if model.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = model.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = model.channel_upsampler_t(xt)

            x, xt = model.crosstransformer(x, xt)

            if model.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = model.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = model.channel_downsampler_t(xt)

        for idx, decode in enumerate(model.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))
            # `pre` contains the output just before final transposed convolution,
            # which is used when the freq. and time branch separate.

            offset = model.depth - len(model.tdecoder)
            if idx >= offset:
                tdec = model.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        # Let's make sure we used all stored skip connections.
        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        S = len(model.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        # to cpu as mps doesnt support complex numbers
        # demucs issue #435 ##432
        # NOTE: in this case z already is on cpu
        # TODO: remove this when mps supports complex numbers
        x_is_mps_xpu = x.device.type in ["mps", "xpu"]
        x_device = x.device
        if x_is_mps_xpu:
            x = x.cpu()

        zout = model._mask(z, x)
        if model.use_train_segment:
            if model.training:
                x = model._ispec(zout, length)
            else:
                x = model._ispec(zout, training_length)
        else:
            x = model._ispec(zout, length)

        # back to mps device
        if x_is_mps_xpu:
            x = x.to(x_device)

        if model.use_train_segment:
            if model.training:
                xt = xt.view(B, S, -1, length)
            else:
                xt = xt.view(B, S, -1, training_length)
        else:
            xt = xt.view(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]
        x = xt + x
        if length_pre_pad:
            x = x[..., :length_pre_pad]
        return x