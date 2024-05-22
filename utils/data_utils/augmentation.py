import torch
import random
import torchaudio

class AudioDataAugmentaion:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def time_stretch(self, waveform, rate=1.0):
        return torchaudio.transforms.TimeStretch()(waveform.unsqueeze(0), rate).squeeze(0)

    def pitch_shift(self, waveform, n_steps):
        return torchaudio.transforms.PitchShift(sample_rate=self.sample_rate, n_steps=n_steps)(waveform)

    def add_noise(self, waveform, noise_factor=0.005):
        noise = torch.randn(waveform.size()) * noise_factor
        noisy_waveform = waveform + noise
        return noisy_waveform

    def flip(self, waveform):
        return torch.flip(waveform, dims=[-1])

    def scale(self, waveform, scale_factor=1.0):
        return waveform * scale_factor

    def remix(self, waveform1, waveform2):
        if waveform1.shape != waveform2.shape:
            raise ValueError("Input waveforms must have the same shape")
        return waveform1 + waveform2

    def random_time_stretch(self, waveform):
        rate = random.uniform(0.8, 1.2)
        return self.time_stretch(waveform, rate)

    def random_pitch_shift(self, waveform):
        n_steps = random.randint(-5, 5)
        return self.pitch_shift(waveform, n_steps)

    def random_add_noise(self, waveform):
        noise_factor = random.uniform(0.001, 0.01)
        return self.add_noise(waveform, noise_factor)

    def random_scale(self, waveform):
        scale_factor = random.uniform(0.7, 1.3)
        return self.scale(waveform, scale_factor)

    def random_flip(self, waveform):
        if random.random() < 0.5:
            return self.flip(waveform)
        return waveform

    def augment(self, waveform):
        aug_waveform = waveform.clone()
        if random.random() < 0.5:
            aug_waveform = self.random_time_stretch(aug_waveform)
        if random.random() < 0.5:
            aug_waveform = self.random_pitch_shift(aug_waveform)
        if random.random() < 0.5:
            aug_waveform = self.random_add_noise(aug_waveform)
        if random.random() < 0.5:
            aug_waveform = self.random_scale(aug_waveform)
        if random.random() < 0.5:
            aug_waveform = self.random_flip(aug_waveform)
        return aug_waveform