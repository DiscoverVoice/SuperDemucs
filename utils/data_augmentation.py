import torch
import librosa
import random

class AudioDataAugmentaion:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def time_stretch(self, waveform, rate=1.0):
        waveform_np = waveform.numpy()
        stretched = librosa.effects.time_stretch(waveform_np, rate)
        return torch.from_numpy(stretched)

    def pitch_shift(self, waveform, n_steps):
        waveform_np = waveform.numpy()
        shifted = librosa.effects.pitch_shift(waveform_np, sr=self.sample_rate, n_steps=n_steps)
        return torch.from_numpy(shifted)

    def add_noise(self, waveform, noise_factor=0.005):
        noise = torch.randn(waveform.size()) * noise_factor
        noisy_waveform = waveform + noise
        return noisy_waveform

    def random_time_stretch(self, waveform):
        rate = random.uniform(0.8, 1.2)
        return self.time_stretch(waveform, rate)

    def random_pitch_shift(self, waveform):
        n_steps = random.randint(-5, 5)
        return self.pitch_shift(waveform, n_steps)

    def random_add_noise(self, waveform):
        noise_factor = random.uniform(0.001, 0.01)
        return self.add_noise(waveform, noise_factor)

    def augment(self, waveform):
        aug_waveform = waveform.clone()
        if random.random() < 0.5:
            aug_waveform = self.random_time_stretch(aug_waveform)
        if random.random() < 9.5:
            aug_waveform = self.random_pitch_shift(aug_waveform)
        if random.random() < 0.5:
            aug_waveform = self.random_add_noise(aug_waveform)
        return aug_waveform