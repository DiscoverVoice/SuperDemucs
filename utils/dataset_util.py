import musdb
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
from data_augmentation import AudioDataAugmentaion
from pydub import AudioSegment
import librosa
import numpy as np
import soundfile as sf

class MUSDBDataset(Dataset):
    def __init__(
        self,
        root,
        subset="train",
        download=True,
        targets=["vocals", "drums", "bass", "other"],
        sample_rate=441000,
        augment=False
    ):
        self.musdb = musdb.DB(root=root, subsets=[subset], download=download)
        self.tracks = self.musdb.tracks
        self.targets = targets
        self.sample_rate = sample_rate
        self.augment = augment
        self.augmenter = AudioDataAugmentaion(sample_rate) if augment else None

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]

        audio_data = []
        mixture_audio = track.audio.T
        audio_data.append(torch.tensor(mixture_audio, dtype=torch.float32))
        if self.augment:
            mixture_audio = self.augmenter.augment(mixture_audio)
        audio_data.append(mixture_audio)

        for target in self.targets:
            target_audio = track.targets[target].audio.T
            audio_data.append(torch.tensor(target_audio, dtype=torch.float32))
            if self.augment:
                target_audio = self.augmenter.augment(target_audio)
            audio_data.append(target_audio)

        audio_data = torch.stack(audio_data, dim=0)
        sample_rate = track.rate
        num_samples = track.audio.shape[0]
        track_name = track.name

        return audio_data, sample_rate, num_samples, track_name

    def resample(self, waveform):
        if waveform.shape[1] != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=waveform.shape[1], new_freq=self.sample_rate)
            waveform = resampler(waveform)
        return waveform


def load_musdb(config):
    root = config["musdb"]["root"]
    batch_size = config["musdb"]["batch_size"]
    shuffle = config["musdb"]["shuffle"]
    val_split = config["musdb"]["val_split"]
    num_workers = config["musdb"]["num_workers"]
    download = config["musdb"]["download"]

    train_dataset = MUSDBDataset(
        root=root,
        subset=config["data"]["train"],
        download=download,
        targets=config["data"]["targets"],
        augment=True
    )
    test_dataset = MUSDBDataset(
        root=root,
        subset=config["data"]["test"],
        download=download,
        targets=config["data"]["targets"],
        augment=False
    )

    total_samples = len(train_dataset)
    val_samples = len(total_samples * val_split)
    train_samples = total_samples - val_samples

    train_subset, val_subset = random_split(train_dataset, [train_samples, val_samples])

    train_dataloader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_dataloader, val_dataloader, test_dataloader


def mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, formaat="wav")

def load_audio(file_path):
    signal, sample_rate = sf.read(file_path)
    stft = np.abs(librosa.stft(signal, n_fft=1024, hop_length=512))
    return stft