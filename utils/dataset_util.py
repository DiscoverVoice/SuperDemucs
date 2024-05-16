import musdb
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import sys

pwd = os.getcwd()
sys.path.append(os.path.dirname(pwd)) 
from SuperDemucs.utils.paths import p

class MUSDBDataset(Dataset):
    def __init__(self, root, subset='train', download=True, targets=['vocals', 'drums', 'bass', 'other']):
        self.musdb = musdb.DB(root=root, subsets=[subset], download=download)
        self.tracks = self.musdb.tracks
        self.targets = targets

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]
        
        audio_data = []
        mixture_audio = track.audio.T
        audio_data.append(torch.tensor(mixture_audio, dtype=torch.float32))
        
        for target in self.targets:
            target_audio = track.targets[target].audio.T
            audio_data.append(torch.tensor(target_audio, dtype=torch.float32))

        audio_data = torch.stack(audio_data, dim=0)
        sample_rate = track.rate
        num_samples = track.audio.shape[0]
        track_name = track.name
        
        return audio_data, sample_rate, num_samples, track_name

def load_musdb(download=False, batch_size=4, shuffle=True, val_split=0.1):
    train_dataset = MUSDBDataset(root=p.data_dir, subset='train', download=download)
    test_dataset = MUSDBDataset(root=p.data_dir, subset='test', download=download)

    train_subset, val_subset = random_split(train_dataset, [train_samples, val_samples])
    
    train_dataloader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader