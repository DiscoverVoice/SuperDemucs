import time
import librosa
from tqdm import tqdm
import os
import glob
import torch
import numpy as np
import soundfile as sf
import torch.nn as nn
from utils.audio_utils import demix_track, demix_track_demucs
from utils.model_utils import get_model_from_config

import warnings
warnings.filterwarnings("ignore")


def run_folder(model, input_folder, store_dir, config, device, model_type, extract_instrumental=False, verbose=False):
    start_time = time.time()
    model.eval()
    all_mixtures_path = glob.glob(input_folder + '/*.*')
    print('Total files found: {}'.format(len(all_mixtures_path)))

    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]

    if not os.path.isdir(store_dir):
        os.mkdir(store_dir)

    if not verbose:
        all_mixtures_path = tqdm(all_mixtures_path)

    for path in all_mixtures_path:
        if not verbose:
            all_mixtures_path.set_postfix({'track': os.path.basename(path)})
        try:
            mix, sr = librosa.load(path, sr=44100, mono=False)
            mix = mix.T
        except Exception as e:
            print('Can read track: {}'.format(path))
            print('Error message: {}'.format(str(e)))
            continue

        # Convert mono to stereo if needed
        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis=-1)

        mixture = torch.tensor(mix.T, dtype=torch.float32)
        if model_type == 'htdemucs':
            res = demix_track_demucs(config, model, mixture, device)
        else:
            res = demix_track(config, model, mixture, device)
        for instr in instruments:
            sf.write("{}/{}_{}.wav".format(store_dir, os.path.basename(path)[:-4], instr), res[instr].T, sr, subtype='FLOAT')

        if 'vocals' in instruments and extract_instrumental:
            instrum_file_name = "{}/{}_{}.wav".format(store_dir, os.path.basename(path)[:-4], 'instrumental')
            sf.write(instrum_file_name, mix - res['vocals'].T, sr, subtype='FLOAT')

    time.sleep(1)
    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))


def proc_folder(model_type, config_path, start_check_point, input_folder, store_dir, device_ids, extract_instrumental):
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config(model_type, config_path)
    if start_check_point != '':
        print('Start from checkpoint: {}'.format(start_check_point))
        state_dict = torch.load(start_check_point)
        if model_type == 'htdemucs':
            if 'state' in state_dict:
                state_dict = state_dict['state']
        model.load_state_dict(state_dict)
    print("Instruments: {}".format(config.training.instruments))

    if torch.cuda.is_available():
        if type(device_ids) == int:
            device = torch.device(f'cuda:{device_ids}')
            model = model.to(device)
        else:
            device = torch.device(f'cuda:{device_ids[0]}')
            model = nn.DataParallel(model, device_ids=device_ids).to(device)
    else:
        device = 'cpu'
        print('CUDA is not available. Run inference on CPU. It will be very slow...')
        model = model.to(device)

    run_folder(model, input_folder, store_dir, config, device, model_type, extract_instrumental, verbose=False)


if __name__ == "__main__":
    # Example of calling the proc_folder function directly with parameters
    model_type = 'mdx23c'
    config_path = 'path/to/config/file'
    start_check_point = 'path/to/checkpoint'
    input_folder = 'path/to/input/folder'
    store_dir = 'path/to/store/results'
    device_ids = [0]  # List of GPU ids, or a single int for one GPU
    extract_instrumental = True

    proc_folder(model_type, config_path, start_check_point, input_folder, store_dir, device_ids, extract_instrumental)