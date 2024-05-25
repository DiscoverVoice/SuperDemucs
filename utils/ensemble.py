import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment

def stft(wave, nfft, hl):
    wave_left = np.asfortranarray(wave[0])
    wave_right = np.asfortranarray(wave[1])
    spec_left = librosa.stft(wave_left, n_fft=nfft, hop_length=hl)
    spec_right = librosa.stft(wave_right, n_fft=nfft, hop_length=hl)
    spec = np.asfortranarray([spec_left, spec_right])
    return spec

def istft(spec, hl, length):
    spec_left = np.asfortranarray(spec[0])
    spec_right = np.asfortranarray(spec[1])
    wave_left = librosa.istft(spec_left, hop_length=hl, length=length)
    wave_right = librosa.istft(spec_right, hop_length=hl, length=length)
    wave = np.asfortranarray([wave_left, wave_right])
    return wave

def absmax(a, *, axis):
    dims = list(a.shape)
    dims.pop(axis)
    indices = np.ogrid[tuple(slice(0, d) for d in dims)]
    argmax = np.abs(a).argmax(axis=axis)
    indices.insert((len(a.shape) + axis) % len(a.shape), argmax)
    return a[tuple(indices)]

def absmin(a, *, axis):
    dims = list(a.shape)
    dims.pop(axis)
    indices = np.ogrid[tuple(slice(0, d) for d in dims)]
    argmax = np.abs(a).argmin(axis=axis)
    indices.insert((len(a.shape) + axis) % len(a.shape), argmax)
    return a[tuple(indices)]

def lambda_max(arr, axis=None, key=None, keepdims=False):
    idxs = np.argmax(key(arr), axis)
    if axis is not None:
        idxs = np.expand_dims(idxs, axis)
        result = np.take_along_axis(arr, idxs, axis)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result
    else:
        return arr.flatten()[idxs]

def lambda_min(arr, axis=None, key=None, keepdims=False):
    idxs = np.argmin(key(arr), axis)
    if axis is not None:
        idxs = np.expand_dims(idxs, axis)
        result = np.take_along_axis(arr, idxs, axis)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result
    else:
        return arr.flatten()[idxs]

def average_waveforms(pred_track, weights, algorithm):
    pred_track = np.array(pred_track)
    final_length = pred_track.shape[-1]

    mod_track = []
    for i in range(pred_track.shape[0]):
        if algorithm == 'avg_wave':
            mod_track.append(pred_track[i] * weights[i])
        elif algorithm in ['median_wave', 'min_wave', 'max_wave']:
            mod_track.append(pred_track[i])
        elif algorithm in ['avg_fft', 'min_fft', 'max_fft', 'median_fft']:
            spec = stft(pred_track[i], 2048, 1024)
            if algorithm in ['avg_fft']:
                mod_track.append(spec * weights[i])
            else:
                mod_track.append(spec)
    pred_track = np.array(mod_track)

    if algorithm in ['avg_wave']:
        pred_track = pred_track.sum(axis=0)
        pred_track /= np.array(weights).sum().T
    elif algorithm in ['median_wave']:
        pred_track = np.median(pred_track, axis=0).T
    elif algorithm in ['min_wave']:
        pred_track = np.array(pred_track)
        pred_track = lambda_min(pred_track, axis=0, key=np.abs).T
    elif algorithm in ['max_wave']:
        pred_track = np.array(pred_track)
        pred_track = lambda_max(pred_track, axis=0, key=np.abs).T
    elif algorithm in ['avg_fft']:
        pred_track = pred_track.sum(axis=0)
        pred_track /= np.array(weights).sum()
        pred_track = istft(pred_track, 1024, final_length).T
    elif algorithm in ['min_fft']:
        pred_track = np.array(pred_track)
        pred_track = lambda_min(pred_track, axis=0, key=np.abs)
        pred_track = istft(pred_track, 1024, final_length).T
    elif algorithm in ['max_fft']:
        pred_track = np.array(pred_track)
        pred_track = absmax(pred_track, axis=0)
        pred_track = istft(pred_track, 1024, final_length).T
    elif algorithm in ['median_fft']:
        pred_track = np.array(pred_track)
        pred_track = np.median(pred_track, axis=0)
        pred_track = istft(pred_track, 1024, final_length).T
    return pred_track

def ensemble_files(file_pairs, algorithm='avg_wave', output_dir="output", output_type='.wav'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for file_pair in tqdm(file_pairs, desc="Processing files"):
        files = file_pair['files']
        output_name = file_pair['output']
        
        data = []
        for f in files:
            if not os.path.isfile(f):
                print(f"Error. Can't find file: {f}. Check paths.")
                continue
            try:
                wav, sr = librosa.load(f, sr=None, mono=False)
                data.append(wav)
            except Exception as e:
                print(f"Error reading file {f}: {e}")
        
        if len(data) == 0:
            print(f"No valid files found for {output_name}. Skipping...")
            continue

        weights = np.ones(len(data))
        res = average_waveforms(data, weights, algorithm)

        output_file = output_path / (output_name.replace('.wav', output_type))

        try:
            if output_type == '.mp3':
                temp_wav = output_path / (output_name.replace('.wav', '_temp.wav'))
                sf.write(temp_wav, res.T, sr, format='WAV')
                audio = AudioSegment.from_wav(temp_wav)
                audio.export(output_file, format='mp3')
                os.remove(temp_wav)
            else:
                sf.write(output_file, res.T, sr, format='WAV')
        except Exception as e:
            print(f"Error writing file {output_file}: {e}")

def get_matching_file_pairs(root_dirs, audio_type='.wav'):
    files_dict = {}
    for root_dir in root_dirs:
        for path in Path(root_dir).rglob(f'*{audio_type}'):
            filename = path.name
            if filename not in files_dict:
                files_dict[filename] = []
            files_dict[filename].append(str(path))
    
    file_pairs = [{'files': paths, 'output': filename} for filename, paths in files_dict.items() if len(paths) > 1]
    return file_pairs

