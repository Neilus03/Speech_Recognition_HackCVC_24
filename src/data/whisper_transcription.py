'''import torch
from pathlib import Path
from torch import nn
import whisper

def transcribe_audio(audio_path, transcription_path, model_size='base'):
    
    # cuda device
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    
    # Load the Whisper model and move to the specified device
    model = whisper.load_model(model_size).to(device)
    
    # Transcribe the audio
    result = model.transcribe(audio_path)
    
    # Save the transcription to a file
    with open(transcription_path, 'w') as f:
        f.write(result['text'])
        print(result['text'])
        
    return result['text']

base_directory = Path("/data3fast/users/group02/videos/tracks")
count = 0
for folder in base_directory.iterdir():
    count += 1
    if folder.is_dir():
        audio_file = next(folder.glob("*.mp3"), None)
        if audio_file:
            transcription_file = folder / "transcription.txt"
            print(f"Transcribing audio from {audio_file}...")

            # Transcribe audio to text
            transcribe_audio(str(audio_file), str(transcription_file))

            print(f"Finished transcribing. Transcription saved to {transcription_file}.")
        else:
            print(f"No MP3 file found in {folder}.")'''
            
            
import torch
from pathlib import Path
import whisper
from multiprocessing import Pool

# Define a list of GPU device IDs to use
gpu_devices = ['cuda:7', 'cuda:8', 'cuda:9']
CUDA_VISIBLE_DEVICES = 7,8,9

def transcribe_audio(audio_path, transcription_path, model_size='base', device_id=None):
    try:
        # Set the CUDA device if a specific device ID is provided
        if device_id is not None:
            device = torch.device(device_id)
        else:
            # Default to CPU if no device ID is provided
            device = torch.device('cpu')

        # Load the Whisper model and move it to the specified device
        model = whisper.load_model(model_size).to(device)

        # Transcribe the audio
        result = model.transcribe(audio_path)

        # Save the transcription to a file
        with open(transcription_path, 'w') as f:
            f.write(result['text'])
            #print(result['text'])
            
        return result['text']
    except Exception as e:
        return str(e)

def process_folder(folder):
    transcribed = 0
    if folder.is_dir():
        audio_file = next(folder.glob("*.mp3"), None)
        if audio_file:
            transcription_file = folder / "transcription.txt"
            print(f"Transcribing audio from {audio_file}...")

            # Transcribe audio to text using one of the specified GPUs
            device_id = gpu_devices[len(folder.parts) % len(gpu_devices)]  # Choose GPU based on folder depth
            transcribe_audio(str(audio_file), str(transcription_file), device_id=device_id)

            print(f"Finished transcribing. Transcription saved to {transcription_file}.")
        else:
            print(f"No MP3 file found in {folder}.")

if __name__ == "__main__":
    base_directory = Path("/data3fast/users/group02/videos/tracks")
    
    # Create a Pool of worker processes
    with Pool(processes=3) as pool:  # Adjust the number of processes as needed
        # Use pool.map to process folders in parallel
        pool.map(process_folder, base_directory.iterdir())

'''
import os
import torch
from pathlib import Path
import whisper
from multiprocessing import Pool, Manager

# Define a list of GPU device IDs to use
gpu_devices = ['cuda:7', 'cuda:8', 'cuda:9']

def transcribe_audio(audio_path, transcription_path,  device_id, model_size='base'):
    try:
        # Set the CUDA device
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        device = torch.device('cuda:7')

        # Load the Whisper model and move it to the specified device
        model = whisper.load_model(model_size).to(device)

        # Transcribe the audio
        result = model.transcribe(audio_path)

        # Save the transcription to a file
        with open(transcription_path, 'w') as f:
            f.write(result['text'])

        return result['text']
    except Exception as e:
        return str(e)

def process_folder(args):
    folder, transcribed_count = args
    transcribed = 0
    if folder.is_dir():
        audio_file = next(folder.glob("*.mp3"), None)
        if audio_file:
            transcription_file = folder / "transcription.txt"
            print(f"Transcribing audio from {audio_file}...")

            # Transcribe audio to text using one of the specified GPUs
            device_id = gpu_devices[len(folder.parts) % len(gpu_devices)]  # Choose GPU based on folder depth
            transcribe_audio(str(audio_file), str(transcription_file), device_id=device_id)
            
            transcribed += 1
            transcribed_count.value += 1
            print(f"Finished transcribing. Transcription saved to {transcription_file}.")
            print(f"Total audios transcribed: {transcribed_count.value}")

    return transcribed

if __name__ == "__main__":
    base_directory = Path("/data3fast/users/group02/videos/tracks")
    
    # Create a Pool of worker processes
    manager = Manager()
    transcribed_count = manager.Value('i', 0)
    
    with Pool(processes=3) as pool:  # Adjust the number of processes as needed
        # Use pool.map to process folders in parallel and get the list of transcribed counts
        transcribed_counts = pool.map(process_folder, [(folder, transcribed_count) for folder in base_directory.iterdir()])
        
    total_transcribed = sum(transcribed_counts)
    print(f"Total audios transcribed: {total_transcribed}")
'''