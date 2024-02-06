
import torch
from pathlib import Path
import whisper
from multiprocessing import Pool

# Define a list of GPU device IDs to use
gpu_devices = ['cuda:7', 'cuda:8', 'cuda:9']
CUDA_VISIBLE_DEVICES = 7,8,9

def transcribe_audio(audio_path, transcription_path, device_id="cuda:0", model_size='base'):
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

        # Print the path where the transcription should be saved
        print(f"Transcription path: {transcription_path}")

        # Save the transcription to a file
        with open(transcription_path, 'w') as f:
            f.write(result['text'])
            
            print("transcription path:",transcription_path)
            print(result['text'])
            exit()
        return result['text']
    except Exception as e:
        return str(e)


def process_folder(folder):
    if folder.is_dir():
        audio_file = next(folder.glob("*.mp3"), None)
        if audio_file:
            transcription_path = folder / "transcription.txt"
            #print(f"Transcribing audio from {audio_file}...")

            # Transcribe audio to text using one of the specified GPUs
            device_id = "cuda:0" # Choose GPU based on folder depth
            transcribe_audio(str(audio_file), str(transcription_path), device_id=device_id)

            #print(f"Finished transcribing. Transcription saved to {transcription_path}.")
        else:
            #print(f"No MP3 file found in {folder}.")

if __name__ == "__main__":
    base_directory = Path("/data3fast/users/group02/videos/tracks")
    
    # Create a Pool of worker processes
    with Pool(processes=3) as pool:  # Adjust the number of processes as needed
        # Use pool.map to process folders in parallel
        pool.map(process_folder, base_directory.iterdir())
