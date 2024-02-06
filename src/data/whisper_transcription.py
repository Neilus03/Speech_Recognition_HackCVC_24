import torch
from pathlib import Path
import whisper

# CUDA device setup
device = ("cuda:1" if torch.cuda.is_available() else "cpu")

def transcribe_audio(audio_path, transcription_path, model_size='medium'):
    # Load the Whisper model and move to the specified device
    model = whisper.load_model(model_size).to(device)

    # Transcribe the audio
    result = model.transcribe(audio_path)
    
    # Save the transcription to a file
    with open(transcription_path, 'w') as f:
        f.write(result['text'])
        
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
            print(f"No MP3 file found in {folder}.")
            
    if count > 3:  # For testing purposes  
        break
