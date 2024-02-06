import whisper
model = whisper.load_model("base")

result = model.transcribe("/data3fast/users/group02/videos/tracks/_3-ipSl6zfE/EP204 - Pimp My Setup_trimmed.mp3")
print(result["text"])