import whisper
model = whisper.load_model("base")

result = model.transcribe("/data3fast/users/group02/videos/The Fear of Success Dr Marty Hauff at TEDxYouth@WISS_trimmed.mp3")
print(result["text"])