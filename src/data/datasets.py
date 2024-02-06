#File for dataloader

from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


# This file should create the dataset that will be used to train the model.
#This includes the transcriptions in txt, the mouth images in npz and the keypoints in npz

#The structure of the dataset is as follows:
#inside this folder: "/data3fast/users/group02/videos/tracks/"
#we have the following structure:
#- folder containing:
#   - transcription.txt
#   - face_frames.npz
#   - keypoints.npz

class LipReadingDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.load_data()
        
    def load_data(self):
        for file in os.listdir(self.root_dir):
            #Get the transcription
            if file == 'transcription.txt':
                transcription_path = os.path.join(self.root_dir, file)
                with open(transcription_path, 'r') as file:
                    transcription = file.read() #This is the transcription
            #Get the face frames
            elif file == 'face_frames.npz':
                face_frames_path = os.path.join(self.root_dir, file)
                face_frames = np.load(face_frames_path)
            #Get the keypoints
            elif file == 'keypoints.npz':
                keypoints_path = os.path.join(self.root_dir, file)
                keypoints = np.load(keypoints_path)
        self.data.append((transcription, face_frames, keypoints))
        return self.data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
#Example of how to use the dataset
if __name__ == "__main__":
    dataset = LipReadingDataset("/data3fast/users/group02/videos/tracks/")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for transcription, face_frames, keypoints in dataloader:
        print(transcription, face_frames, keypoints)
        break