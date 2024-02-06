#File for dataloader

    
# This file should create the dataset that will be used to train the model.
#This includes the transcriptions in txt, the mouth images in npz and the keypoints in npz

#The structure of the dataset is as follows:
#inside this folder: "/data3fast/users/group02/videos/tracks/"
#we have the following structure:
#- folder containing:
#   - transcription.txt
#   - face_frames.npz
#   - keypoints.npz
    
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class LipReadingDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.load_data()
        
    def load_data(self):
        # Each subdirectory in root_dir contains one set of transcription, face_frames, and keypoints
        for subdir in next(os.walk(self.root_dir))[1]: 
            dir_path = os.path.join(self.root_dir, subdir)
            transcription_path = os.path.join(dir_path, 'transcription.txt')
            face_frames_path = os.path.join(dir_path, 'face_frames.npz')
            keypoints_path = os.path.join(dir_path, 'keypoints.npz')
            
            if os.path.exists(transcription_path) and os.path.exists(face_frames_path) and os.path.exists(keypoints_path):
                with open(transcription_path, 'r') as file:
                    transcription = file.read()
                face_frames = np.load(face_frames_path, allow_pickle=True)['arr_0'] # The npz file contains a list of frames 
                keypoints = np.load(keypoints_path, allow_pickle=True)['arr_0']
                
                self.data.append({
                    'transcription': transcription,
                    'face_frames': face_frames,
                    'keypoints': keypoints
                })

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return a dictionary for each item in the dataset
        return {
            'transcription': self.data[idx]['transcription'], # This is the transcription txt
            'face_frames': self.data[idx]['face_frames'], # This is the face_frames npz	
            'keypoints': self.data[idx]['keypoints'] # This is the keypoints npz	
        }

# Example of how to use the dataset
if __name__ == "__main__":
    mock_data_folder = "/home/GROUP02/Speech_Recognition_HackCVC_24/src/data/t/"
    
    dataset = LipReadingDataset(mock_data_folder)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for data in dataloader:
        print(data['transcription'], data['face_frames'], data['keypoints'])
        break
