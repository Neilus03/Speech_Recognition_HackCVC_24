#File for dataloader

    
# This file should create the dataset that will be used to train the model.
#This includes the transcriptions in txt, the mouth images in npz and the keypoints in npz

#The structure of the dataset is as follows:
#inside this folder: "/data3fast/users/group02/videos/tracks/"
#we have the following structure:
#- folder containing:
#   - transcription.txt
#   - faces.npz
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
        # Each subdirectory in root_dir contains one set of transcription, faces, and keypoints
        for subdir in next(os.walk(self.root_dir))[1]: 
            dir_path = os.path.join(self.root_dir, subdir)
            transcription_path = os.path.join(dir_path, 'transcription.txt')
            #faces_path = os.path.join(dir_path, 'faces.npz')
            #keypoints_path = os.path.join(dir_path, 'keypoints.npz')
            
            if os.path.exists(transcription_path):# and os.path.exists(faces_path) and os.path.exists(keypoints_path):
                with open(transcription_path, 'r') as file:
                    transcription = file.read()
                    
                #faces = np.load(faces_path, allow_pickle=True)['arr_0'] # The npz file contains a list of frames 
                #keypoints = np.load(keypoints_path, allow_pickle=True)['arr_0']
                
                self.data.append({
                    'transcription': transcription,
                    #'faces': faces,
                    'words':  list(transcription.split()), #but this are the words and we need the letters so we will change it to the letters in the next line 
                    'tokens': list(transcription) #this is the list of letters of the transcription but padding is needed
                    #'keypoints': keypoints
                })
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return a dictionary for each item in the dataset
        item = self.data[idx]
        tokens = item['tokens']
        print(tokens)
        return {
            'transcription': item['transcription'],
            #'faces': item['faces'],
            'tokens': tokens,
            "words": item['words']
            #'keypoints': item['keypoints']
        }


# Example of how to use the dataset
if __name__ == "__main__":
    mock_data_folder = "/home/GROUP02/Speech_Recognition_HackCVC_24/src/data/t/"
    real_data_folder = "/data3fast/users/group02/videos/tracks/"
    
    dataset = LipReadingDataset(real_data_folder)
    print(len(dataset))
