#Collator function to collate the data from the data source

#Import the necessary libraries
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn 
from torchvision import transforms
from src.learning.text_tokenizer.tokenizers import CharTokenizer

class LipReadingCollator:
    def __init__(self, face_transforms = None, lip_keypoints_transforms = None, tokenizer = None):
        self.face_transforms = face_transforms
        self.lip_keypoints_transforms = lip_keypoints_transforms
        self.tokenizer = tokenizer
        return 
    

    def collate_fn(self, batch): 
        '''
        This function will take in a batch of data and collate it into a single tensor.
        The batch is a list of dictionaries, where each dictionary contains the keys 'transcription', 'face_frames', and 'lip_keypoints'.
        The collate function should return a dictionary with the same keys, but the values should be a single tensor, each with the 
        proper transformation applied and properly tokenized.
        
        The key 'transcription' should be tokenized using the tokenizer, the key 'face_frames' should be transformed using the 
        transforms and then stacked into a single tensor. The key 'lip_keypoints' should be transformed using the transforms and
        then stacked into a single tensor.
        '''
        transcriptions = []  # To store tokenized transcriptions
        text_tokens = []
        #face_frames = []     # To store transformed face frames
        #lip_keypoints = []       # To store transformed lip_keypoints   
        max_tokens = max([len(sample['transcription']) for sample in batch]) # Get the maximum length of the transcriptions in the batch
        
        
        # Iterate through each sample in the batch
        for sample in batch:
            
            text_token = sample['tokens']
            
            #face_frames = sample['face_frames']
            #lip_keypoints = sample['lip_keypoints']
            transcription = sample['transcription']
            
                
            text_tokenized = torch.from_numpy(
                self.tokenizer(
                    text_token + [self.tokenizer.padding_token] * (max_tokens - len(text_token))
                )
            )
            print(text_tokenized)
            text_tokens.append(text_tokenized)
            
            '''# Tokenize the transcription using the provided tokenizer (if available)
            if self.tokenizer:
                transcription = self.tokenizer(sample['transcription'])
                max_len = max(max_len, len(transcription))
                transcription = transcription + ['<PAD>'] * (max_len - len(transcription))
                print(transcription)
            else:
                transcription = sample['transcription']
                #transcription = torch.tensor(transcription)'''
            
            # Apply transforms to face frames (if transforms are provided)
            #if self.face_transforms:
                #transformed_face_frames = self.face_transforms(sample['face_frames'])
                    
            #else:
                #transformed_face_frames = sample['face_frames']
             
            #Apply transforms to lip_keypoints (if transforms are provided)   
            #if self.lip_keypoints_transforms:
                #transformed_lip_keypoints = self.lip_keypoints_transforms(sample['lip_keypoints'])
            #else:
                #transformed_lip_keypoints = sample['lip_keypoints']

            # Append tokenized transcription tensor, transformed face frames, and lip_keypoints to respective lists
            #face_frames.append(transformed_face_frames)
            #lip_keypoints.append(transformed_lip_keypoints)

        # Stack the lists of transcriptions, face frames, and lip_keypoints into tensors
        text_tokens = torch.stack(text_tokens)
        #face_frames = torch.stack(face_frames)
        #lip_keypoints = torch.stack(lip_keypoints)

        # Create a dictionary with the collated data
        collated_data = {
            'text_tokens': text_tokens,
            #'face_frames': face_frames,
            #'lip_keypoints': lip_keypoints
        }
        
        return collated_data

     
    def __call__(self, batch): #__call__ method is used to call the instance of the class and it allows the class to be called as a function
        return self.collate_fn(batch)
    
    def __repr__(self): #__repr__ method is used to return a string representation of the object
        return f"LipReadingCollator(face transforms={self.face_transforms}, lip_keypoints_transforms={self.lip_keypoints_transforms}, tokenizer={self.tokenizer})"
        
        
if __name__ == "__main__":
    # Example usage of the LipReadingCollator
    # Create a mock dataset
    class MockDataset(Dataset):
        def __init__(self):
            self.data = [
                {
                    'transcription': "hello",
                    'tokens': ['h', 'e', 'l', 'l', 'o'],
                    #'face_frames': np.random.rand(10, 3, 64, 64),
                    #'lip_keypoints': np.random.rand(10, 68, 2)
                },
                {
                    'transcription': "word",
                    'tokens': ['w', 'o', 'r', 'd'],
                    #'face_frames': np.random.rand(10, 3, 64, 64),
                    #'lip_keypoints': np.random.rand(10, 68, 2)
                }
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Create a mock dataset
    dataset = MockDataset()

    # Create a collator
    collator = LipReadingCollator(tokenizer=CharTokenizer())

    # Create a DataLoader using the collator
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator, shuffle=True)

    # Iterate through the DataLoader
    for batch in dataloader:
        print(batch)
        break