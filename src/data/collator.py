#Collator function to collate the data from the data source

#Import the necessary libraries
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn 
from torchvision import transforms


class LipReadingCollator:
    def __init__(self, face_transforms = None, keypoints_transforms = None, tokenizer = None):
        self.face_transforms = face_transforms
        self.keypoints_transforms = keypoints_transforms
        self.tokenizer = tokenizer
        return 
    

    def collate_fn(self, batch): 
        '''
        This function will take in a batch of data and collate it into a single tensor.
        The batch is a list of dictionaries, where each dictionary contains the keys 'transcription', 'faces', and 'keypoints'.
        The collate function should return a dictionary with the same keys, but the values should be a single tensor, each with the 
        proper transformation applied and properly tokenized.
        
        The key 'transcription' should be tokenized using the tokenizer, the key 'faces' should be transformed using the 
        transforms and then stacked into a single tensor. The key 'keypoints' should be transformed using the transforms and
        then stacked into a single tensor.
        '''
        transcriptions = []  # To store tokenized transcriptions
        text_tokens = []
        faces = []     # To store transformed face frames
        keypoints = []       # To store transformed keypoints   
        max_tokens = max([len(sample['transcription']) for sample in batch]) # Get the maximum length of the transcriptions in the batch
        
        
        # Iterate through each sample in the batch
        for sample in batch:
            
            text_token = sample['tokens']
            faces = sample['faces']
            keypoints = sample['keypoints']
            transcription = sample['transcription']
            
                
            text_tokenized = torch.from_numpy(
                self.tokenizer(
                    text_token + [self.tokenizer.padding_token] * (max_tokens - len(text_token))
                )
            )
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
            if self.face_transforms:
                transformed_faces = self.face_transforms(sample['faces'])
                    
            else:
                transformed_faces = sample['faces']
             
            #Apply transforms to keypoints (if transforms are provided)   
            if self.keypoints_transforms:
                transformed_keypoints = self.keypoints_transforms(sample['keypoints'])
            else:
                transformed_keypoints = sample['keypoints']

            # Append tokenized transcription tensor, transformed face frames, and keypoints to respective lists
            text_tokens.append(text_token)
            faces.append(transformed_faces)
            keypoints.append(transformed_keypoints)

        # Stack the lists of transcriptions, face frames, and keypoints into tensors
        #transcriptions = torch.stack(transcriptions)
        #faces = torch.stack(faces)
        #keypoints = torch.stack(keypoints)

        # Create a dictionary with the collated data
        collated_data = {
            'transcription': transcriptions,
            'faces': faces,
            'keypoints': keypoints
        }
        
        return collated_data

     
    def __call__(self, batch): #__call__ method is used to call the instance of the class and it allows the class to be called as a function
        return self.collate_fn(batch)
    
    def __repr__(self): #__repr__ method is used to return a string representation of the object
        return f"LipReadingCollator(face transforms={self.face_transforms}, keypoints_transforms={self.keypoints_transforms}, tokenizer={self.tokenizer})"
        
        
if __name__ == "__main__":
    # Example usage of the LipReadingCollator
    # Create a mock dataset
    class MockDataset(Dataset):
        def __init__(self):
            self.data = [
                {
                    'transcription': "hello",
                    'faces': np.random.rand(10, 3, 64, 64),
                    'keypoints': np.random.rand(10, 68, 2)
                },
                {
                    'transcription': "word",
                    'faces': np.random.rand(10, 3, 64, 64),
                    'keypoints': np.random.rand(10, 68, 2)
                }
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Create a mock dataset
    dataset = MockDataset()

    # Create a collator
    collator = LipReadingCollator()

    # Create a DataLoader using the collator
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collator, shuffle=True)

    # Iterate through the DataLoader
    for batch in dataloader:
        print(batch)
        break