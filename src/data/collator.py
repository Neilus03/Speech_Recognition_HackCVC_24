#Collator function to collate the data from the data source

#Import the necessary libraries
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import nn 
from torchvision import transforms
from src.learning.text_tokenizer.tokenizers import CharTokenizer

KEYPOINTS_SIZE = 144

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
        keypoints = np.zeros((len(batch), max([sample['lip_keypoints'].shape[0] for sample in batch]), KEYPOINTS_SIZE)) - 1

        lip_keypoints = []       # To store transformed lip_keypoints
        max_tokens = max([len(sample['transcription']) for sample in batch]) # Get the maximum length of the transcriptions in the batch
        in_num_frames = []
        # Iterate through each sample in the batch
        for idx, sample in enumerate(batch):


            in_num_frames.append(sample['lip_keypoints'].shape[0])
            keypoints[idx, :sample['lip_keypoints'].shape[0], :] = sample['lip_keypoints']
            text_token = sample['tokens']
            
            lip_keypoints.append(sample['lip_keypoints'])
            transcriptions.append(sample['transcription'])
            text_tokenized = torch.from_numpy(
                self.tokenizer(
                    text_token + [self.tokenizer.padding_token] * (max_tokens - len(text_token))
                )
            )

            text_tokens.append(text_tokenized)


        # Stack the lists of transcriptions, face frames, and lip_keypoints into tensors
        text_tokens = torch.stack(text_tokens)
        # Create a dictionary with the collated data
        collated_data = {
            'labels':text_tokens,
            'lip_keypoints': torch.from_numpy(keypoints).float(),
            'transcriptions': transcriptions,
            'output_lengths': [len(x) for x in transcriptions], # Number of frames,
            'input_lengths': in_num_frames
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