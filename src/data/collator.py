#Collator function to collate the data from the data source


class LipReadingCollator:
    def __init__(self, transforms = None, tokenizer = None):
        self.transforms = transforms
        self.tokenizer = tokenizer
    
    def collate_fn(batch):
        '''
        This function will take in a batch of data and collate it into a single tensor
        The batch is a list of dictionaries, where each dictionary contains the keys 'transcription', 'face_frames', and 'keypoints'
        The collate function should return a dictionary with the same keys, but the values should be a single tensor, each with the 
        proper transformation applied and properly tokenized
        
        The key 'transcription' should be tokenized using the tokenizer, the key 'face_frames' should be transformed using the 
        transforms and then stacked into a single tensor The key 'keypoints' should be transformed using the transforms and
        then stacked into a single tensor
        '''
        
        