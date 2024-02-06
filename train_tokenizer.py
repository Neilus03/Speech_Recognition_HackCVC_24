from src.learning.text_tokenizer.tokenizers import CharTokenizer
from src.data.datasets import LipReadingDataset
from src.data.collator import LipReadingCollator
from torch.utils.data import DataLoader

# Create a tokenizer for the LipReadingDataset
tokenizer = CharTokenizer(dataset=LipReadingDataset("/data3fast/users/group02/videos/tracks/"), tokenizer_name="tokenizer", save_on_init=True)
collator = LipReadingCollator(face_transforms=None, lip_keypoints_transforms=None, tokenizer=tokenizer)
print(len(LipReadingDataset("/data3fast/users/group02/videos/tracks/")))
dataloaders = DataLoader(LipReadingDataset("/data3fast/users/group02/videos/tracks/"), batch_size=1, shuffle=True, collate_fn=collator.collate_fn)

print(len(dataloaders))
# Iterate through the DataLoader
for batch in dataloaders:
    print(batch)
    break

