from src.learning.text_tokenizer.tokenizers import CharTokenizer
from src.data.datasets import LipReadingDataset
from src.data.collator import LipReadingCollator
from torch.utils.data import DataLoader
from src.vision.models import EncoderCNNLSTM, TransformerDecoder
from src.learning.train_loop.ctc_loop import train_ctc

import torch

# Create a tokenizer for the LipReadingDataset
tokenizer = CharTokenizer(dataset=LipReadingDataset("/data3fast/users/group02/videos/tracks/"), tokenizer_name="tokenizer", save_on_init=True)
collator = LipReadingCollator(face_transforms=None, lip_keypoints_transforms=None, tokenizer=tokenizer)
dataloaders = DataLoader(LipReadingDataset("/data3fast/users/group02/videos/tracks/"), batch_size=1, shuffle=True, collate_fn=collator.collate_fn, drop_last=True)


device = 'cuda' # torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('dataloader lenght', len(dataloaders))


encoder = EncoderCNNLSTM(
    input_size=144,
    hidden_size=512,
    num_layers=2,
    char_num=len(tokenizer), device=device
)

decoder = TransformerDecoder(encoder, 512, 256,
                             1, len(tokenizer), 4)

train_ctc(0, dataloaders, optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3), model = decoder,
          loss_function=torch.nn.CTCLoss())



