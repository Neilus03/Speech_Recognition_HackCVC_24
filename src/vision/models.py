import torch.nn as nn
import torch
import torchvision.models as models
import os
import numpy as np

class EncoderCNNLSTM(nn.Module):
    def __init__(self, encoder, lstm, input_size, hidden_size, num_layers):
        super(EncoderCNNLSTM, self).__init__()

        # ResNet a pre-trained CNN model as an encoder
        self.ResNet = encoder
        self.lstm = lstm

        self.projection = torch.nn.Linear(input_size, hidden_size)
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, num_layers)

    def forward(self, X):
        '''
        :param X:
           Input tensor of shape [batch_size, num_frames, channels, height, width]
        :return: Dict
        '''
        batch_size, num_frames, channels, height, width = X.size()

        # Reshape the input to [batch_size*num_frames, channels, height, width]
        X = X.view(batch_size * num_frames, channels, height, width)

        # Pass the batch through the CNN encoder
        cnn_output = self.ResNet(X)
        print(cnn_output.size())

        # Reshape CNN output to [batch_size, num_frames, features]
        cnn_output = cnn_output.view(batch_size, num_frames, -1)
        print(cnn_output.size())

        # Pass the CNN output through the LSTM
        lstm_output, (h_n, c_n) = self.lstm(cnn_output)

        return {
            'features': lstm_output,
            'hidden_states': (h_n, c_n)
        }

class TransformerDecoder(nn.Module):
    def __init__(self, encoder, encoder_input_size, decoder_token_size, decoder_depth, vocab_size, decoder_width):
        super(TransformerDecoder, self).__init__()

        self.encoder = encoder
        self.projection = torch.nn.Linear(encoder_input_size, decoder_token_size)
        self.memory = torch.nn.Linear(encoder_input_size, decoder_token_size)

        self.gelu_fn = torch.nn.GELU()

        self.layer = nn.TransformerDecoderLayer(d_model=decoder_token_size, nhead=decoder_width)
        self.decoder = nn.TransformerDecoder(self.layer, num_layers=decoder_depth)

        self.lm_head = torch.nn.Linear(decoder_token_size, vocab_size)

    def forward(self, X):

        '''
        :param X:
           Dict: Encoder will manage the keys of the dict.
                Encoder has to return FEATURES of the video as a key of another dictionary
        :return: Dict
        '''

        encoder_output = self.encoder(X)['features']  # Pass the batch X through the encoder

        memory = self.memory(encoder_output)

        projected = self.gelu_fn(self.projection(encoder_output))  # Project encoder output to decoder token size
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(projected.size(0)).to(projected.device)

        # Perform decoding using TransformerDecoder
        decoded = self.gelu_fn(self.decoder(
            tgt=projected,
            memory=memory,
            tgt_mask=tgt_mask
        ))


        # Project the decoder output to vocabulary space
        output = self.lm_head(decoded)

        return {
            'features': decoded,
            'language_head_output': output,
            'hidden_states': None
        }
    
# Specify the directory where your .npz files are located
# directory_path = '/home/adriangar8/Documents/academia/CVC/hack_repo/src/data/faces/ingles'

# List all the files in the directory
# files = os.listdir(directory_path)

"""

# Loop through the files and count the number of arrays in each .npz file
for file_name in files:
    if file_name.endswith('.npz'):

        file_path = os.path.join(directory_path, file_name)

        arrays = np.load(file_path)
        num_arrays = len(arrays["face_frames"])

        print(f'File: {file_name}, Number of Arrays: {num_arrays}')

"""

batch_size = 1
frame_length = 92
channels = 3
h = 224
w = 224

# Create a random batch of data
X = torch.rand(batch_size, frame_length, channels, h, w).cuda(6)

# Create an instance of the EncoderCNNLSTM
encoder = EncoderCNNLSTM(
    encoder=models.resnet50(pretrained=True).cuda(6),
    lstm=torch.nn.LSTM(2048, 512, 2).cuda(6),
    input_size=2048,
    hidden_size=512,
    num_layers=2
)

# Pass the batch through the encoder
encoder_output = encoder(X)

print(f'Encoder Output: {encoder_output}')