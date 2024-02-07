import torch.nn as nn
import torch
import torchvision.models as models
import os
import numpy as np



def linear_constructor(topology: list):
    seq = []
    for n, size in enumerate(topology[:-1]):
        seq.extend([
            nn.ReLU(),
            nn.Linear(size, topology[n + 1])
        ])

    return seq


class EncoderCNNLSTM(nn.Module):
    def __init__(self,  input_size, hidden_size, num_layers, char_num, projection_topology = [128, 128], device = 'cuda'):
        super(EncoderCNNLSTM, self).__init__()

        self.projection = torch.nn.Sequential(*linear_constructor([input_size] + projection_topology + [hidden_size]))
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, num_layers)
        self.last_projection = torch.nn.Linear(hidden_size, char_num)
        self.device = device

        self.to(self.device)

    def forward(self, X):
        '''
        :param X:
           Input tensor of shape [batch_size, num_frames, channels, height, width]
        :return: Dict
        '''

        keypoints = X['lip_keypoints'].to(self.device)
        batch, seq, _ = keypoints.shape

        projected = self.projection(keypoints.view(batch*seq, -1)).reshape(batch, seq, -1)

        lstm_output, (h_n, c_n) = self.lstm(projected.transpose(1,0))

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
        self.to(encoder.device)

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