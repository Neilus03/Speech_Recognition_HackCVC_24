import torch.nn as nn
import torch

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