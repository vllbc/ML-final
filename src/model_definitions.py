import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        
        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        memory = self.transformer_encoder(src)
        output = self.decoder(memory[:, -1, :])
        return output

class CustomModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, output_dim, cnn_out_channels=64, kernel_size=3, dropout=0.1):
        super(CustomModel, self).__init__()
        self.d_model = d_model
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_out_channels, kernel_size=kernel_size, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # The input to the transformer will be the output from the CNN
        self.encoder_layer = nn.Linear(cnn_out_channels, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # Final linear layer
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_dim)
        # CNN expects: (batch_size, input_dim, seq_len)
        src = src.permute(0, 2, 1)
        
        # Pass through CNN
        cnn_out = self.cnn(src) # (batch_size, cnn_out_channels, seq_len)
        
        # Permute for transformer: (batch_size, seq_len, cnn_out_channels)
        cnn_out = cnn_out.permute(0, 2, 1)
        
        # Project to d_model
        src = self.encoder_layer(cnn_out)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Pass through transformer encoder
        transformer_out = self.transformer_encoder(src)
        
        # Take the output of the last time step
        output = self.decoder(transformer_out[:, -1, :])
        
        return output 