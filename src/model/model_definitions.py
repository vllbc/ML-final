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

class HTFN(nn.Module):
    def __init__(self, input_dim, output_dim, short_kernel_size=3, mid_kernel_size=7, long_kernel_size=15, 
                 gru_hidden_dim=64, gru_num_layers=2, cross_attention_heads=4, 
                 transformer_d_model=128, transformer_nhead=4, transformer_num_layers=3, 
                 transformer_dim_feedforward=256, dropout=0.1):
        super(HTFN, self).__init__()

        self.gru_hidden_dim = gru_hidden_dim

        self.short_stream = nn.Sequential(
            nn.Conv1d(input_dim, gru_hidden_dim, kernel_size=short_kernel_size, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.mid_stream = nn.Sequential(
            nn.Conv1d(input_dim, gru_hidden_dim, kernel_size=mid_kernel_size, padding='same', dilation=2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.long_stream = nn.Sequential(
            nn.Conv1d(input_dim, gru_hidden_dim, kernel_size=long_kernel_size, padding='same', dilation=4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.gru_short = nn.GRU(gru_hidden_dim, gru_hidden_dim, gru_num_layers, batch_first=True, dropout=dropout)
        self.gru_mid = nn.GRU(gru_hidden_dim, gru_hidden_dim, gru_num_layers, batch_first=True, dropout=dropout)
        self.gru_long = nn.GRU(gru_hidden_dim, gru_hidden_dim, gru_num_layers, batch_first=True, dropout=dropout)

        self.attention_short = nn.MultiheadAttention(gru_hidden_dim, cross_attention_heads, dropout=dropout, batch_first=True)
        self.attention_mid = nn.MultiheadAttention(gru_hidden_dim, cross_attention_heads, dropout=dropout, batch_first=True)
        self.attention_long = nn.MultiheadAttention(gru_hidden_dim, cross_attention_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(gru_hidden_dim)
        self.norm2 = nn.LayerNorm(gru_hidden_dim)
        self.norm3 = nn.LayerNorm(gru_hidden_dim)
        
        self.projection = nn.Linear(3 * gru_hidden_dim, transformer_d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model, 
            nhead=transformer_nhead, 
            dim_feedforward=transformer_dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
        
        self.output_decoder = nn.Linear(transformer_d_model, output_dim)

    def forward(self, src):
        src_permuted = src.permute(0, 2, 1)
        
        short_out = self.short_stream(src_permuted).permute(0, 2, 1) # -> (batch, seq, gru_hidden)
        mid_out = self.mid_stream(src_permuted).permute(0, 2, 1)
        long_out = self.long_stream(src_permuted).permute(0, 2, 1)

        gru_short_out, _ = self.gru_short(short_out)
        gru_mid_out, _ = self.gru_mid(mid_out)
        gru_long_out, _ = self.gru_long(long_out)

        short_q = gru_short_out
        mid_long_kv = torch.cat([gru_mid_out, gru_long_out], dim=1)
        refined_short, _ = self.attention_short(short_q, mid_long_kv, mid_long_kv)
        refined_short = self.norm1(refined_short + short_q)

        mid_q = gru_mid_out
        short_long_kv = torch.cat([gru_short_out, gru_long_out], dim=1)
        refined_mid, _ = self.attention_mid(mid_q, short_long_kv, short_long_kv)
        refined_mid = self.norm2(refined_mid + mid_q)

        long_q = gru_long_out
        short_mid_kv = torch.cat([gru_short_out, gru_mid_out], dim=1)
        refined_long, _ = self.attention_long(long_q, short_mid_kv, short_mid_kv)
        refined_long = self.norm3(refined_long + long_q)

        fused_features = torch.cat([refined_short, refined_mid, refined_long], dim=2)
        projected_features = self.projection(fused_features)
        
        transformer_out = self.transformer_encoder(projected_features)
        
        final_representation = transformer_out[:, -1, :]
        
        output = self.output_decoder(final_representation)
        
        return output 