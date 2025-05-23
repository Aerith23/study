import torch
import torch.nn as nn
import torch.nn.functional as F

class DualAttention(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(DualAttention, self).__init__()
        self.time_attn = nn.Linear(input_dim, 1)
        self.feature_attn = nn.Linear(seq_len, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # Time attention
        time_weights = torch.softmax(self.time_attn(x), dim=1)  # (batch, seq_len, 1)
        time_out = torch.sum(x * time_weights, dim=1)  # (batch, input_dim)

        # Feature attention
        x_trans = x.transpose(1, 2)  # (batch, input_dim, seq_len)
        feature_weights = torch.softmax(self.feature_attn(x_trans), dim=2)  # (batch, input_dim, 1)
        feature_out = torch.sum(x_trans * feature_weights, dim=2)  # (batch, input_dim)

        out = time_out + feature_out  # (batch, input_dim)
        return out

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded

class DynamicRepair(nn.Module):
    def __init__(self, window_size, input_dim):
        super(DynamicRepair, self).__init__()
        self.window_size = window_size
        self.adjust_gate = nn.Linear(input_dim, input_dim)

    def forward(self, x, reconstructed):
        # x, reconstructed: (batch, seq_len, input_dim)
        diff = x - reconstructed
        repair_signal = torch.tanh(self.adjust_gate(diff))
        repaired = reconstructed + repair_signal
        return repaired

class AnomalyDetectorAndRepair(nn.Module):
    def __init__(self, input_dim, seq_len, hidden_dim, window_size):
        super(AnomalyDetectorAndRepair, self).__init__()
        self.attn = DualAttention(input_dim, seq_len)
        self.encoder_decoder = LSTMAutoencoder(input_dim, hidden_dim, seq_len)
        self.repair = DynamicRepair(window_size, input_dim)
        self.threshold = 0.1  

    def forward(self, x):
        attn_output = self.attn(x)  # (batch, input_dim)
        x_reconstructed = self.encoder_decoder(x)  # (batch, seq_len, input_dim)

        # 计算残差作为异常检测
        anomaly_score = torch.mean(torch.abs(x - x_reconstructed), dim=[1, 2])  # (batch,)

        # 阈值判断
        anomaly_mask = (anomaly_score > self.threshold).float().unsqueeze(1).unsqueeze(2)

        # 仅对异常样本进行修复
        repaired = self.repair(x, x_reconstructed)

        final_output = anomaly_mask * repaired + (1 - anomaly_mask) * x

        return {
            "anomaly_score": anomaly_score,
            "reconstructed": x_reconstructed,
            "repaired": repaired,
            "final_output": final_output
        }
