import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention # Pro attention with mask
        self.cross_attention = cross_attention # full attention with mask
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask # attn_mask 指定mask 的方式, 但是这个参数实际代码没用到, 里边自己定义了另外的mask方式
        )[0])
        x = self.norm1(x) # torch.Size([32, 72, 512])

        x = x + self.dropout(self.cross_attention(
            x, cross, cross, # 来自encoder的输出 ,作为 k 和 v 的input
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers) # 传进来一个decoder layers list
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            # 下一层的x : 输入是上一层的输出, 但是 来自encoder的结果一直是作为输入, 不变
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            # x :  torch.Size([32, 72, 512]) # 上一层的输入
            # cross : torch.Size([32, 48, 512]) # encoder的输出
        if self.norm is not None:
            x = self.norm(x)

        return x