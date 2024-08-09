import torch
import torch.nn.functional as F
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, MSPatchAttentionLayer, MSPatchLinearAttention, AttentionLayer
from layers.Embed import MultiPatchEmbedding
from layers.Autoformer_EncDec import series_decomp


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class MSPatch(nn.Module):
    def __init__(self, configs):
        super(MSPatch, self).__init__()
        self.layer = configs.e_layers
        # Encoder
        self.encoders = nn.ModuleList([Encoder(
            [
                EncoderLayer(
                    MSPatchAttentionLayer(
                        MSPatchLinearAttention(False, configs.factor, attention_dropout=configs.dropout,
                                               output_attention=configs.output_attention), configs.d_model,
                        configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        ) for _ in range(3)])

    def forward(self, enc_out_list):

        for i in range(len(enc_out_list)):
            enc_out_list[i], _ = self.encoders[i](enc_out_list[i])

        return enc_out_list  # 32*7,12,16


class PatchMix(nn.Module):
    def __init__(self, configs):
        super(PatchMix, self).__init__()
        self.depthwise_conv1 = nn.Conv2d(in_channels=configs.enc_in, out_channels=configs.enc_in, kernel_size=(3, 1),
                                         stride=(3, 1), groups=configs.enc_in)
        # self.pointwise_conv1 = nn.Conv2d(in_channels=configs.enc_in, out_channels=configs.enc_in, kernel_size=1, stride=1)

        self.depthwise_conv2 = nn.Conv2d(in_channels=configs.enc_in, out_channels=configs.enc_in, kernel_size=(4, 1),
                                         stride=(1, 1), groups=configs.enc_in)
        # self.pointwise_conv2 = nn.Conv2d(in_channels=configs.enc_in, out_channels=configs.enc_in, kernel_size=1, stride=1)

        self.drop = nn.Dropout(0.3)
        self.act = nn.GELU()
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)

    def forward(self, x):
        # x0 = self.drop(self.act(self.pointwise_conv1(self.depthwise_conv1(x[2])))) + x[1]
        x0 = self.drop(self.act(self.depthwise_conv1(x[2]))) + x[1]
        x0 = self.norm1(x0)
        # x1 = self.drop(self.pointwise_conv2(self.depthwise_conv2(x0))) + x[0]
        x1 = self.drop(self.depthwise_conv2(x0)) + x[0]
        x1 = self.norm2(x1)
        return [x1, x0, x[2]]


class Model(nn.Module):

    def __init__(self, configs, patch_len=[96, 48, 16], stride=[96, 24, 8]):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.layer = configs.e_layers
        self.block_layer = configs.block_layers
        # padding = stride
        padding = [0, 24, 8]
        self.patch_num = [1, 4, 12]
        # patching and embedding
        self.patch_embedding = MultiPatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)
        # series decomposition
        self.decomposition = series_decomp(configs.moving_avg)
        # MSPatch
        self.MSPatch_s_Blocks = nn.ModuleList([MSPatch(configs) for _ in range(configs.block_layers)])
        self.MSPatch_t_Blocks = nn.ModuleList([MSPatch(configs) for _ in range(configs.block_layers)])
        self.patch_mix = PatchMix(configs)
        # Prediction Head
        self.head_nf_list = [configs.d_model * self.patch_num[0], configs.d_model * self.patch_num[1], configs.d_model * self.patch_num[2]]
        self.head_list = nn.ModuleList([
            FlattenHead(configs.enc_in, self.head_nf_list[i], configs.pred_len, head_dropout=configs.dropout)
            for i in range(len(self.head_nf_list))
        ])

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # series decomposition type 1
        x_enc = x_enc.permute(0, 2, 1)
        x_enc, n_vars = self.patch_embedding(x_enc)
        enc_out_s_list = []
        enc_out_t_list = []
        for i in range(len(x_enc)):
            x_enc_s, x_enc_t = self.decomposition(x_enc[i])
            enc_out_s_list.append(x_enc_s)
            enc_out_t_list.append(x_enc_t)

        for i in range(self.block_layer):
            enc_out_s_list = self.MSPatch_s_Blocks[i](enc_out_s_list)  # 32*7,12,16
        for i in range(self.block_layer):
            enc_out_t_list = self.MSPatch_t_Blocks[i](enc_out_t_list)  # 32*7,12,16

        enc_out_list = [a + b for a, b in zip(enc_out_s_list, enc_out_t_list)]  # 32*7,12,16

        for i in range(len(enc_out_list)):
            enc_out_list[i] = torch.reshape(enc_out_list[i],
                                            (-1, n_vars, enc_out_list[i].shape[-2], enc_out_list[i].shape[-1]))
        enc_out_list = self.patch_mix(enc_out_list)

        for i in range(len(self.head_list)):
            enc_out_list[i] = self.head_list[i](enc_out_list[i])

        dec_out = torch.stack(enc_out_list, dim=-1).mean(-1)
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
