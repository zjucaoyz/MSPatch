from math import sqrt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn, Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from Time_Series_Library.utils.masking import TriangularCausalMask
from torch_geometric.nn import EGConv
from torch_geometric.utils import dense_to_sparse


class Predict(nn.Module):
    def __init__(self, individual, c_out, seq_len, pred_len, dropout):
        super(Predict, self).__init__()
        self.individual = individual
        self.c_out = c_out

        if self.individual:
            self.seq2pred = nn.ModuleList()
            self.dropout = nn.ModuleList()
            for i in range(self.c_out):
                self.seq2pred.append(nn.Linear(seq_len, pred_len))
                self.dropout.append(nn.Dropout(dropout))
        else:
            self.seq2pred = nn.Linear(seq_len, pred_len)
            self.dropout = nn.Dropout(dropout)

    # (B,  c_out , seq)
    def forward(self, x):
        if self.individual:
            out = []
            for i in range(self.c_out):
                per_out = self.seq2pred[i](x[:, i, :])
                per_out = self.dropout[i](per_out)
                out.append(per_out)
            out = torch.stack(out, dim=1)
        else:
            out = self.seq2pred(x)
            out = self.dropout(out)

        return out


class Attention_Block(nn.Module):
    def __init__(self, d_model, d_ff=None, n_heads=8, dropout=0.1, activation="relu"):
        super(Attention_Block, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = self_attention(FullAttention, d_model, n_heads=n_heads)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class self_attention(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(self_attention, self).__init__()
        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention(attention_dropout=0.1)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)
        out = self.out_projection(out)
        return out, attn


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        # return V.contiguous()
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, seq_len, n_heads):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.seq_len = seq_len
        self.n_heads = n_heads

        # Labml.ai改进
        self.linear_l = nn.Linear(self.in_features, self.out_features, bias=False)
        self.linear_r = nn.Linear(self.in_features, self.out_features, bias=False)
        self.attn = nn.Linear(self.seq_len, 1, bias=False)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(0.6)

    def _prepare_attentional_mechanism_input(self, Wh):  # 32,32,7,96
        N = Wh.size()[2]  # number of nodes 7
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=2)
        Wh_repeated_alternating = Wh.repeat(1, 1, N, 1)
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=3)
        return all_combinations_matrix.view(Wh.size(0), Wh.size(1), N, N, 2 * Wh.size()[3])

    def _prepare_attentional_mechanism_input_v2(self, g_l, g_r):  # 32,7,96
        N = g_l.size()[1]  # number of nodes 7
        Wh_repeated_in_chunks = g_r.repeat_interleave(N, dim=1)
        Wh_repeated_alternating = g_l.repeat(1, N, 1)
        all_combinations_matrix = Wh_repeated_alternating + Wh_repeated_in_chunks
        return all_combinations_matrix.view(g_l.size(0), N, N, g_l.size()[2])

    def forward(self, input, adj):
        h_heads = []
        for i in range(self.n_heads):
            g_l = self.linear_l(input)
            g_r = self.linear_r(input)
            a_input = self._prepare_attentional_mechanism_input_v2(g_l.permute(0, 2, 1),
                                                                   g_r.permute(0, 2, 1))  # 32,32,7,7,96
            e = self.attn(self.activation(a_input)).squeeze(-1)  # 32,32,7,7
            zero_vec = -9e15 * torch.ones_like(e)  # 32,32,7,7
            attention = torch.where(adj > 0, e, zero_vec)  # 32,32,7,7
            attention = F.softmax(attention, dim=-1)  # 32,32,7,7
            attention = self.dropout(attention)  # 32,32,7,7
            h_prime = torch.einsum('bij,bjf->bif', attention, g_r.permute(0, 2, 1))
            # h_prime = torch.matmul(attention, g_r)  # 32,32,7,7 | 32,32,7,96 -> 32,32,7,96 可以验证一下
            h_heads.append(h_prime)
        out = torch.mean(torch.stack(h_heads, dim=-1), dim=-1).squeeze(-1)  # 32,11,512

        return out


class GraphBlock(nn.Module):
    def __init__(self, c_out, d_model, conv_channel, skip_channel,
                 gcn_depth, dropout, propalpha, seq_len, node_dim):
        super(GraphBlock, self).__init__()

        self.nodevec1 = nn.Parameter(torch.randn(c_out, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, c_out), requires_grad=True)
        self.start_conv = nn.Conv2d(1, conv_channel, (d_model - c_out + 1, 1))
        self.gconv1 = mixprop(conv_channel, skip_channel, gcn_depth, dropout, propalpha)
        self.gelu = nn.GELU()
        self.end_conv = nn.Conv2d(skip_channel, seq_len, (1, seq_len))
        self.linear = nn.Linear(c_out, d_model)
        self.norm = nn.LayerNorm(d_model)

    # x in (B, T, d_model)
    # Here we use a mlp to fit a complex mapping f (x)
    def forward(self, x):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        out = x.unsqueeze(1).transpose(2, 3)  # (B, 1, d_model, T)
        out = self.start_conv(out)  # (B, conv_channel, c_out, T)
        out = self.gelu(self.gconv1(out, adp))  # (B, conv_channel, c_out, T)
        out = self.end_conv(out).squeeze()
        out = self.linear(out)

        return self.norm(x + out)


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class Adj_Graph_Block(nn.Module):
    def __init__(self, c_out, d_model, conv_channel, skip_channel,
                 gcn_depth, dropout, propalpha, seq_len, node_dim):
        super(Adj_Graph_Block, self).__init__()

        self.nodevec1 = nn.Parameter(torch.randn(c_out, node_dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(node_dim, c_out), requires_grad=True)
        # self.timevec1 = nn.Parameter(torch.randn(seq_len, node_dim), requires_grad=True)
        # self.timevec2 = nn.Parameter(torch.randn(node_dim, seq_len), requires_grad=True)
        self.start_conv = nn.Conv2d(1, conv_channel, (d_model - c_out + 1, 1))
        # 时间维度
        self.gconv2 = mixprop(conv_channel, skip_channel, gcn_depth, dropout, propalpha)
        # 特征维度
        # self.gconv1 = GraphAttentionLayer(seq_len, seq_len, seq_len, n_heads=1)
        self.gelu = nn.GELU()
        self.end_conv = nn.Conv2d(skip_channel, seq_len, (1, seq_len))
        self.linear = nn.Linear(c_out, d_model)
        self.norm = nn.LayerNorm(d_model)

    def add_cross_var_adj(self, adj):
        k = 3
        k = min(k, adj.shape[0])
        mask = (adj < torch.topk(adj, k=adj.shape[0] - k)[0][..., -1, None]) * (
                adj > torch.topk(adj, k=adj.shape[0] - k)[0][..., -1, None])
        mask_pos = adj >= torch.topk(adj, k=k)[0][..., -1, None]
        mask_neg = adj <= torch.kthvalue(adj, k=k)[0][..., -1, None]
        return mask, mask_pos, mask_neg

    def generate_scale_mask(self, adj, scale):
        # 获取邻接矩阵的大小
        rows, cols = adj.shape
        # 创建一个与邻接矩阵同形状的全True矩阵
        mask = torch.ones_like(adj, dtype=torch.bool)
        # 遍历每一列，每隔 `scale` 个元素设为 False，并保证下三角矩阵
        for j in range(cols):
            for i in range(rows):
                if i > j and (i % scale == scale - 1):
                    mask[i, j] = False
                if i == j:
                    mask[i, j] = False
                if i == j + 1:
                    mask[i, j] = False

        return mask

    def logits_warper(self, adj, indices_to_remove, mask_pos, mask_neg, filter_value=-float("Inf")):
        # print('adj:',adj)
        mask_pos_inverse = ~mask_pos
        mask_neg_inverse = ~mask_neg
        # Replace values for mask_pos rows
        processed_pos = mask_pos * F.softmax(adj.masked_fill(mask_pos_inverse, filter_value), dim=-1)
        # Replace values for mask_neg rows
        processed_neg = -1 * mask_neg * F.softmax((1 / (adj + 1)).masked_fill(mask_neg_inverse, filter_value), dim=-1)
        # Combine processed rows for both cases
        processed_adj = processed_pos + processed_neg
        return processed_adj

    def top_k(self, adj, k=3):
        values, indices = torch.topk(adj, k, dim=-1)
        sparse_adp = torch.zeros_like(adj)
        for i in range(adj.size(0)):
            sparse_adp[i, indices[i]] = values[i]
        return F.softmax(sparse_adp, dim=-1)

    # x in (B, T, d_model)
    # Here we use a mlp to fit a complex mapping f (x)
    def forward(self, x):
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=-1)
        adp = self.top_k(adp)
        # 特征维度的 mask设计
        # mask, mask_pos, mask_neg = self.add_cross_var_adj(adp)
        # adp = self.logits_warper(adp, mask, mask_pos, mask_neg)
        out = x.unsqueeze(1).transpose(2, 3)  # (B, 1, d_model, T)
        out = self.start_conv(out)  # (B, conv_channel, c_out, T)
        out = self.gelu(self.gconv2(out, adp))
        # out = self.gelu(self.gconv1(out, adp))  # (B, conv_channel, c_out, T)
        out = self.end_conv(out).squeeze()
        # 空洞卷积
        # out = self.dilated_conv(out)
        out = self.linear(out)

        return self.norm(x + out), adp


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        # x = torch.einsum('ncwl,vl->ncvw', (x, A))
        # x = torch.einsum('ncwl,wv->nclv',(x,A)
        return x.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class mixprop(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = linear((gdep + 1) * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, a)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


class simpleVIT(nn.Module):
    def __init__(self, in_channels, emb_size, patch_size=2, depth=1, num_heads=4, dropout=0.1, init_weight=True):
        super(simpleVIT, self).__init__()
        self.emb_size = emb_size
        self.depth = depth
        self.to_patch = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, 2 * patch_size + 1, padding=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.layers = nn.ModuleList([])
        for _ in range(self.depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, dropout),
                FeedForward(emb_size, emb_size)
            ]))

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, N, _, P = x.shape
        x = self.to_patch(x)
        # x = x.permute(0, 2, 3, 1).reshape(B,-1, N)
        for norm, attn, ff in self.layers:
            x = attn(norm(x)) + x
            x = ff(x) + x

        x = x.transpose(1, 2).reshape(B, self.emb_size, -1, P)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)
