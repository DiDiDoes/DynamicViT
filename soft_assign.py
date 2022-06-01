import collections.abc
from itertools import repeat
from functools import partial
import numpy as np

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channel=3, embed_dim=192, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim
        self.flatten = flatten

        self.proj = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2) # BCHW -> BNC
        x = self.norm(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=3, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attn=False, token_weight=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale # BHNN
        # attention with re-weighting
        if token_weight is None:
            attn = attn.softmax(dim=-1)
        else:
            max_attn = torch.max(attn, dim=-1, keepdim=True)[0]
            attn = attn - max_attn

            cls_token_weight = torch.ones(B, 1, 1, device=token_weight.device)
            token_weight = torch.cat((cls_token_weight, token_weight), dim=1).reshape(B, 1, 1, N)
            attn = attn.to(torch.float32).exp_() * token_weight
            attn = attn / attn.sum(dim=-1, keepdim=True)
            attn = attn.type_as(max_attn)
        dropped_attn = self.attn_drop(attn)

        x = (dropped_attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn
        else:
            return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # TODO: check what is DropPath
        # self.drop_path = DropPath()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x, return_attn=False, token_weight=None):
        if return_attn:
            x_res, attn = self.attn(self.norm1(x), return_attn=True, token_weight=token_weight)
            x = x + x_res
            x = x + self.mlp(self.norm2(x))
            return x, attn
        else:
            x = x + self.attn(self.norm1(x), token_weight=token_weight)
            x = x + self.mlp(self.norm2(x))
            return x

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
            dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.bias = None

    def forward(self, x, adj):
        x_copy = x.clone()
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y

class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
            pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs=1

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
                input_dim, hidden_dim, embedding_dim, num_layers, 
                add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        # self.bn = nn.BatchNorm1d(x.size()[1])
        self.bn_module = nn.LayerNorm(input_dim)

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
            normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
                [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                        normalize_embedding=normalize, dropout=dropout, bias=self.bias) 
                 for i in range(num_layers-2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        # bn_module = nn.BatchNorm1d(x.size()[1]).to(self.device)
        # return bn_module(x)
        return self.bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):
        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''
        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        for i in range(len(conv_block)):
            x = conv_block[i](x,adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x,adj)
        x_all.append(x)
        # x_tensor: [batch_size x num_nodes x embedding]
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        # conv
        assign_tensor = self.gcn_forward(x, adj,
                self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        assign_tensor = nn.Softmax(dim=-1)(self.pred_model(assign_tensor))
        
        # update x and adj
        x = torch.matmul(torch.transpose(assign_tensor, 1, 2), x)
        adj = torch.transpose(assign_tensor, 1, 2) @ adj @ assign_tensor
        
        return x, adj, assign_tensor


class TokenReduceViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channel=3, num_classes=1000, embed_dim=192,
                 depth=12, reducer=None, reduce_location=[3, 6, 9], num_tokens=[137, 96, 67], distance="soft",
                 num_heads=3, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None, act_layer=None,
                 niter=100):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_size = patch_size
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_channel=in_channel, embed_dim=embed_dim,
            flatten=False
        )
        num_patches = self.patch_embed.num_patches

        self.adj = self.get_adj(img_size, patch_size)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, norm_layer=norm_layer, act_layer=act_layer
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.reduce_location = reduce_location
        self.num_tokens = num_tokens
        # self.reducers = nn.ModuleList([reducer(num_token=n, distance=distance, niter=niter) for n in num_tokens])
        # self.reducers = num_tokens
        # self.reducers = reducer
        # self.reducers = nn.ModuleList([reducer(C=embed_dim, K=n) for n in num_tokens])
        if distance == "soft":
            self.predictor = nn.ModuleList([GcnEncoderGraph(embed_dim, embed_dim, embed_dim, n, 3) for n in num_tokens])
        self.distance = distance

        # representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequantial(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.loss_fn = nn.CrossEntropyLoss()

    def get_adj(self, img_size, patch_size):
        H = W = img_size // patch_size
        pos_x, pos_y = np.meshgrid(np.arange(H), np.arange(W))
        pos_x, pos_y = pos_x.flatten(), pos_y.flatten()
        pos_xx = (pos_x[:, None] - pos_x[None, :]) ** 2
        pos_yy = (pos_y[:, None] - pos_y[None, :]) ** 2
        adj = ((pos_xx + pos_yy) <= 1).astype(float)

        sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
        adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)

        adj = torch.from_numpy(adj).float()
        return adj

    def forward_features_soft(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        adj = self.adj.to(x.device).unsqueeze(0).expand(B, -1, -1)

        reducer_idx = 0
        token_weight = torch.ones(B, H*W, 1, device=x.device)
        for idx, block in enumerate(self.blocks):
            if idx in self.reduce_location:
                cls_token = x[:, :1, :]
                x = x[:, 1:, :]
                x, adj, assignment = self.predictor[reducer_idx](x, adj)
                x = torch.cat((cls_token, x), dim=1)
                token_weight = assignment.transpose(1, 2) @ token_weight
                reducer_idx += 1
            x = block(x, token_weight=token_weight)
            # print(x.shape)
        x = self.norm(x)
        return self.pre_logits(x[:, 0]), None

    def forward(self, x):
        if self.distance == "soft":
            x, n_iters = self.forward_features_soft(x)
        else:
            raise NotImplementedError
        x = self.head(x)
        return x
