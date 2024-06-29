import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math
from nets.kernels import *
from torch.distributions import Categorical

# implements skip-connection module
class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)

# implements MLP module
class MLP(torch.nn.Module):
    def __init__(self,
                input_dim = 128,
                feed_forward_dim = 64,
                embedding_dim = 64,
                output_dim = 1
    ):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, feed_forward_dim)
        self.fc2 = torch.nn.Linear(feed_forward_dim, embedding_dim)
        self.fc3 = torch.nn.Linear(embedding_dim, output_dim)
        self.dropout = torch.nn.Dropout(p=0.05)
        self.ReLU = nn.ReLU(inplace = True)
        
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, in_):
        result = self.ReLU(self.fc1(in_))
        result = self.dropout(result)
        result = self.ReLU(self.fc2(result))
        result = self.fc3(result).squeeze(-1)
        return result

# implements Normalization module
class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalization = normalization

        if not self.normalization == 'layer':
            self.normalizer = normalizer_class(embed_dim, affine=True)

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.normalization == 'layer':
            return (input - input.mean((1,2)).view(-1,1,1)) / torch.sqrt(input.var((1,2)).view(-1,1,1) + 1e-05)

        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input

class CompatNeighbour(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(CompatNeighbour, self).__init__()
    
        n_heads = 4
        
        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.W_Q = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_K = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        
        self.agg = MLP(12, 32, 32, 1, 0)

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h, rec, visited_order_map, selection_sig):
        
        pre = rec.argsort()
        post = rec.gather(1, rec)
        batch_size, graph_size, input_dim = h.size()

        flat = h.contiguous().view(-1, input_dim) #################   reshape

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        hidden_Q = torch.matmul(flat, self.W_Q).view(shp)
        hidden_K = torch.matmul(flat, self.W_K).view(shp)
        
        Q_pre = hidden_Q.gather(2, pre.view(1, batch_size, graph_size, 1).expand_as(hidden_Q))
        K_post = hidden_K.gather(2, post.view(1, batch_size, graph_size, 1).expand_as(hidden_Q))
    
        compatibility = ((Q_pre * hidden_K).sum(-1) + (hidden_Q * K_post).sum(-1) - (Q_pre * K_post).sum(-1))[:,:,1:]
        
        compatibility_pairing = torch.cat((compatibility[:,:,:graph_size // 2], compatibility[:,:,graph_size // 2:]), 0)

        compatibility_pairing = self.agg(torch.cat((compatibility_pairing.permute(1,2,0), 
                                                    selection_sig.permute(0,2,1)),-1)).squeeze()
        
        return  compatibility_pairing

class Reinsertion(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(Reinsertion, self).__init__()
    
        n_heads = 4
        
        if val_dim is None:
            # assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        
        self.norm_factor = 1 / math.sqrt(2 * embed_dim)  # See Attention is all you need


        self.compater_insert1 = MultiHeadCompat(n_heads,
                                        embed_dim,
                                        embed_dim,
                                        embed_dim,
                                        key_dim)
        
        self.compater_insert2 = MultiHeadCompat(n_heads,
                                        embed_dim,
                                        embed_dim,
                                        embed_dim,
                                        key_dim)
        
        self.agg = MLP(16, 32, 32, 1, 0)

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)


    def forward(self, h, pos_pickup, pos_delivery, rec, mask=None):
        
        batch_size, graph_size, input_dim = h.size()
        shp = (batch_size, graph_size, graph_size, self.n_heads)
        shp_p = (batch_size, -1, 1, self.n_heads)
        shp_d = (batch_size, 1, -1, self.n_heads)
        
        arange = torch.arange(batch_size, device = h.device)
        h_pickup = h[arange,pos_pickup].unsqueeze(1)
        h_delivery = h[arange,pos_delivery].unsqueeze(1)
        h_K_neibour = h.gather(1, rec.view(batch_size, graph_size, 1).expand_as(h))
        
        compatibility_pickup_pre = self.compater_insert1(h_pickup, h).permute(1,2,3,0).view(shp_p).expand(shp)
        compatibility_pickup_post = self.compater_insert2(h_pickup, h_K_neibour).permute(1,2,3,0).view(shp_p).expand(shp)
        compatibility_delivery_pre = self.compater_insert1(h_delivery, h).permute(1,2,3,0).view(shp_d).expand(shp)
        compatibility_delivery_post = self.compater_insert2(h_delivery, h_K_neibour).permute(1,2,3,0).view(shp_d).expand(shp)
        
        compatibility = self.agg(torch.cat((compatibility_pickup_pre, 
                                            compatibility_pickup_post, 
                                            compatibility_delivery_pre, 
                                            compatibility_delivery_post),-1)).squeeze()
        return compatibility

# implements the encoder for Critic net
class MultiHeadAttentionLayerforCritic(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
            kernel='no_kernel',
            kernel_enabled = True
    ):
        super(MultiHeadAttentionLayerforCritic, self).__init__(
            SkipConnection(
                    MultiHeadAttentionOrigin(
                        n_heads,
                        input_dim=embed_dim,
                        embed_dim=embed_dim,
                        kernel=kernel,
                        kernel_enabled=kernel_enabled
                    )                
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                    nn.Sequential(
                        nn.Linear(embed_dim, feed_forward_hidden),
                        nn.ReLU(inplace = True),
                        nn.Linear(feed_forward_hidden, embed_dim)
                    ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
            ),
            Normalization(embed_dim, normalization)
        ) 
                    
# implements the decoder for Critic net
class ValueDecoder(nn.Module):
    def __init__(
            self,
            embed_dim,
            input_dim,
    ):
        super(ValueDecoder, self).__init__()
        self.hidden_dim = embed_dim
        self.embedding_dim = embed_dim
        
        # for Pooling
        self.project_graph = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.project_node = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False) 
        
        # for output
        self.MLP = MLP(input_dim, embed_dim)


    def forward(self, h_em): 
        # mean Pooling
        mean_pooling = h_em.mean(1)
        graph_feature = self.project_graph(mean_pooling)[:, None, :]
        node_feature = self.project_node(h_em)
        fusion = node_feature + graph_feature.expand_as(node_feature)
        
        #pass through value_head, get estimated values
        value = self.MLP(fusion.mean(1))
      
        return value


# implements the orginal Multi-head Self-Attention module
class MultiHeadAttentionOrigin(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
            kernel='no_kernel',
            kernel_enabled=True
    ):
        super(MultiHeadAttentionOrigin, self).__init__()

        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.kernel_func = kernels.get(kernel)
        self.kernel_enabled = kernel_enabled

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q):
        
        h = q  # compute self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)   
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        if self.kernel_enabled:
            compatibility = self.norm_factor * self.kernel_func(Q,K)
        else:
            compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        attn = F.softmax(compatibility, dim=-1)   
       
        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


# implements the propsoed Multi-head DAC-Att module
class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
            kernel='no_kernel',
            kernel_enabled = [True, False]
    ):
        super(MultiHeadAttention, self).__init__()
        
        self.n_heads = n_heads
        
        self.key_dim = self.val_dim = embed_dim // n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.norm_factor = 1 / math.sqrt(1 * self.key_dim)

        # W_h^Q in the paper
        self.W_query_node = nn.Parameter(torch.Tensor(n_heads, self.input_dim, self.key_dim))
        # W_g^Q in the paper
        self.W_query_pos = nn.Parameter(torch.Tensor(n_heads, self.input_dim, self.key_dim))
        # W_h^K in the paper
        self.W_key_node = nn.Parameter(torch.Tensor(n_heads, self.input_dim, self.key_dim))
        # W_g^K in the paper
        self.W_key_pos = nn.Parameter(torch.Tensor(n_heads, self.input_dim, self.key_dim))
        
        # W_h^V and W_h^Vref in the paper
        self.W_val_node = nn.Parameter(torch.Tensor(2 * n_heads, self.input_dim, self.val_dim))
        # W_g^V and W_g^Vref in the paper
        self.W_val_pos = nn.Parameter(torch.Tensor(2 * n_heads, self.input_dim, self.val_dim))

        self.kernel_func = kernels.get(kernel)
        self.kernel_enabled = kernel_enabled
        
        # W_h^O and W_g^O in the paper
        if embed_dim is not None:
            self.W_out_node = nn.Parameter(torch.Tensor( n_heads, 2 * self.key_dim, embed_dim))
            self.W_out_pos = nn.Parameter(torch.Tensor( n_heads, 2 * self.key_dim, embed_dim))
        
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, h_node_in, h_pos_in): #input (NFEs, PFEs)
        
        # h,g should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h_node_in.size()

        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_v = (2, self.n_heads, batch_size, graph_size, -1)
        
        h_node = h_node_in.contiguous().view(-1, input_dim)
        h_pos = h_pos_in.contiguous().view(-1, input_dim)
        

        Q_node = torch.matmul(h_node, self.W_query_node).view(shp)
        Q_pos = torch.matmul(h_pos, self.W_query_pos).view(shp)

        K_node = torch.matmul(h_node, self.W_key_node).view(shp)
        K_pos = torch.matmul(h_pos, self.W_key_pos).view(shp)
        
        V_node = torch.matmul(h_node, self.W_val_node).view(shp_v)
        V_pos = torch.matmul(h_pos, self.W_val_pos).view(shp_v)

        # Get attention correlations and norm by softmax
        if self.kernel_enabled[0]:
            node_correlations = self.norm_factor * self.kernel_func(Q_node, K_node)
        else:
            node_correlations = self.norm_factor * torch.matmul(Q_node, K_node.transpose(-2, -1))
        
        if self.kernel_enabled[1]:
            pos_correlations =  self.norm_factor * self.kernel_func(Q_pos, K_pos)
        else:
            pos_correlations =  self.norm_factor * torch.matmul(Q_pos, K_pos.transpose(-2, -1))
        
        attn1 = F.softmax(node_correlations, dim=-1) # head, bs, n, n
        attn2 = F.softmax(pos_correlations, dim=-1) # head, bs, n, n
        
        heads_node_1 = torch.matmul(attn1, V_node[0]) # self-attn
        heads_node_2 = torch.matmul(attn2, V_node[1]) # cross-aspect ref attn
        
        heads_pos_1 = torch.matmul(attn1, V_pos[0]) # cross-aspect ref attn
        heads_pos_2 = torch.matmul(attn2, V_pos[1]) # self-attn
        
        heads_node = torch.cat((heads_node_1, heads_node_2) , -1)
        heads_pos = torch.cat((heads_pos_1, heads_pos_2), -1)

        # get output
        out_node = torch.mm(
            heads_node.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * 2 * self.val_dim),
            self.W_out_node.view(-1, self.embed_dim)
        ).view(batch_size, graph_size, self.embed_dim)
        
        out_pos = torch.mm(
            heads_pos.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * 2 * self.val_dim),
            self.W_out_pos.view(-1, self.embed_dim)
        ).view(batch_size, graph_size, self.embed_dim)

        return out_node, out_pos # dual-aspect representation (NFEs, PFEs)
  
# implements the multi-head compatibility layer used in the DAC decoder
class MultiHeadCompat(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
            kernel='no_kernel'
    ):
        super(MultiHeadCompat, self).__init__()
    
        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(1 * key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))

        self.kernel_func = kernels.get(kernel)

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h = None, mask=None):
        if h is None:
            h = q  # compute self-attention

        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)  
        K = torch.matmul(hflat, self.W_key).view(shp)   

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.kernel_func(Q, K)
        
        return self.norm_factor * compatibility


# implements the DAC decoder
class MultiHeadDecoder(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
            kernel='no_kernel'
    ):
        super(MultiHeadDecoder, self).__init__()
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.input_dim = input_dim
        
        # for MHC sublayer (NFE aspect)
        self.compater_node = MultiHeadCompat(n_heads,
                                        embed_dim,
                                        embed_dim,
                                        embed_dim,
                                        key_dim,
                                        kernel)
        
        # for MHC sublayer (PFE aspect)
        self.compater_pos = MultiHeadCompat(n_heads,
                                embed_dim,
                                embed_dim,
                                embed_dim,
                                key_dim,
                                kernel)
        
        # for Max-Pooling sublayer
        self.project_graph_pos = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_graph_node = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_node_pos = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_node_node = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
    
        
        # for feed-forward aggregation (FFA)sublayer
        self.value_head = MLP(self.n_heads*2, 32, 32, 1)


    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
        
        
    def forward(self, h_em, pos_em, solving_state_for_net):
        
        batch_size, graph_size, dim = h_em.size()
        
        # Max-Pooling sublayer
        h_node_refined = self.project_node_node(h_em) + self.project_graph_node(h_em.max(1)[0])[:, None, :].expand(batch_size, graph_size, dim)
        h_pos_refined = self.project_node_pos(pos_em) + self.project_graph_pos(pos_em.max(1)[0])[:, None, :].expand(batch_size, graph_size, dim)
        
        # MHC sublayer
        compatibility = torch.zeros((batch_size, graph_size, graph_size, self.n_heads * 2), device = h_node_refined.device)
        compatibility[:,:,:,:self.n_heads] = self.compater_pos(h_pos_refined).permute(1,2,3,0)
        compatibility[:,:,:,self.n_heads:] = self.compater_node(h_node_refined).permute(1,2,3,0)
        
        # FFA sublater
        return self.value_head(compatibility).squeeze(-1)

class MultiHeadDecoderPDTSP(nn.Module):
    def __init__(
            self,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
            v_range = 6,
    ):
        super(MultiHeadDecoderPDTSP, self).__init__()
        self.n_heads = n_heads = 1
        self.embed_dim = embed_dim
        self.input_dim = input_dim        
        self.range = v_range
        
        self.compater_removal = CompatNeighbour(n_heads,
                                        embed_dim,
                                        embed_dim,
                                        embed_dim,
                                        key_dim)
        self.compater_reinsertion = Reinsertion(n_heads,
                                        embed_dim,
                                        embed_dim,
                                        embed_dim,
                                        key_dim)
            
        self.project_graph = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.project_node = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
        
        
    def forward(self, problem, h_em, rec, x_in, top2, visited_order_map, pre_action, selection_sig, fixed_action = None, require_entropy = False):        
    
        bs, gs, dim = h_em.size()
        half_pos =  (gs - 1) // 2
        
        arange = torch.arange(bs)
    
        h = self.project_node(h_em) + self.project_graph(h_em.max(1)[0])[:, None, :].expand(bs, gs, dim)
        
        ############# action1 removal

        action_removal_table = torch.tanh(self.compater_removal(h, rec, visited_order_map, selection_sig).squeeze()) * self.range
        if pre_action is not None and pre_action[0,0] > 0:
            action_removal_table[arange, pre_action[:,0]] = -1e20
        log_ll_removal = F.log_softmax(action_removal_table, dim = -1) if self.training else None
        probs_removal = F.softmax(action_removal_table, dim = -1)

        if fixed_action is not None:
            action_removal = fixed_action[:,:1]
        else:
            action_removal = probs_removal.multinomial(1)
        selected_log_ll_action1 = log_ll_removal.gather(1, action_removal) if self.training else torch.tensor(0).to(h.device)
        
        ############# action2
        pos_pickup = (1 + action_removal).view(-1)
        pos_delivery = pos_pickup + half_pos
        mask_table = problem.get_swap_mask(action_removal + 1, visited_order_map, top2).expand(bs, gs, gs).cpu()

        action_reinsertion_table = torch.tanh(self.compater_reinsertion(h, pos_pickup, pos_delivery, rec, mask_table)) * self.range
             
        action_reinsertion_table[mask_table] = -1e20
        
        del visited_order_map, mask_table

        #reshape action_reinsertion_table
        action_reinsertion_table = action_reinsertion_table.view(bs, -1)
        log_ll_reinsertion = F.log_softmax(action_reinsertion_table, dim = -1) if self.training else None
        probs_reinsertion = F.softmax(action_reinsertion_table, dim = -1)

        # fixed action
        if fixed_action is not None:
            p_selected = fixed_action[:,1]
            d_selected = fixed_action[:,2]
            pair_index = p_selected * gs + d_selected
            pair_index = pair_index.view(-1,1)
            action = fixed_action
        else:
            # sample one action
            pair_index = probs_reinsertion.multinomial(1)
            
            p_selected = pair_index // gs 
            d_selected = pair_index % gs     
            action = torch.cat((action_removal.view(bs, -1), p_selected, d_selected),-1)  # pair: no_head bs, 2
        
        selected_log_ll_action2 = log_ll_reinsertion.gather(1, pair_index)  if self.training else torch.tensor(0).to(h.device)
        
        log_ll = selected_log_ll_action1 + selected_log_ll_action2
        
        if require_entropy and self.training:
            dist = Categorical(probs_reinsertion, validate_args=False)
            entropy = dist.entropy()
        else:
            entropy = None

        return action, log_ll, entropy

# implements the DAC encoder
class MultiHeadEncoder(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
            kernel='no_kernel',
            kernel_enabled = [True, False]
    ):
        super(MultiHeadEncoder, self).__init__()
        
        self.MHA_sublayer = MultiHeadAttentionsubLayer(
                        n_heads,
                        embed_dim,
                        feed_forward_hidden,
                        normalization=normalization,
                        kernel=kernel,
                        kernel_enabled = kernel_enabled
                )
        
        self.FFandNorm_sublayer = FFandNormsubLayer(
                        n_heads,
                        embed_dim,
                        feed_forward_hidden,
                        normalization=normalization,
                )
        
    def forward(self, input1, input2):
        out1, out2 = self.MHA_sublayer(input1, input2)
        return self.FFandNorm_sublayer(out1, out2)

# implements the DAC encoder (DAC-Att sublayer)   
class MultiHeadAttentionsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
            kernel='no_kernel',
            kernel_enabled = [True, False]
    ):
        super(MultiHeadAttentionsubLayer, self).__init__()
        
        self.MHA = MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim,
                    kernel=kernel,
                    kernel_enabled = kernel_enabled
                )
        
        self.Norm = Normalization(embed_dim, normalization)
    
    
    def forward(self, input1, input2):
        # Attention and Residual connection
        out1, out2 = self.MHA(input1, input2)
        
        # Normalization
        return self.Norm(out1 + input1), self.Norm(out2 + input2)

# implements the DAC encoder (FFN sublayer)   
class FFandNormsubLayer(nn.Module):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden,
            normalization='layer',
    ):
        super(FFandNormsubLayer, self).__init__()
        
        self.FF1 = nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(inplace = True),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
        
        self.FF2 = nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(inplace = True),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
        
        self.Norm = Normalization(embed_dim, normalization)
    
    def forward(self, input1, input2):
    
        # FF and Residual connection
        out1 = self.FF1(input1)
        out2 = self.FF2(input2)
        
        # Normalization
        return self.Norm(out1 + input1), self.Norm(out2 + input2)

# implements the initilization of the NFEs and the PFEs
class EmbeddingNet(nn.Module):
    
    def __init__(
            self,
            node_dim,
            embedding_dim,
            seq_length,
        ):
        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim
        self.embedding_dim = embedding_dim
        self.embedder = nn.Linear(node_dim, embedding_dim, bias = False)
        
        # Two ways for generalizing CPEs:
        # -- 1. Use the target size CPE directly (default)
        self.pattern = self.Cyclic_Positional_Encoding(seq_length, embedding_dim)
        # -- 2. Use the original size CPE: reuse the wavelength of the original size but make it compatible with the target size (by duplicating or discarding)
        # way 1 works well for most of cases
        # original_size, target_size = 100, 150
        # self.pattern = self.Cyclic_Positional_Encoding(original_size, embedding_dim, target_size=target_size)
        
    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    
    def basesin(self, x, T, fai = 0):
        return np.sin(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)
    
    def basecos(self, x, T, fai = 0):
        return np.cos(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)
    
    # implements the CPE
    def Cyclic_Positional_Encoding(self, n_position, emb_dim, mean_pooling = True, target_size=None):
        
        Td_set = np.linspace(np.power(n_position, 1 / (emb_dim // 2)), n_position, emb_dim // 2, dtype = 'int')
        x = np.zeros((n_position, emb_dim))
         
        for i in range(emb_dim):
            Td = Td_set[i //3 * 3 + 1] if  (i //3 * 3 + 1) < (emb_dim // 2) else Td_set[-1]
            fai = 0 if i <= (emb_dim // 2) else  2 * np.pi * ((-i + (emb_dim // 2)) / (emb_dim // 2))
            longer_pattern = np.arange(0, np.ceil((n_position) / Td) * Td, 0.01)
            if i % 2 ==1:
                x[:,i] = self.basecos(longer_pattern, Td, fai)[np.linspace(0, len(longer_pattern), n_position, dtype = 'int', endpoint = False)]
            else:
                x[:,i] = self.basesin(longer_pattern, Td, fai)[np.linspace(0, len(longer_pattern), n_position, dtype = 'int', endpoint = False)]
                
        pattern = torch.from_numpy(x).type(torch.FloatTensor)
        pattern_sum = torch.zeros_like(pattern)
        
        # for generalization (way 2): reuse the wavelength of the original size but make it compatible with the target size (by duplicating or discarding)
        if target_size is not None:
            pattern = pattern[np.ceil(np.linspace(0, n_position-1,target_size))]
            pattern_sum = torch.zeros_like(pattern)
            n_position = target_size
        
        # averaging the adjacient embeddings if needed (optional, almost the same performance)
        arange = torch.arange(n_position)
        pooling = [0] if not mean_pooling else[-2, -1, 0, 1, 2]
        time = 0
        for i in pooling:
            time += 1
            index = (arange + i + n_position) % n_position
            pattern_sum += pattern.gather(0, index.view(-1,1).expand_as(pattern))
        pattern = 1. / time * pattern_sum - pattern.mean(0)
        
        return pattern    

    def position_encoding(self, solutions, embedding_dim, visited_time):
        
         # batch: batch_size, problem_size, dim
         batch_size, seq_length = solutions.size()
         arange = torch.arange(batch_size)
         
         # expand for every batch
         CPE_embeddings = self.pattern.expand(batch_size, seq_length, embedding_dim).clone().to(solutions.device)
         
         # get index according to the solutions
         if visited_time is None:
             visited_time = torch.zeros((batch_size,seq_length),device = solutions.device)
             pre = torch.zeros((batch_size),device = solutions.device).long()
             for i in range(seq_length):
                visited_time[arange,solutions[arange,pre]] = i+1
                pre = solutions[arange,pre]
         index = (visited_time % seq_length).long().unsqueeze(-1).expand(batch_size, seq_length, embedding_dim)
         
         return torch.gather(CPE_embeddings, 1, index), visited_time.long()

        
    def forward(self, x, solutions, visited_time = None):
        PFEs, visited_time = self.position_encoding(solutions, self.embedding_dim, visited_time)
        NFEs = self.embedder(x)
        return  NFEs, PFEs, visited_time
