import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GATConv
from transformers import BertModel, BertForMaskedLM
import torch_geometric_temporal.nn.recurrent.evolvegcnh
from data_helpers import isin
import torch_geometric_temporal.nn.recurrent.evolvegcno
from typing import Optional, Tuple
from torch import Tensor
from torch.nn import GRU
from torch_geometric.typing import Adj, OptTensor
from torch_sparse import SparseTensor
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class MLMModel(nn.Module):
    """"Class to train dynamic contextualized word embeddings with masked language modeling."""

    def __init__(self, n_times=1, social_dim=50, gnn=None, social_only=False, time_only=False):
        """Initialize dynamic contextualized word embeddings model.

        Args:
            n_times: number of time points (only relevant if time is not ablated)
            social_dim: dimensionality of social embeddings
            gnn: type of GNN (currently 'gat' and 'gcn' are possible)
            social_only: use only social information (temporal ablation)
            time_only: use only temporal information (social ablation)
        """

        super(MLMModel, self).__init__()

        # For ablated models
        self.social_only = social_only
        self.time_only = time_only

        # Contextualizing component
        self.bert = BertForMaskedLM.from_pretrained('/home/ashish/Desktop/Desktop/bert-pretrained')
        self.bert_emb_layer = self.bert.get_input_embeddings()

        # Dynamic component
        if self.social_only:
            self.social_component = SocialComponent(social_dim, gnn)
        elif self.time_only:
            self.offset_components = nn.ModuleList([OffsetComponent() for _ in range(n_times)])
        else:
            self.social_components = nn.ModuleList([SocialComponent(social_dim, gnn) for _ in range(n_times)])

    def forward(self, labels, reviews, masks, segs, users, g_data, times, vocab_filter, embs_only=False):
        """Perform forward pass.

        Args:
            labels: tensor of masked language modeling labels
            reviews: tensor of tokenized reviews
            masks: tensor of attention masks
            segs: tensor of segment indices
            users: tensor of batch user indices
            g_data: graph data object
            times: tensor of batch time points
            vocab_filter: tensor with word types for dynamic component
            embs_only: only compute dynamic type-level embeddings
        """

        # Retrieve BERT input embeddings
        bert_embs = self.bert_emb_layer(reviews)

        # Temporal ablation
        if self.social_only:
            offset_last = None  # No need to compute embeddings at last time point for temporal ablation
            offset_now = torch.cat(
                [self.social_component(bert_embs[i], users[i], g_data) for i, j in enumerate(times)],
                dim=0
            )
            offset_now = offset_now * isin(reviews, vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)

        # Social ablation
        elif self.time_only:
            offset_last = torch.cat(
                [self.offset_components[j](bert_embs[i]) for i, j in enumerate(F.relu(times - 1))],
                dim=0
            )
            offset_now = torch.cat(
                [self.offset_components[j](bert_embs[i]) for i, j in enumerate(times)],
                dim=0
            )
            offset_last = offset_last * isin(reviews, vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)
            offset_now = offset_now * isin(reviews, vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)

        # Full dynamic component
        else:
            offset_last = torch.cat(
                [self.social_components[j](bert_embs[i], users[i], g_data) for i, j in enumerate(F.relu(times - 1))],
                dim=0
            )
            offset_now = torch.cat(
                [self.social_components[j](bert_embs[i], users[i], g_data) for i, j in enumerate(times)],
                dim=0
            )
            offset_last = offset_last * isin(reviews, vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)
            offset_now = offset_now * isin(reviews, vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)

        # Compute dynamic type-level embeddings (input to contextualizing component)
        input_embs = bert_embs + offset_now

        # Only compute dynamic type-level embeddings (not fed into contextualizing component)
        if embs_only:
            return bert_embs, input_embs

        # Pass through contextualizing component
        output = self.bert(inputs_embeds=input_embs, attention_mask=masks, token_type_ids=segs, masked_lm_labels=labels)

        return offset_last, offset_now, output[0]


class SAModel(nn.Module):
    """"Class to train dynamic contextualized word embeddings for sentiment analysis."""

    def __init__(self, n_times=1, social_dim=50, gnn=None):
        """Initialize dynamic contextualized word embeddings model.

        Args:
            n_times: number of time points
            social_dim: dimensionality of social embeddings
            gnn: type of GNN (currently 'gat' and 'gcn' are possible)
        """

        super(SAModel, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert_emb_layer = self.bert.get_input_embeddings()
        self.social_components = nn.ModuleList([SocialComponent(social_dim, gnn) for _ in range(n_times)])
        self.linear_1 = nn.Linear(768, 100)
        self.linear_2 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, reviews, masks, segs, users, g_data, times, vocab_filter, embs_only=False):
        """Perform forward pass.

        Args:
            reviews: tensor of tokenized reviews
            masks: tensor of attention masks
            segs: tensor of segment indices
            users: tensor of batch user indices
            g_data: graph data object
            times: tensor of batch time points
            vocab_filter: tensor with word types for dynamic component
            embs_only: only compute dynamic type-level embeddings
        """

        # Retrieve BERT input embeddings
        bert_embs = self.bert_emb_layer(reviews)
        offset_last = torch.cat(
            [self.social_components[j](bert_embs[i], users[i], g_data) for i, j in enumerate(F.relu(times - 1))],
            dim=0
        )
        offset_now = torch.cat(
            [self.social_components[j](bert_embs[i], users[i], g_data) for i, j in enumerate(times)],
            dim=0
        )
        offset_last = offset_last * isin(reviews, vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)
        offset_now = offset_now * isin(reviews, vocab_filter).float().unsqueeze(-1).expand(-1, -1, 768)

        # Compute dynamic type-level embeddings (input to contextualizing component)
        input_embs = bert_embs + offset_now

        # Only compute dynamic type-level embeddings (not fed into contextualizing component)
        if embs_only:
            return bert_embs, input_embs

        # Pass through contextualizing component
        output_bert = self.dropout(self.bert(inputs_embeds=input_embs, attention_mask=masks, token_type_ids=segs)[1])
        h = self.dropout(torch.tanh(self.linear_1(output_bert)))
        output = torch.sigmoid(self.linear_2(h)).squeeze(-1)

        return offset_last, offset_now, output


class SocialComponent(nn.Module):
    """"Class implementing the social part of the dynamic component."""

    def __init__(self, social_dim=50, gnn=None):
        super(SocialComponent, self).__init__()
        self.gnn_component = GNNComponent(social_dim, gnn)
        self.linear_1 = nn.Linear(768 + social_dim, 768)
        self.linear_2 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)

    def forward(self, embs, users, graph_data):
        user_output = self.gnn_component(users, graph_data)
        user_output = user_output.unsqueeze(0).expand(embs.size(0), -1)
        h = torch.cat((embs, user_output), dim=-1)
        h = self.dropout(torch.tanh(self.linear_1(h)))
        offset = self.linear_2(h).unsqueeze(0)
        return offset


class GNNComponent(nn.Module):
    """"Class implementing the GNN of the dynamic component."""

    def __init__(self, social_dim=50, gnn=None):
        super(GNNComponent, self).__init__()
        self.social_dim = social_dim
        self.gnn = gnn
        if self.gnn == 'gcn':
            self.conv_1 = GCNConv(self.social_dim, self.social_dim)
            self.conv_2 = GCNConv(self.social_dim, self.social_dim)
            
        elif self.gnn == 'gat':
            self.conv_1 = GATConv(self.social_dim, self.social_dim, heads=4, dropout=0.6, concat=False)
            self.conv_2 = GATConv(self.social_dim, self.social_dim, heads=4, dropout=0.6, concat=False)
            
        elif self.gnn == 'roland':
            self.conv_1 = Roland(social_dim)
            self.conv_2 = Roland(social_dim)
            
        else:
            self.linear_1 = nn.Linear(self.social_dim, self.social_dim)
            self.linear_2 = nn.Linear(self.social_dim, self.social_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, users, graph_data):
        if self.gnn == 'gcn' or self.gnn == 'gat' or self.gnn == 'roland' :
            h = self.dropout(torch.tanh(self.conv_1(graph_data.x, graph_data.edge_index)))
            return self.dropout(torch.tanh(self.conv_2(h, graph_data.edge_index)))[users]
        else:
            h = self.dropout(torch.tanh(self.linear_1(graph_data.x[users])))
            return self.dropout(torch.tanh(self.linear_2(h)))


class OffsetComponent(nn.Module):
    """"Class implementing the dynamic component for social ablation."""

    def __init__(self):
        super(OffsetComponent, self).__init__()
        self.linear_1 = nn.Linear(768, 768)
        self.linear_2 = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.2)

    def forward(self, embs):
        h = self.dropout(torch.tanh(self.linear_1(embs)))
        offset = self.linear_2(h).unsqueeze(0)
        return offset


class SABert(nn.Module):
    """"Class to train non-dynamic contextualized word embeddings (BERT) for sentiment analysis."""

    def __init__(self):
        super(SABert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear_1 = nn.Linear(768, 100)
        self.linear_2 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, reviews, masks, segs):
        output_bert = self.dropout(self.bert(reviews, attention_mask=masks, token_type_ids=segs)[1])
        h = self.dropout(torch.tanh(self.linear_1(output_bert)))
        output = torch.sigmoid(self.linear_2(h)).squeeze(-1)
        return output
        

class GCNConv_Fixed_W(MessagePassing):
 

    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv_Fixed_W, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, W: torch.FloatTensor, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            cache = self._cached_edge_index
            if cache is None:
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    self.improved, self.add_self_loops)

        x = torch.matmul(x, W)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        return out


    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j




class Roland(torch.nn.Module):


    def __init__(
        self,
        in_channels: int,
        improved: bool = False,
        cached: bool = False,
        normalize: bool = True,
        add_self_loops: bool = True,
    ):
        super(Roland, self).__init__()

        self.in_channels = in_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.weight = None
        self.initial_weight = torch.nn.Parameter(torch.Tensor(in_channels, in_channels))
        self._create_layers()
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.initial_weight)


    def _create_layers(self):

        self.recurrent_layer = GRU(
            input_size=self.in_channels, hidden_size=self.in_channels, num_layers=1
        )
        for param in self.recurrent_layer.parameters():
            param.requires_grad = True
            param.retain_grad()

        self.conv_layer = GCNConv_Fixed_W(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            improved=self.improved,
            cached=self.cached,
            normalize=self.normalize,
            add_self_loops=self.add_self_loops
        )

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:

        
        if self.weight is None:
            self.weight = self.initial_weight.data
        W = self.weight[None, :, :]
        _, W = self.recurrent_layer(W, W)
        X = self.conv_layer(W.squeeze(dim=0), X, edge_index, edge_weight)
    
        return X  
