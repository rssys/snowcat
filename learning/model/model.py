import os.path as osp
import torch
from fairseq.models.roberta import RobertaModel
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GINEConv
from torch.nn import BatchNorm1d as BatchNorm
from torch.nn import Linear, ReLU, Sequential
from torch import Tensor


class BertGAT(torch.nn.Module):
    def __init__(self, bert_ckpt_dirpath, bert_ckpt_filename, hidden_channels, num_gnn_layers, num_edge_type=6):
        super().__init__()
        # bert model
        self.bert = RobertaModel.from_pretrained(bert_ckpt_dirpath, checkpoint_file=bert_ckpt_filename)
        BERT_EMBED_DIM = self.bert.model.args.encoder_embed_dim
        EDGE_EMBED_DIM = BERT_EMBED_DIM
        # a embedding layer for edge types
        self.edge_attr_embedding = torch.nn.Embedding(num_edge_type, EDGE_EMBED_DIM)

        num_heads = 4
        self.input_gnn = GATConv(BERT_EMBED_DIM, hidden_channels, edge_dim=EDGE_EMBED_DIM, heads=num_heads)
        self.output_gnn = GATConv(hidden_channels * num_heads, 1, edge_dim=EDGE_EMBED_DIM)
        assert num_gnn_layers > 2
        self.hidden_gnns = torch.nn.ModuleList()
        for _ in range(num_gnn_layers - 2):
            conv = GATConv(hidden_channels * num_heads, hidden_channels, edge_dim=EDGE_EMBED_DIM, heads=num_heads)
            self.hidden_gnns.append(conv)


    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # edge_index: graph connectivity matrix of shape [2, num_edges]
        # x.shape = [num_nodes, num_tokens]
        x = self.bert.extract_features(x)[:, 0, :]
        # x.shape = [num_nodes, BERT_EMBED_DIM]

        # edge_attr.shape = [num_edges, 1]
        edge_attr = self.edge_attr_embedding(edge_attr)[:, 0, :]
        # edge_attr.shape = [num_edges, EDGE_EMBED_DIM]

        # x.shape = [num_nodes, BERT_EMBED_DIM]
        x = self.input_gnn(x, edge_index, edge_attr).relu()
        for conv in self.hidden_gnns:
            x = conv(x, edge_index, edge_attr).relu()
        x = self.output_gnn(x, edge_index, edge_attr)

        # x.shape = [num_nodes, 1]
        return x



class BertGATv2(torch.nn.Module):
    def __init__(self, bert_ckpt_dirpath, bert_ckpt_filename, hidden_channels, num_gnn_layers, num_edge_type=6):
        super().__init__()
        # bert model
        self.bert = RobertaModel.from_pretrained(bert_ckpt_dirpath, checkpoint_file=bert_ckpt_filename)
        BERT_EMBED_DIM = self.bert.model.args.encoder_embed_dim
        EDGE_EMBED_DIM = BERT_EMBED_DIM
        # a embedding layer for edge types
        self.edge_attr_embedding = torch.nn.Embedding(num_edge_type, EDGE_EMBED_DIM)

        num_heads = 4
        self.input_gnn = GATv2Conv(BERT_EMBED_DIM, hidden_channels, edge_dim=EDGE_EMBED_DIM, heads=num_heads)
        self.output_gnn = GATv2Conv(hidden_channels * num_heads, 1, edge_dim=EDGE_EMBED_DIM)
        assert num_gnn_layers > 2
        self.hidden_gnns = torch.nn.ModuleList()
        for _ in range(num_gnn_layers - 2):
            conv = GATv2Conv(hidden_channels * num_heads, hidden_channels, edge_dim=EDGE_EMBED_DIM, heads=num_heads)
            self.hidden_gnns.append(conv)


    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # edge_index: graph connectivity matrix of shape [2, num_edges]
        # x.shape = [num_nodes, num_tokens]
        x = self.bert.extract_features(x)[:, 0, :]
        # x.shape = [num_nodes, BERT_EMBED_DIM]

        # edge_attr.shape = [num_edges, 1]
        edge_attr = self.edge_attr_embedding(edge_attr)[:, 0, :]
        # edge_attr.shape = [num_edges, EDGE_EMBED_DIM]

        # x.shape = [num_nodes, BERT_EMBED_DIM]
        x = self.input_gnn(x, edge_index, edge_attr).relu()
        for conv in self.hidden_gnns:
            x = conv(x, edge_index, edge_attr).relu()
        x = self.output_gnn(x, edge_index, edge_attr)

        # x.shape = [num_nodes, 1]
        return x



class BertTrans(torch.nn.Module):
    def __init__(self, bert_ckpt_dirpath, bert_ckpt_filename, hidden_channels, num_gnn_layers, num_edge_type=6):
        super().__init__()
        # bert model
        self.bert = RobertaModel.from_pretrained(bert_ckpt_dirpath, checkpoint_file=bert_ckpt_filename)
        BERT_EMBED_DIM = self.bert.model.args.encoder_embed_dim
        EDGE_EMBED_DIM = BERT_EMBED_DIM
        # a embedding layer for edge types
        self.edge_attr_embedding = torch.nn.Embedding(num_edge_type, EDGE_EMBED_DIM)

        num_heads = 4
        self.input_gnn = TransformerConv(BERT_EMBED_DIM, hidden_channels, edge_dim=EDGE_EMBED_DIM, heads=num_heads)
        self.output_gnn = TransformerConv(hidden_channels * num_heads, 1, edge_dim=EDGE_EMBED_DIM)
        assert num_gnn_layers > 2
        self.hidden_gnns = torch.nn.ModuleList()
        for _ in range(num_gnn_layers - 2):
            conv = TransformerConv(hidden_channels * num_heads, hidden_channels, edge_dim=EDGE_EMBED_DIM, heads=num_heads)
            self.hidden_gnns.append(conv)


    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # edge_index: graph connectivity matrix of shape [2, num_edges]
        # x.shape = [num_nodes, num_tokens]
        x = self.bert.extract_features(x)[:, 0, :]
        # x.shape = [num_nodes, BERT_EMBED_DIM]

        # edge_attr.shape = [num_edges, 1]
        edge_attr = self.edge_attr_embedding(edge_attr)[:, 0, :]
        # edge_attr.shape = [num_edges, EDGE_EMBED_DIM]

        # x.shape = [num_nodes, BERT_EMBED_DIM]
        x = self.input_gnn(x, edge_index, edge_attr).relu()
        for conv in self.hidden_gnns:
            x = conv(x, edge_index, edge_attr).relu()
        x = self.output_gnn(x, edge_index, edge_attr)

        # x.shape = [num_nodes, 1]
        return x

class BertGINE(torch.nn.Module):
    def __init__(self, bert_ckpt_dirpath, bert_ckpt_filename, hidden_channels, num_gnn_layers, num_edge_type=6):
        super().__init__()
        # bert model
        self.bert = RobertaModel.from_pretrained(bert_ckpt_dirpath, checkpoint_file=bert_ckpt_filename)
        BERT_EMBED_DIM = self.bert.model.args.encoder_embed_dim
        EDGE_EMBED_DIM = BERT_EMBED_DIM
        # a embedding layer for edge types
        self.edge_attr_embedding = torch.nn.Embedding(num_edge_type, EDGE_EMBED_DIM)

        nn = Sequential(
                Linear(BERT_EMBED_DIM, hidden_channels * 4, bias=False),
                BatchNorm(hidden_channels * 4),
                ReLU(),
                Linear(hidden_channels * 4, hidden_channels * 4, bias=False),
                )
        self.input_gnn = GINEConv(nn, edge_dim=EDGE_EMBED_DIM, train_eps=True)

        nn = Sequential(
                Linear(hidden_channels * 4, 1, bias=False),
                )
        self.output_gnn = GINEConv(nn, edge_dim=EDGE_EMBED_DIM, train_eps=True)

        assert num_gnn_layers > 2
        self.hidden_gnns = torch.nn.ModuleList()
        for _ in range(num_gnn_layers - 2):
            nn = Sequential(
                    Linear(hidden_channels * 4, hidden_channels * 4, bias=False),
                    BatchNorm(hidden_channels * 4),
                    ReLU(),
                    Linear(hidden_channels * 4, hidden_channels * 4, bias=False),
                    BatchNorm(hidden_channels * 4),
                    )
            conv = GINEConv(nn, edge_dim=EDGE_EMBED_DIM, train_eps=True)
            self.hidden_gnns.append(conv)


    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # edge_index: graph connectivity matrix of shape [2, num_edges]
        # x.shape = [num_nodes, num_tokens]
        x = self.bert.extract_features(x)[:, 0, :]
        # x.shape = [num_nodes, BERT_EMBED_DIM]

        # edge_attr.shape = [num_edges, 1]
        edge_attr = self.edge_attr_embedding(edge_attr)[:, 0, :]
        # edge_attr.shape = [num_edges, EDGE_EMBED_DIM]

        # x.shape = [num_nodes, BERT_EMBED_DIM]
        x = self.input_gnn(x, edge_index, edge_attr).relu()
        for conv in self.hidden_gnns:
            x = conv(x, edge_index, edge_attr).relu()
        x = self.output_gnn(x, edge_index, edge_attr)

        # x.shape = [num_nodes, 1]
        return x


def create_model(model_config):
    """Create the model instance"""
    PRETRAINED_BERT_FILEPATH = model_config["pretrained_bert_ckpt_filepath"]
    head_tail = osp.split(PRETRAINED_BERT_FILEPATH)
    bert_ckpt_dirpath = head_tail[0]
    bert_ckpt_filename = head_tail[1]
    hidden_dim_size = int(model_config["hidden_dim_size"])
    num_gnn_layer = int(model_config["num_gnn_layer"])
    gnn_arch = model_config["gnn_arch"].lower()
    if gnn_arch == "transformer":
        model = BertTrans(bert_ckpt_dirpath, bert_ckpt_filename, \
                hidden_dim_size, num_gnn_layer)
    elif gnn_arch == "gine":
        model = BertGINE(bert_ckpt_dirpath, bert_ckpt_filename, \
                hidden_dim_size, num_gnn_layer)
    elif gnn_arch == "gatv2":
        model = BertGATv2(bert_ckpt_dirpath, bert_ckpt_filename, \
                hidden_dim_size, num_gnn_layer)
    elif gnn_arch == "gat":
        model = BertGAT(bert_ckpt_dirpath, bert_ckpt_filename, \
                hidden_dim_size, num_gnn_layer)
    else:
        print(f"{gnn_arch} is not supported")
        exit(0)
    return model
