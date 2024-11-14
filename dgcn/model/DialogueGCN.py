import torch
import torch.nn as nn

from .SeqContext import SeqContext
from .EdgeAtt import EdgeAtt
from .GCN import GCN
from .Classifier import Classifier
from .functions import batch_graphify
from .BIM import BIMModule
import dgcn

log = dgcn.utils.get_logger()


class DialogueGCN(nn.Module):

    def __init__(self, args):
        super(DialogueGCN, self).__init__()
        u_dim = 768
        g_dim = 128
        h1_dim = 64
        h2_dim = 64
        hc_dim = 64
        tag_size = 7

        self.wp = args.wp
        self.wf = args.wf
        self.device = args.device

        self.rnn = SeqContext(u_dim, g_dim, args)
        self.edge_att = EdgeAtt(g_dim, args)
        self.gcn = GCN(g_dim, h1_dim, h2_dim, args)

        self.bim = BIMModule(f_dim=g_dim, v_dim=h2_dim)
        
        # Keep original classifier
        self.clf = Classifier(g_dim + h2_dim, hc_dim, tag_size, args)

        edge_type_to_idx = {}
        for j in range(args.n_speakers):
            for k in range(args.n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx
        log.debug(self.edge_type_to_idx)

    def get_rep(self, data):
        node_features = self.rnn(data["text_len_tensor"], data["text_tensor"])  # [batch_size, mx_len, D_g]
        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
            node_features, data["text_len_tensor"], data["speaker_tensor"], self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att, self.device)

        graph_out = self.gcn(features, edge_index, edge_norm, edge_type)

        # Apply BIM before concatenation
        features, graph_out = self.bim(features, graph_out)  # [mx_len, D_g], [mx_len, h2_dim]

        return features, graph_out

    def forward(self, data):
        graph_out, features = self.get_rep(data)
        out = self.clf(torch.cat([features, graph_out], dim=-1), data["text_len_tensor"])
        return out

    def get_loss(self, data):
        graph_out, features = self.get_rep(data)
        loss = self.clf.get_loss(torch.cat([features, graph_out], dim=-1),
                                 data["label_tensor"], data["text_len_tensor"])
        return loss