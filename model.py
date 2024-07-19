import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU() if act == 'prelu' else act
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        
        return self.act(out)


class AvgReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 1)


class MaxReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(MaxReadout, self).__init__()

    def forward(self, seq):
        return torch.max(seq,1).values


class MinReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(MinReadout, self).__init__()

    def forward(self, seq):
        return torch.min(seq, 1).values


class WSReadout(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self):
        super(WSReadout, self).__init__()

    def forward(self, seq, query):
        query = query.permute(0,2,1)
        sim = torch.matmul(seq,query)
        sim = F.softmax(sim,dim=1)
        sim = sim.repeat(1, 1, 64)
        out = torch.mul(seq,sim)
        out = torch.sum(out,1)
        return out


class Discriminator(nn.Module):
    """
    Forked from GRAND-Lab/CoLA
    """
    def __init__(self, n_h, negsamp_round):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

        self.negsamp_round = negsamp_round

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, s_bias1=None, s_bias2=None):
        scs = []
        scs.append(self.f_k(h_pl, c))
        c_mi = c
        for _ in range(self.negsamp_round):
            c_mi = torch.cat((c_mi[-1, :].unsqueeze(0), c_mi[:-1, :]), dim=0)
            scs.append(self.f_k(h_pl, c_mi))
        logits = torch.cat(tuple(scs))
        return logits


class Model(nn.Module):
    def __init__(self, feat_size, motifs_size, hidden_dim, subgraph_size, negsamp_ratio, activation, readout):
        super(Model, self).__init__()
        self.read_mode = readout
        self.encoder_motif = GCN(motifs_size, hidden_dim, activation)
        self.encoder_feat = GCN(feat_size, hidden_dim, activation)
        self.MLP_motifs = nn.Sequential(
            nn.Linear(hidden_dim * (subgraph_size - 1), hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, motifs_size),
            nn.PReLU()
        )
        self.MLP_feat = nn.Sequential(
            nn.Linear(hidden_dim * (subgraph_size - 1), hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, feat_size),
            nn.PReLU()
        )
        self.TRL = nn.Sequential(nn.Linear(hidden_dim, 1))
        if readout == 'max':
            self.read = MaxReadout()
        elif readout == 'min':
            self.read = MinReadout()
        elif readout == 'avg':
            self.read = AvgReadout()
        elif readout == 'weighted_sum':
            self.read = WSReadout()

        self.disc = Discriminator(hidden_dim, negsamp_ratio)

    def forward(self, features, motifs, adj_feat, adj_motif, sparse=False):
        hidden_motifs = self.encoder_motif(motifs, adj_motif, sparse)
        hidden_features = self.encoder_feat(features, adj_feat, sparse)
        sub_size = hidden_features.shape[1]
        batch = hidden_motifs.shape[0]
        features_neibors = hidden_features[:, :sub_size - 2, :]
        neibors_feat = features_neibors.reshape(batch, -1)
        motifs_neibors = hidden_motifs[:, :sub_size - 2, :]
        neibors_motif = motifs_neibors.reshape(batch, -1)



        motifs_rec = self.MLP_motifs(neibors_motif)
        feat_rec = self.MLP_feat(neibors_feat)

        if self.read_mode != 'weighted_sum':
            neibors_feat_readout = torch.squeeze(torch.bmm(torch.transpose(torch.sigmoid(self.TRL(hidden_motifs[:, :sub_size - 1, :] - hidden_motifs[:, -1:, :])), 1, 2), hidden_features[:, :sub_size - 1, :]), 1)
            target_node_feat = hidden_features[:, -1, :]

        logits = self.disc(neibors_feat_readout, target_node_feat)

        return logits, motifs_rec, feat_rec