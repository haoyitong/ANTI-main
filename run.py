from model import Model
from utils import *
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import random
import os
import argparse
from tqdm import tqdm
import dgl
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set argument
parser = argparse.ArgumentParser(description='ANTI')
parser.add_argument('--dataset', type=str, default='dblp')
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--embedding_dim', type=int, default='64')
parser.add_argument('--num_epoch', type=int)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--auc_test_rounds', type=int, default=50)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--alpha', type=float)
parser.add_argument('--beta', type=float)

args = parser.parse_args()


if __name__ == '__main__':
    batch_size = args.batch_size
    subgraph_size = args.subgraph_size
    # Set random seed
    dgl.random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    print('Dataset: {}  and emebding_dim{}'.format(args.dataset, args.embedding_dim), flush=True)
    if args.dataset == 'cora':
        args.lr = 1e-3
        args.num_epoch = 100
        args.alpha = 0.1
        args.beta = 0.9
    
    print('alpha:{}, beta:{}'.format(args.alpha, args.beta))
    subgraph_size = args.subgraph_size
    # Load and preprocess data
    adj, features, motifs, _, _, _, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

    scaler_raw_features = MinMaxScaler()
    scaler_raw_motifs = MinMaxScaler()

    raw_feature = features.todense()
    raw_motifs = motifs.todense()
    raw_feature = scaler_raw_features.fit_transform(raw_feature)
    raw_motifs = scaler_raw_motifs.fit_transform(raw_motifs)

    features, _ = preprocess_features(features)
    motifs, _ = preprocess_features(motifs)

    dgl_graph = adj_to_dgl_graph(adj)
    nb_nodes = features.shape[0]
    ft_size = features.shape[1]
    motifs_size = motifs.shape[1]

    adj = normalize_adj(adj)
    adj = (adj + sp.eye(adj.shape[0])).todense()
    device = 'cuda:0'
    features = torch.FloatTensor(features[np.newaxis]).to(device)
    raw_feature = torch.FloatTensor(raw_feature[np.newaxis]).to(device)
    motifs = torch.FloatTensor(motifs[np.newaxis]).to(device)
    raw_motifs = torch.FloatTensor(raw_motifs[np.newaxis]).to(device)
    adj = torch.Tensor(adj[np.newaxis]).to(device)

    # Initialize model and optimiser
    model = Model(ft_size, motifs_size, args.embedding_dim, subgraph_size, args.negsamp_ratio, 'prelu',
                                   args.readout)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if torch.cuda.is_available():
        print('Using CUDA')
        model.cuda()
        features = features.cuda()
        raw_feature = raw_feature.cuda()
        motifs = motifs.cuda()
        raw_motif = raw_motifs.cuda()
        adj = adj.cuda()

    if torch.cuda.is_available():
        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
    else:
        b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))
    xent = nn.CrossEntropyLoss()

    cnt_wait = 0
    best = 1e9
    best_t = 0
    batch_num = nb_nodes // batch_size + 1

    added_adj_zero_row = torch.zeros((nb_nodes, 1, subgraph_size))
    added_adj_zero_col = torch.zeros((nb_nodes, subgraph_size + 1, 1))
    added_adj_zero_col[:, -1, :] = 1.
    added_feat_zero_row = torch.zeros((nb_nodes, 1, ft_size))

    mse_loss = nn.MSELoss(reduction='mean')

    # Train model
    with tqdm(total=args.num_epoch) as pbar:
        pbar.set_description('Training')
        for epoch in range(args.num_epoch):

            loss_full_batch = torch.zeros((nb_nodes, 1))
            if torch.cuda.is_available():
                loss_full_batch = loss_full_batch.cuda()

            model.train()

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            total_loss = 0.

            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                lbl = torch.unsqueeze(
                    torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))),
                    1)

                batch_adj_feat = []
                batch_adj_motifs = []
                batch_feat = []
                batch_motifs = []
                batch_raw_feat = []
                batch_raw_motifs = []

                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))
                added_motifs_zero_row = torch.zeros((cur_batch_size, 1, motifs_size))

                if torch.cuda.is_available():
                    lbl = lbl.cuda()
                    added_adj_zero_row = added_adj_zero_row.cuda()
                    added_adj_zero_col = added_adj_zero_col.cuda()
                    added_feat_zero_row = added_feat_zero_row.cuda()
                    added_motifs_zero_row = added_motifs_zero_row.cuda()
                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    cur_motifs = motifs[:, subgraphs[i], :]
                    cur_raw_feat = raw_feature[:, subgraphs[i], :]
                    cur_raw_motifs = raw_motifs[:, subgraphs[i], :]

                    batch_adj_feat.append(cur_adj)
                    batch_adj_motifs.append(cur_adj)
                    batch_feat.append(cur_feat)
                    batch_motifs.append(cur_motifs)
                    batch_raw_feat.append(cur_raw_feat)
                    batch_raw_motifs.append(cur_raw_motifs)

                batch_adj_feat = torch.cat(batch_adj_feat)
                batch_adj_feat = torch.cat((batch_adj_feat, added_adj_zero_row), dim=1)
                batch_adj_feat = torch.cat((batch_adj_feat, added_adj_zero_col), dim=2)

                batch_adj_motifs = torch.cat(batch_adj_motifs)
                batch_adj_motifs = torch.cat((batch_adj_motifs, added_adj_zero_row), dim=1)
                batch_adj_motifs = torch.cat((batch_adj_motifs, added_adj_zero_col), dim=2)

                batch_feat = torch.cat(batch_feat)
                batch_feat = torch.cat((batch_feat[:, :-1, :], added_feat_zero_row, batch_feat[:, -1:, :]),
                                       dim=1)

                batch_motifs = torch.cat(batch_motifs)
                batch_motifs = torch.cat(
                    (batch_motifs[:, :-1, :], added_motifs_zero_row, batch_motifs[:, -1:, :]), dim=1)

                batch_raw_feat = torch.cat(batch_raw_feat)
                batch_raw_feat = torch.cat(
                    (batch_raw_feat[:, :-1, :], added_feat_zero_row, batch_raw_feat[:, -1:, :]), dim=1)

                batch_raw_motifs = torch.cat(batch_raw_motifs)
                batch_raw_motifs = torch.cat(
                    (batch_raw_motifs[:, :-1, :], added_motifs_zero_row, batch_raw_motifs[:, -1:, :]), dim=1)

                logits, motifs_rec, feat_rec = model(batch_feat, batch_motifs, batch_adj_feat, batch_adj_motifs,
                                                     sparse=False)
                batch = motifs_rec.shape[0]
                loss_con = torch.mean(b_xent(logits, lbl))
                loss_rec = args.beta * mse_loss(motifs_rec, batch_raw_motifs[:, -1, :]) + (
                        1 - args.beta) * mse_loss(feat_rec, batch_raw_feat[:, -1, :])
                loss = args.alpha * loss_con + (1 - args.alpha) * loss_rec
                # loss_rec = mse_loss(motifs_rec, batch_raw_motifs[:, -1, :]) + mse_loss(feat_rec, batch_raw_feat[:, -1, :])
                # loss = loss_con + loss_rec
                loss.backward()
                optimiser.step()
                loss = loss.detach().cpu().numpy()

                if not is_final_batch:
                    total_loss += loss

            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes

            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), 'best_model.pkl')
            else:
                cnt_wait += 1

            pbar.set_postfix(loss=mean_loss)
            pbar.update(1)

    # Test model
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_model.pkl'))

    multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
    multi_round_ano_score_p = np.zeros((args.auc_test_rounds, nb_nodes))
    multi_round_ano_score_n = np.zeros((args.auc_test_rounds, nb_nodes))

    with tqdm(total=args.auc_test_rounds) as pbar_test:
        pbar_test.set_description('Testing')
        for round in range(args.auc_test_rounds):

            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)

            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))

                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                batch_adj_feat = []
                batch_adj_motifs = []
                batch_feat = []
                batch_motifs = []
                batch_raw_feat = []
                batch_raw_motifs = []

                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))
                added_motifs_zero_row = torch.zeros((cur_batch_size, 1, motifs_size))

                if torch.cuda.is_available():
                    lbl = lbl.cuda()
                    added_adj_zero_row = added_adj_zero_row.cuda()
                    added_adj_zero_col = added_adj_zero_col.cuda()
                    added_feat_zero_row = added_feat_zero_row.cuda()
                    added_motifs_zero_row = added_motifs_zero_row.cuda()
                for i in idx:
                    cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                    cur_feat = features[:, subgraphs[i], :]
                    cur_motifs = motifs[:, subgraphs[i], :]
                    cur_raw_feat = raw_feature[:, subgraphs[i], :]
                    cur_raw_motifs = raw_motifs[:, subgraphs[i], :]

                    batch_adj_feat.append(cur_adj)
                    batch_adj_motifs.append(cur_adj)
                    batch_feat.append(cur_feat)
                    batch_motifs.append(cur_motifs)
                    batch_raw_feat.append(cur_raw_feat)
                    batch_raw_motifs.append(cur_raw_motifs)

                batch_adj_feat = torch.cat(batch_adj_feat)
                batch_adj_feat = torch.cat((batch_adj_feat, added_adj_zero_row), dim=1)
                batch_adj_feat = torch.cat((batch_adj_feat, added_adj_zero_col), dim=2)

                batch_adj_motifs = torch.cat(batch_adj_motifs)
                batch_adj_motifs = torch.cat((batch_adj_motifs, added_adj_zero_row), dim=1)
                batch_adj_motifs = torch.cat((batch_adj_motifs, added_adj_zero_col), dim=2)

                batch_feat = torch.cat(batch_feat)
                batch_feat = torch.cat((batch_feat[:, :-1, :], added_feat_zero_row, batch_feat[:, -1:, :]),
                                       dim=1)

                batch_motifs = torch.cat(batch_motifs)
                batch_motifs = torch.cat(
                    (batch_motifs[:, :-1, :], added_motifs_zero_row, batch_motifs[:, -1:, :]), dim=1)

                batch_raw_feat = torch.cat(batch_raw_feat)
                batch_raw_feat = torch.cat(
                    (batch_raw_feat[:, :-1, :], added_feat_zero_row, batch_raw_feat[:, -1:, :]),
                    dim=1)

                batch_raw_motifs = torch.cat(batch_raw_motifs)
                batch_raw_motifs = torch.cat(
                    (batch_raw_motifs[:, :-1, :], added_motifs_zero_row, batch_raw_motifs[:, -1:, :]), dim=1)

                with torch.no_grad():
                    logits, motifs_rec, feat_rec = model(batch_feat, batch_motifs, batch_adj_feat,
                                                         batch_adj_motifs, sparse=False)
                    logits = torch.squeeze(logits)
                    logits = torch.sigmoid(logits)

                pdist = nn.PairwiseDistance(p=2)

                scaler_rec = MinMaxScaler()
                score_rec = args.beta * pdist(motifs_rec, batch_raw_motifs[:, -1, :]) + (1 - args.beta) * pdist(
                    feat_rec, batch_raw_feat[:, -1, :])
                score_rec = score_rec.cpu().numpy()
                ano_score_rec = scaler_rec.fit_transform(score_rec.reshape(-1, 1)).reshape(-1)

                scaler_con = MinMaxScaler()
                score_con = - (logits[:cur_batch_size] - logits[cur_batch_size:]).cpu().numpy()
                ano_score_con = scaler_con.fit_transform(score_con.reshape(-1, 1)).reshape(-1)

                ano_score = args.alpha * ano_score_con + (1 - args.alpha) * ano_score_rec
                multi_round_ano_score[round, idx] = ano_score

            pbar_test.update(1)

    ano_score_final = np.mean(multi_round_ano_score, axis=0)

    precisions = precision(ano_score_final, ano_label, (50, 100, 150))
    recalls = recall(ano_score_final, ano_label, (50, 100, 150))
    auc_final = roc_auc_score(ano_label, ano_score_final)
    pr_final = auc_pr(ano_score_final, ano_label)

    print('AUC:{:.4f}'.format(auc_final))
    print('PR:{:4f}'.format(pr_final))
    print('Precision@K:{:.4f} {:.4f} {:.4f}'.format(precisions[0], precisions[1], precisions[2]))
    print('Recall@K:{:.4f} {:.4f} {:.4f}'.format(recalls[0], recalls[1], recalls[2]))

