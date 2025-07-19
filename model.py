import datetime
import os.path
import torch.nn.functional as F
import dgl, math, torch
import numpy as np
import torch.nn as nn
import dgl.function as fn
from dgl.nn import GATConv, GraphConv, SAGEConv
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.model_selection import train_test_split
from utils import calculate_regularization_loss,print_execution_time


class EarlyStopping:
    """Early stops the training if loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, modelfile='model.pt'):
        """
        Args:
            patience (int): How long to wait after last time loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf
        self.model_file = modelfile

    def __call__(self, loss, model):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_model(self.model_file)
        else:
            self.best_score = score
            self.save_checkpoint(loss, model)
            self.counter = 0

    def save_checkpoint(self, loss, model):
        '''Saves model when loss decrease.'''
        if self.verbose:
            print(f'Loss decreased ({self.loss_min:.6f} --> {loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.model_file)
        self.loss_min = loss


class GPASS(nn.Module):
    def __init__(self, args, GNNmodel='GAT'):
        super(GPASS, self).__init__()
        self.n_hid = args.n_hidden
        self.edge_emb_layer = args.edge_emb_layer
        self.gnn_layer = args.gnn_layer
        self.ncf_hidden = args.ncf_hidden
        self.m_numb = args.m_numb
        self.d_numb = args.d_numb
        self.cm_numb = args.cm_numb
        self.cd_numb = args.cd_numb
        self.num_nodes = args.num_nodes
        self.num_rels = args.num_rels
        self.device = args.device
        self.Dynamic_train = args.Dynamic_train
        self.Use_GNN_Embedding = args.Use_GNN_Embedding
        if hasattr(args, 'GNNkwargs'):
            self.GNNkwargs = args.GNNkwargs
        else:
            self.GNNkwargs = {'gnn_layer': args.gnn_layer}
        self.args = args

        self.best_auc = 0.0

        self.emb = nn.Parameter(torch.empty(self.num_nodes, self.n_hid))
        self.norm = nn.LayerNorm((self.edge_emb_layer + 1) * self.n_hid)

        model_mapping = {
            'GAT': GAT_Model,
            'GCN': GCN_Model,
            'GIN': GIN_Model,
            'GraphSAGE': GraphSAGE_Model,
            'GNN': GNN_Model,
        }

        if isinstance(GNNmodel, str):
            if GNNmodel in model_mapping:
                GNNmodel = model_mapping[GNNmodel]
            else:
                raise ValueError(f"Unknown GNN model: {GNNmodel}")

        if isinstance(GNNmodel, type):
            x1_GNNmodel = GNNmodel(self.n_hid, self.n_hid, **self.GNNkwargs)
            x2_GNNmodel = GNNmodel(self.n_hid, self.n_hid, **self.GNNkwargs)
            edge_GNNmodel = GNNmodel(self.n_hid, self.n_hid, **self.GNNkwargs)

        else:
            raise TypeError("GNNmodel should be either a string or a class")

        self.layers = nn.ModuleList()
        for i in range(0, self.edge_emb_layer):
            self.layers.append(EdgeGNNLayer(self.n_hid, self.n_hid,
                                         layer_norm=True, dropout=args.dropout,
                                         activation=nn.LeakyReLU(0.2, inplace=True),
                                         GNN_embedding_Model=edge_GNNmodel
                                         ))
        self.decoder = NCFDecoder(self.ncf_hidden)

        self.emb_dim = self.n_hid + self.edge_emb_layer * self.n_hid

        self.x1_percep_model = Perception_Model(self.n_hid, self.n_hid, Dynamic_train=self.Dynamic_train, Use_GNN_Embedding=self.Use_GNN_Embedding, GNN_embedding_Model=x1_GNNmodel)
        self.x2_percep_model = Perception_Model(self.n_hid, self.n_hid, Dynamic_train=self.Dynamic_train, Use_GNN_Embedding=self.Use_GNN_Embedding, GNN_embedding_Model=x2_GNNmodel)
        self.model_train = True
        self.init_parameters = False
        self.to(device=args.device)

    def adaptive_mloss(self, y_pred, y_true, method=None, lamb_reg=1e-4):
        positive_preds = y_pred[y_true == 1]
        pos_count = positive_preds.size(0)

        negative_indices = torch.where(y_true == 0)[0]
        neg_count = len(negative_indices)

        if neg_count >= pos_count:
            selected_neg_indices = torch.randperm(neg_count)[:pos_count]
        else:
            selected_neg_indices = torch.randint(0, neg_count, (pos_count,))

        balanced_negative_preds = y_pred[negative_indices[selected_neg_indices]]

        bpr_loss = -torch.log(torch.sigmoid(positive_preds -balanced_negative_preds)).mean()
        reg_loss = calculate_regularization_loss(self)
        loss_bce = F.binary_cross_entropy_with_logits(y_pred, y_true)
        loss_components = [bpr_loss, reg_loss, loss_bce]
        if not hasattr(self, 'adaptive_loss'):
            self._adaptive_loss = AdaptiveWeightLoss(num_tasks=len(loss_components))
        total_loss = self._adaptive_loss(loss_components)
        return total_loss

    def reset_parameters(self):
        self.init_parameters = True
        nn.init.normal_(self.emb)

        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight' in name and param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
        for name, param in self.decoder.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)


    def predict(self, user, item):
        # y = torch.einsum('bi,ij,bj ->b', user, self.W, item) / user.shape[1]
        # print(y.shape)
        tensor = self.decoder(user, item)
        y =tensor.masked_fill(torch.isnan(tensor), 0.5)
        return y

    def embeding_forward(self, x, graph):
        all_emb = [x]
        for idx, layer in enumerate(self.layers):
            x = layer(graph, x)
            all_emb += [x]
        x = torch.cat(all_emb, dim=1)
        # x = self.norm(x)
        self.save_data = x
        return x

    def forward(self, graph, g1, g2):
        x = self.emb
        graph = graph.clone()
        x1 = x[:self.m_numb, :]
        x2 = x[self.m_numb:self.m_numb + self.d_numb, :]
        x1_new = self.emb[-2, :].reshape((1, -1))
        x2_new = self.emb[-1, :].reshape((1, -1))
        x1_cat = torch.cat([x1, x1_new], dim=0)
        x2_cat = torch.cat([x2, x2_new], dim=0)
        if self.Dynamic_train == True:
            num_samples_x1 = min(10, int(x1.size(0) * 0.01)+1)
            num_samples_x2 = min(10, int(x1.size(0) * 0.01)+1)
            x1_indices = torch.arange(x1_cat.size(0)).to(self.device)
            x2_indices = torch.arange(x2_cat.size(0)).to(self.device)

            num_samples_x1 = min(num_samples_x1, x1.size(0))
            num_samples_x2 = min(num_samples_x2, x2.size(0))

            indices_x1 = torch.randperm(x1.size(0))[:num_samples_x1].to(self.device)
            mask1 = ~torch.isin(x1_indices, indices_x1).to(self.device)
            x1_filtered = x1_indices[mask1].to(self.device)
            indices_x2 = torch.randperm(x2.size(0))[:num_samples_x2].to(self.device)
            mask2 = ~torch.isin(x2_indices, indices_x2).to(self.device)
            x2_filtered = x2_indices[mask2].to(self.device)

            x1 = self.x1_percep_model(x1_cat, g1, x1_filtered, indices_x1)
            x2 = self.x2_percep_model(x2_cat, g2, x2_filtered, indices_x2)

            nodes_to_remove = torch.cat([indices_x1, self.m_numb + indices_x2])
            edge_ids_to_remove = []

            max_numb = self.m_numb + self.d_numb

            edges_to_remove = []

            for node in nodes_to_remove:
                succ = graph.successors(node)
                prede = graph.predecessors(node)
                for nei in succ:
                    if nei < max_numb:
                        edge_ids_to_remove.append(graph.edge_ids(node, nei))
                        edges_to_remove.append((node.item(), nei.item()))
                for nei in prede:
                    if nei < max_numb:
                        edge_ids_to_remove.append(graph.edge_ids(nei, node))
                        edges_to_remove.append((node.item(), nei.item()))


            if edge_ids_to_remove:
                edges_to_remove_tensor = torch.tensor(edges_to_remove, device=self.device).T  # 转置为 (2, N)
                edge_ids_to_remove = torch.cat(edge_ids_to_remove)
                graph = dgl.remove_edges(graph, edge_ids_to_remove)

                self.removed_edge = edges_to_remove_tensor

        else:
            x1 = self.x1_percep_model(x1_cat, g1)
            x2 = self.x2_percep_model(x2_cat, g2)

        self.x_rna = x1
        self.x_disease = x2
        x = torch.cat([x1[:-1, ], x2[:-1, ], x[self.m_numb + self.d_numb:-2, :], x1[-1, :].reshape((1, -1)),
                       x2[-1, :].reshape((1, -1))], dim=0)

        x = self.embeding_forward(x, graph)[:-2,]
        self.rep = x
        return x

    def predict_new_node(self, graph, g1, g2, newg1_idx=False, newg2_idx=False):
        x = self.emb
        x1 = x[:self.m_numb, :]
        x2 = x[self.m_numb:self.m_numb + self.d_numb, :]
        x1_new = self.emb[-2, :].reshape((1, -1))
        x2_new = self.emb[-1, :].reshape((1, -1))
        x1_filtered = torch.arange(self.m_numb).to(self.device)
        x2_filtered = torch.arange(self.d_numb).to(self.device)
        x1_cat = torch.cat([x1, x1_new], dim=0)
        x2_cat = torch.cat([x2, x2_new], dim=0)
        self.x1_percep_model.Moldel_Predict=True
        self.x2_percep_model.Moldel_Predict=True
        if self.Dynamic_train == True:
            if newg1_idx == False and newg2_idx == False:
                x1 = self.x1_percep_model(x1_cat, g1)
                x2 = self.x2_percep_model(x2_cat, g2)

            elif newg1_idx == False:
                x1 = self.x1_percep_model(x1_cat, g1)
                indices_x2 = torch.tensor([x2_cat.shape[0] - 1]).to(self.device)
                x2 = self.x2_percep_model(x2_cat, g2, x2_filtered, indices_x2)

            elif newg2_idx == False:
                indices_x1 = torch.tensor([x1_cat.shape[0] - 1]).to(self.device)
                x1 = self.x1_percep_model(x1_cat, g1, x1_filtered, indices_x1)
                x2 = self.x2_percep_model(x2_cat, g2)

            else:
                indices_x1 = torch.tensor([x1_cat.shape[0] - 1]).to(self.device)
                indices_x2 = torch.tensor([x2_cat.shape[0] - 1]).to(self.device)
                x1 = self.x1_percep_model(x1_cat, g1, x1_filtered, indices_x1)
                x2 = self.x2_percep_model(x2_cat, g2, x2_filtered, indices_x2)


        else:
            x1 = self.x1_percep_model(x1_cat, g1)
            x2 = self.x2_percep_model(x2_cat, g2)

        x = torch.cat([x1[:-1, ], x2[:-1, ], x[self.m_numb + self.d_numb:-2, :], x1[-1, :].reshape((1, -1)),
                       x2[-1, :].reshape((1, -1))], dim=0)
        x = self.embeding_forward(x, graph)
        self.x1_percep_model.Moldel_Predict=False
        self.x2_percep_model.Moldel_Predict=False
        return x

    def save_model_state(self, kf, train_idx, test_idx, y, src_train, tgt_train, src_test, tgt_test, graph, x_encode):
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.y = y
        self.concat_same_m_d(kf, src_train, tgt_train, src_test, tgt_test, graph, x_encode)

    def concat_same_m_d(self, kf, src_train, tgt_train, src_test, tgt_test, graph, x_encode):
        # train_data_concat = torch.cat((self.save_data[src_train],self.save_data[tgt_train]),dim=1)

        # test_data_concat = torch.cat((self.save_data[src_test],self.save_data[tgt_test]),dim=1)
        # print(test_data_concat.shape)
        test_data_concat = torch.cat((self(graph, x_encode)[src_test], self(graph, x_encode)[tgt_test]), dim=1)
        train_data_concat = torch.cat((self(graph, x_encode)[src_train], self(graph, x_encode)[tgt_train]), dim=1)
        # y_train = self.y['y'][self.train_idx].cpu().numpy()
        # y_test = self.y['y'][self.test_idx].cpu().numpy()

        joblib.dump({'train_data': train_data_concat,
                     'test_data': test_data_concat,
                     'y_train': self.y['y'][self.train_idx].cpu().numpy(),
                     'y_test': self.y['y'][self.test_idx].cpu().numpy(),
                     },
                    './mid_data/' + 'nl' + str(kf) + '-heat-kf_best_cat_data.dict')
        # return train_data_concat,test_data_concat,y_train,y_test

    def train_model(self, biodata):
        """Function to train the model based on provided indices and parameters."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        y = biodata.y
        graph = biodata.graph.clone()
        gmss = biodata.mss_graph
        gdss = biodata.dss_graph
        train_idx, test_idx = train_test_split(np.arange(y['y'].shape[0]), test_size=self.args.test_size)

        src_train, tgt_train = y['y_edge'][0][train_idx], y['y_edge'][1][train_idx]
        src_test, tgt_test = y['y_edge'][0][test_idx], y['y_edge'][1][test_idx]

        # Edge removal for test edges
        true_src = y['y_edge'][0][test_idx[(y['y'][test_idx] == 1).reshape(-1)]]
        true_tgt = y['y_edge'][1][test_idx[(y['y'][test_idx] == 1).reshape(-1)]]
        for _src, _tgt in zip(true_src, true_tgt):
            graph.remove_edges((_src, _tgt))

        early_stopping = EarlyStopping(patience=self.args.patience, modelfile=self.args.save_model_file)

        start_time = datetime.datetime.now()
        for epoch in range(1, self.args.epochs + 1):
            self.train()
            optimizer.zero_grad()
            rep = self.forward(graph, gmss, gdss)
            preds = self.predict(rep[src_train], rep[tgt_train])
            loss = self.adaptive_mloss(preds, y['y'][train_idx].reshape(-1, ).to(self.device), self.args.Loss)
            loss.backward()
            optimizer.step()
            if epoch % self.args.print_epoch == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
                self.eval()
                with torch.no_grad():
                    rep = self.forward(graph, gmss, gdss)
                    preds = self.predict(rep[src_test], rep[tgt_test])
                    loss = self.mloss(preds, y['y'][test_idx].reshape(-1, ).to(self.device), self.args.Loss)
                    out_pred = preds.to('cpu').detach().numpy()
                    y_true = y['y'][test_idx].to('cpu').detach().numpy()
                    auc = roc_auc_score(y_true, out_pred)
                    print('AUC:', auc)
                    print_execution_time(start_time, epoch)
                    early_stopping(loss, self)
                    if self.args.early_stop == True and early_stopping.early_stop:
                        print('EarlyStopping: run {} iteration'.format(epoch + 1))
                        break

    def Overall_Refactoring_ASS_Embedding(self, biodata):
        self.eval()
        graph = biodata.graph.clone()
        gmss = biodata.mss_graph
        gdss = biodata.dss_graph
        with torch.no_grad():
            rep = self.forward(graph, gmss, gdss)
        miRNA_rep = rep[:self.m_numb]
        disease_rep = rep[self.m_numb:]
        self.AssScores = np.zeros((self.m_numb, self.d_numb))
        batch_size = 1000
        for i in range(0, self.m_numb, batch_size):
            end_i = min(i + batch_size, self.m_numb)
            miRNA_batch = miRNA_rep[i:end_i]
            for j in range(self.d_numb):
                disease_j = disease_rep[j:j + 1].expand(end_i - i, -1)
                scores = self.predict(miRNA_batch, disease_j)
                self.AssScores[i:end_i, j] = scores.cpu().detach().numpy()
        biodata.ASS_Embedding['EdgeScores'] = self.AssScores
        biodata.ASS_Embedding['miRNA_embeding'] = miRNA_rep.cpu().detach().numpy()
        biodata.ASS_Embedding['Disease_embeding'] = disease_rep.cpu().detach().numpy()
        return self.AssScores


    def Overall_Fill(self, biodata):
        self.eval()
        y_pred_all = []
        y_true_all = []
        with torch.no_grad():
            pred_list = []
            if self.args.new_sp == 'RNA':
                for ss, s_idx in biodata.get_new_mss():
                    biodata.add_one_new_mss(ss)
                    graph = biodata.graph.clone()
                    gmss = biodata.new_mss_graph
                    gdss = biodata.dss_graph
                    src_test = [-2 for _ in range(biodata.d_numb)]
                    tgt_test = [biodata.m_numb + i for i in range(biodata.d_numb)]
                    rep = self.predict_new_node(graph, gmss, gdss, newg1_idx=True)
                    preds = self.predict(rep[src_test], rep[tgt_test]).to('cpu').detach().numpy().reshape(-1, )
                    biodata.clear_newss()
                    pred_list.append(preds)

            if self.args.new_sp == 'disease':
                for ss, s_idx in biodata.get_new_dss():
                    biodata.add_one_new_dss(ss)
                    graph = biodata.graph.clone()
                    gmss = biodata.mss_graph
                    gdss = biodata.new_dss_graph
                    src_test = [-1 for _ in range(biodata.m_numb)]
                    tgt_test = [0 + i for i in range(biodata.m_numb)]
                    rep = self.predict_new_node(graph, gmss, gdss, newg2_idx=True)
                    preds = self.predict(rep[src_test], rep[tgt_test]).to('cpu').detach().numpy().reshape(-1, )
                    biodata.clear_newss()
                    pred_list.append(preds)

            if self.args.new_sp == 'RNA-disease':
                for mss, midx in biodata.get_new_mss():
                    y_pred_list = []
                    for dss, didx in biodata.get_new_dss():
                        biodata.add_one_new_mss(mss)
                        biodata.add_one_new_dss(dss)

                        graph = biodata.graph.clone()
                        gmss = biodata.new_mss_graph
                        gdss = biodata.new_dss_graph
                        src_test = [-2]
                        tgt_test = [-1]
                        rep = self.predict_new_node(graph, gmss, gdss, newg1_idx=True, newg2_idx=True)
                        preds = self.predict(rep[src_test], rep[tgt_test]).to('cpu').detach().numpy().reshape(-1, )
                        y_pred_list.append(preds[0])
                        biodata.clear_newss()

                    pred_list.append(np.array(y_pred_list))
            self._AssFill = np.stack(pred_list)
            return self._AssFill

    def predict_score(self,biodata, midx, didx):
        self.eval()
        graph = biodata.graph.clone()
        gmss = biodata.mss_graph
        gdss = biodata.dss_graphs
        rep = self.forward(graph, gmss, gdss)
        return self.predict(rep[midx], rep[self.m_numb + didx]).to('cpu').detach().numpy()

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class EdgeGNNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 dropout=0.0,
                 layer_norm=False,
                 GNN_embedding_Model=None):
        super(EdgeGNNLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        if GNN_embedding_Model == None:
            GNN_embedding_Model = GAT_Model(in_feats, out_feats)
        self.GNN_embedding_Model = GNN_embedding_Model

        if self.bias:
            self.h_bias = nn.Parameter(torch.empty(out_feats))
            nn.init.zeros_(self.h_bias)

        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feats)

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, feat):
        with g.local_scope():
            g = g.clone()
            g.add_edges(g.nodes(), g.nodes())
            g.ndata['h'] = self.GNN_embedding_Model(feat ,g)
            node_rep = g.ndata['h']
            if self.layer_norm:
                node_rep = self.layer_norm_weight(node_rep)
            if self.bias:
                node_rep = node_rep + self.h_bias
            if self.activation:
                node_rep = self.activation(node_rep)
            node_rep = self.dropout(node_rep)
            return node_rep


class Perception_Model(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 Dynamic_train=True,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 dropout=0.0,
                 layer_norm=False,
                 Use_GNN_Embedding=False,
                 GNN_embedding_Model=None):

        super(Perception_Model, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        if GNN_embedding_Model == None:
            GNN_embedding_Model = GAT_Model(in_feats, out_feats)
        self.GNN_embedding_Model = GNN_embedding_Model

        if self.bias:
            self.h_bias = nn.Parameter(torch.empty(out_feats))
            nn.init.zeros_(self.h_bias)

        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feats)

        self.dropout = nn.Dropout(dropout)
        self.Dynamic_train = Dynamic_train
        self.Moldel_Predict = False
        self.Use_GNN_Embedding = Use_GNN_Embedding
        self.w = nn.Parameter(torch.tensor(-100.0))
        self.linear = nn.Linear(in_feats,in_feats)

    def forward(self,
                feat,
                g,
                x_input_indices=None,
                x_none_indices=None):
        feat = self.linear(feat)
        with g.local_scope():
            g = g.clone()
            g.ndata['h'] = feat
            if (self.Moldel_Predict == True or self.Dynamic_train == True) and x_input_indices != None and x_none_indices != None:
                known_feat = feat[x_input_indices]
                n_known = len(x_input_indices)
                n_unknown = len(x_none_indices)
                if n_unknown == 0:
                    full_feat = known_feat
                    g.ndata['h'] = full_feat
                else:
                    in_edges = g.in_edges(x_none_indices, form='all')  # (u, v, eid)
                    mask = torch.isin(in_edges[0], x_input_indices)
                    u = in_edges[0][mask]
                    v = in_edges[1][mask]
                    eid = in_edges[2][mask]

                    if len(u) == 0:
                        unknown_feat = torch.zeros((n_unknown, feat.shape[1]), device=feat.device)
                    else:
                        weights = g.edata['weight'][eid]
                        known_map = torch.full((g.number_of_nodes(),), -1, dtype=torch.long, device=g.device)
                        known_map[x_input_indices] = torch.arange(n_known, device=g.device)

                        unknown_map = torch.full((g.number_of_nodes(),), -1, dtype=torch.long, device=g.device)
                        unknown_map[x_none_indices] = torch.arange(n_unknown, device=g.device)

                        src_ids = known_map[u]
                        dst_ids = unknown_map[v]

                        g_bipartite = dgl.graph(
                            (src_ids, dst_ids + n_known),
                            num_nodes=n_known + n_unknown,
                            device=g.device
                        )

                        g_bipartite.ndata['feat'] = torch.zeros(
                            (n_known + n_unknown, feat.shape[1]),
                            device=feat.device
                        )
                        g_bipartite.ndata['feat'][:n_known] = feat[x_input_indices]  # 设置已知特征

                        g_bipartite.edata['weight'] = weights.unsqueeze(1)

                        g_bipartite.update_all(
                            fn.u_mul_e('feat', 'weight', 'm'),
                            fn.sum('m', 'feat_sum')
                        )
                        g_bipartite.update_all(
                            fn.copy_e('weight', 'w'),
                            fn.sum('w', 'weight_sum')
                        )
                        unknown_nodes = torch.arange(n_known, n_known + n_unknown, device=g.device)
                        feat_sum = g_bipartite.ndata['feat_sum'][unknown_nodes]
                        weight_sum = g_bipartite.ndata['weight_sum'][unknown_nodes].clamp(min=1e-6)
                        unknown_feat = feat_sum / weight_sum
                    full_feat = torch.cat([known_feat, unknown_feat], dim=0)
                    g.ndata['h'] = full_feat
            else:
                g.ndata['h'] = feat
                full_feat = feat
            g.add_edges(g.nodes(), g.nodes())
            if self.Use_GNN_Embedding == False:
                return full_feat

            g.ndata['h'] = self.GNN_embedding_Model(feat, g)
            node_rep = g.ndata['h']
            if self.activation:
                node_rep = self.activation(node_rep)
            node_rep = self.dropout(node_rep)
            return F.sigmoid(self.w) * node_rep + (1 - F.sigmoid(self.w)) * full_feat


class GNN_Base(nn.Module):
    def __init__(self, in_feats, out_feats):
        """
        Base class for Graph Neural Network (GNN).

        Parameters:
        -----------
        in_feats : int
            Input feature size for each node.
        out_feats : int
            Output feature size for each node.
        """
        super(GNN_Base, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats

    def forward(self, feat, g):
        """
        Forward pass for the GNN model (to be overridden by subclasses).

        Parameters:
        -----------
        feat : torch.Tensor
            Node features.
        g : dgl.DGLGraph
            The graph to perform message passing on.

        Returns:
        --------
        torch.Tensor
            Output node features.
        """
        with g.local_scope():
            pass


class GAT_Model(GNN_Base):
    def __init__(self, in_feats, out_feats, gnn_layer=4, num_heads=4):
        """
        Graph Attention Network (GAT) model implementation.

        Parameters:
        -----------
        in_feats : int
            Input feature size for each node.
        out_feats : int
            Output feature size for each node.
        gat_layers : int, optional, default=4
            Number of GAT layers to stack.
        num_heads : int, optional, default=4
            Number of attention heads in each GAT layer.
        """
        super(GAT_Model, self).__init__(in_feats, out_feats)
        self.gat_layers = gnn_layer
        self.num_heads = num_heads

        # List of GAT layers
        self.attentions = nn.ModuleList()
        for _ in range(self.gat_layers):
            self.attentions.append(GATConv(in_feats, out_feats,
                                           num_heads=num_heads,
                                           allow_zero_in_degree=True  # 允许入度为0的节点
                                            ))

    def forward(self, feat, g):
        """
        Forward pass for the GAT model.

        Parameters:
        -----------
        feat : torch.Tensor
            Input node features.
        g : dgl.DGLGraph
            The graph structure on which message passing is performed.

        Returns:
        --------
        torch.Tensor
            Node representations after passing through the GAT layers.
        """
        with g.local_scope():
            h = feat
            for i, attention in enumerate(self.attentions):
                h = torch.sum(attention(g, h), dim=1)  # Summing across heads
                g.ndata['h'] = h
        return h



class GCN_Model(GNN_Base):
    def __init__(self, in_feats, out_feats, gnn_layer=4):
        """
        Graph Convolutional Network (GCN) model implementation.
        """
        super(GCN_Model, self).__init__(in_feats, out_feats)
        self.gcn_layers = gnn_layer
        self.convs = nn.ModuleList()
        for _ in range(self.gcn_layers):
            self.convs.append(GraphConv(in_feats, out_feats))

    def forward(self, feat, g):
        with g.local_scope():
            h = feat
            for i, conv in enumerate(self.convs):
                h = conv(g, h)
                g.ndata['h'] = h
        return h


class GIN_Model(GNN_Base):
    def __init__(self, in_feats, out_feats, gnn_layer=4):
        """
        Graph Isomorphism Network (GIN) model implementation.
        """
        super(GIN_Model, self).__init__(in_feats, out_feats)
        self.gin_layers = gnn_layer
        self.convs = nn.ModuleList()
        for _ in range(self.gin_layers):
            self.convs.append(GraphConv(in_feats, out_feats))

    def forward(self, feat, g):
        with g.local_scope():
            h = feat
            for i, conv in enumerate(self.convs):
                h = conv(g, h)
                g.ndata['h'] = h
        return h


class GraphSAGE_Model(GNN_Base):
    def __init__(self, in_feats, out_feats, gnn_layer=4):
        """
        Graph SAGE (Sample and Aggregation) model implementation.
        """
        super(GraphSAGE_Model, self).__init__(in_feats, out_feats)
        self.sage_layers = gnn_layer
        self.convs = nn.ModuleList()
        for _ in range(self.sage_layers):
            self.convs.append(SAGEConv(in_feats, out_feats, aggregator_type='mean'))

    def forward(self, feat, g):
        with g.local_scope():
            h = feat
            for i, conv in enumerate(self.convs):
                h = conv(g, h)
                g.ndata['h'] = h
        return h


class GNN_Model(GNN_Base):
    def __init__(self, in_feats, out_feats, gnn_layer=4):
        """
        Generic Graph Neural Network (GNN) model implementation.
        """
        super(GNN_Model, self).__init__(in_feats, out_feats)
        self.gnn_layer = gnn_layer
        self.convs = nn.ModuleList()
        for _ in range(self.gnn_layer):
            self.convs.append(GraphConv(in_feats, out_feats))

    def forward(self, feat, g):
        with g.local_scope():
            h = feat
            for i, conv in enumerate(self.convs):
                h = conv(g, h)
                g.ndata['h'] = h
        return h


class NCFDecoder(nn.Module):
    def __init__(self, NCF_hidden):
        super(NCFDecoder, self).__init__()
        assert NCF_hidden in [1024, 512, 256, 128, 64, 32, 16]
        mlp_layers = [NCF_hidden, int(NCF_hidden/2), int(NCF_hidden/4)]
        self.gmf_layer = nn.LazyLinear(1)

        self.mlp_layers = nn.ModuleList()
        self.mlp_layers.append(nn.LazyLinear(mlp_layers[0]))
        self.mlp_layers.append(nn.ReLU())
        input_dim = mlp_layers[0]
        for layer_dim in mlp_layers[1:]:
            self.mlp_layers.append(nn.Linear(input_dim, layer_dim))
            self.mlp_layers.append(nn.ReLU())
            input_dim = layer_dim

        self.mlp_output = nn.Linear(input_dim, 1)

        self.final_layer = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        gmf_output = self.gmf_layer(user * item)

        mlp_input = torch.cat([user, item], dim=1)
        for layer in self.mlp_layers:
            mlp_input = layer(mlp_input)
        mlp_output = self.mlp_output(mlp_input)

        combined = torch.cat([gmf_output, mlp_output], dim=1)
        output = self.final_layer(combined)
        return self.sigmoid(output).squeeze(-1)


class AdaptiveWeightLoss(nn.Module):
    def __init__(self, num_tasks=3, initial_weights=None, decay_rate=0.95, epsilon=1e-8,
                 clip_value=10.0, normalize_grad=True, use_gradient_clipping=True):
        super(AdaptiveWeightLoss, self).__init__()

        if initial_weights is None:
            self.weights = nn.Parameter(torch.ones(num_tasks) * 0.1)
        else:
            self.weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))

        self.register_buffer('grad_history', torch.zeros(num_tasks))
        self.decay_rate = decay_rate
        self.epsilon = epsilon

        self.register_buffer('loss_history', torch.zeros(num_tasks, 10))  # 保存最近10轮的损失
        self.register_buffer('loss_mean', torch.zeros(num_tasks))
        self.register_buffer('loss_std', torch.zeros(num_tasks))
        self.loss_idx = 0

        self.clip_value = clip_value
        self.normalize_grad = normalize_grad
        self.use_gradient_clipping = use_gradient_clipping

    def forward(self, loss_components):
        """
        Args:
            loss_components:
        Returns:
        """
        # 确保损失值有效
        for i, loss in enumerate(loss_components):
            if torch.isnan(loss) or torch.isinf(loss):
                loss_components[i] = torch.tensor(1e6, device=loss.device, requires_grad=False)

        normalized_weights = F.softmax(self.weights, dim=0)

        weighted_losses = [w * loss for w, loss in zip(normalized_weights, loss_components)]
        total_loss = sum(weighted_losses)
        with torch.no_grad():
            for i, loss in enumerate(loss_components):
                self.loss_history[i, self.loss_idx % 10] = loss.item()

            self.loss_idx += 1
            self.loss_mean = torch.mean(self.loss_history, dim=1)
            self.loss_std = torch.std(self.loss_history, dim=1) + self.epsilon

            if self.loss_idx >= 10:
                self.adjust_weights(loss_components)

        return total_loss

    def adjust_weights(self, loss_components):
        loss_changes = []
        for i in range(len(loss_components)):
            current_loss = loss_components[i].item()
            change = current_loss / (self.loss_mean[i] + self.epsilon)
            if change > 100.0:
                change = 100.0
            elif change < 0.02:
                change = 0.02

            loss_changes.append(change)
        changes_tensor = torch.tensor(loss_changes, device=self.weights.device)
        normalized_changes = changes_tensor / (torch.sum(changes_tensor) + self.epsilon)
        self.weights.data = self.weights.data - 0.1 * normalized_changes
        if self.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.weights, self.clip_value)
        self.weights.data = torch.clamp(self.weights, min=-100.0, max=100.0)