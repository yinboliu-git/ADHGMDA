import joblib
import numpy as np
import torch, dgl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from utils import *
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Nimbus Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["figure.titleweight"] = "bold"

class DiModel():
    def __init__(self, k, ss, TopN=None):
        """
        Applies KMeans clustering to the given dataset and saves the one-hot encoded cluster IDs.

        Parameters:
        - k (int): The number of clusters for KMeans.
        - df (str): The dataset name.
        """
        print('Kmeans get Di sim of {df}')
        # Load m_gs.csv and m_ss.csv files
        m_gs = ss

        self.scaler = StandardScaler()
        m_gs = self.scaler.fit_transform(m_gs)
        # Cluster m_gs using KMeans and encode cluster IDs as one-hot vectors
        max_k = k
        k_range = range(1, max_k + 1)
        wss = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=100, random_state=42)
            kmeans.fit(m_gs)
            wss.append(kmeans.inertia_)

        diff1 = np.diff(wss)
        diff2 = np.diff(diff1)
        elbow = np.argmax(diff2) + 1
        k = k_range[elbow]
        if TopN is None:
            TopN = 1
        self.kmeans_model = KMeans(n_clusters=k, init='k-means++', n_init=100)
        m_gc = self.kmeans_model.fit_transform(m_gs)
        m_gc_copy = np.copy(m_gc)
        min_two_idx = np.argsort(m_gc_copy, axis=1)[:, :TopN]
        m_gc_copy[:, :] = 0
        for i, idx in enumerate(min_two_idx):
            m_gc_copy[i, idx] = 1
        self.k = k
        self.TopN = TopN
        assert self.TopN <= k
        self.DiC = m_gc_copy

    def get_DiC(self):
        return self.DiC

    def transform(self, ss):
        if len(ss.shape) == 1:
            ss = ss.reshape((1,-1))
        m_gc = self.kmeans_model.transform(ss)
        m_gc_copy = np.copy(m_gc)
        min_two_idx = np.argsort(m_gc_copy, axis=1)[:, :self.TopN]
        m_gc_copy[:, :] = 0
        for i, idx in enumerate(min_two_idx):
            m_gc_copy[i, idx] = 1
        return m_gc_copy


class Config:
    """Configuration class to store model and training parameters."""
    def __init__(self,data_name='data'):
        self.data_name=data_name
        self.datapath = './data/'
        self.save_file = f'./save_file/{data_name}/'
        self.kfold = 5
        self.repeat = 10
        self.self_encode_len = 256
        self.globel_random = 42
        self.search_args = {'arg_name': [], 'arg_value': []}
        self.epochs = 1000 #5000
        self.print_epoch = 10 #20

        self.lr = 0.001
        self.weight_decay = 1e-6
        # self.reg = 0.0002
        # self.decay = 0.985
        # self.decay_step = 1
        self.patience = 30
        self.feature_perception = True
        self.test_size = 0.0 # 独立测试集 正式训练为0
        self.val_test_size = 0.05 # 验证集
        self.save_model_file = './model.pt'
        self.early_stop = True

        self.edge_emb_layer = 2
        self.gnn_layer = 2
        self.n_hidden = 32
        self.ncf_hidden = 128

        self.kmeans_max_K = 10   # [5, 10, 15, 20]
        self.dropout = 0.2

        self.neg_beta = 1
        self.noise_level = 0

        self.Use_GNN_Embedding = True
        self.Dynamic_train = False

        self.mask_sp = None
        self.new_sp = None

        self.save_ASS_Embedding = False
        self.edge_number = None

        self.kmeans_TopN = 1
        self.TopK_ss = 5
        set_seed(self.globel_random)

    def get_data_params(self, biodata):
        self.m_numb = biodata.m_numb
        self.d_numb = biodata.d_numb
        self.cm_numb = biodata.cm_numb
        self.cd_numb = biodata.cd_numb
        self.num_nodes = biodata.num_nodes
        self.num_rels = biodata.num_rels
        self.device = biodata.device


    def set_new_sp(self, new_sp):
        '''
        :param new_sp:
            new_sp in ['RNA', 'disease', 'RNA-disease']
        :return: None
        '''
        assert new_sp in ['RNA', 'disease', 'RNA-disease']
        self.new_sp = new_sp

    def reset_data_name(self, data_name):
        self.save_file = f'./save_file/{data_name}/'
        self.data_name = data_name

    def copy(self):
        return copy.deepcopy(self)


class BioData():
    def __init__(self, md_file=None, mss_list=None, dss_list=None,  m_encoding_list=[None,],
                 d_encoding_list=[None, ], x_encoding=False,  device='cpu', data_name='data',
                 config=Config(), sep=','):
            """
                Initialize the graph structure and data for the model based on input files and configurations.

                Parameters:
                -----------
                md_file : str
                    Path to the miRNA-disease association matrix file (CSV format). This file represents the relationships between miRNAs and diseases.

                mss_list : list
                    List of paths to miRNA similarity matrices (CSV format). These matrices describe similarity relationships between miRNAs.

                dss_list : list
                    List of paths to disease similarity matrices (CSV format). These matrices describe similarity relationships between diseases.

                kmeans_max_K : int, optional, default=20
                    Number of clusters to use for KMeans clustering for miRNA and disease similarity.

                kmeans_TopN : int, optional, default=1
                    Number of top clusters to use after applying KMeans clustering for similarity.

                TopK_ss : int, optional, default=20
                    Number of top similarity scores to retain for each node in the similarity matrices.

                m_encoding_list : list, optional, default=[None]
                    List of encoding strategies to apply to the miRNA similarity data.

                d_encoding_list : list, optional, default=[None]
                    List of encoding strategies to apply to the disease similarity data.

                x_encoding : bool, optional, default=False
                    Whether to apply specific encoding to input features.

                device : str, optional, default='cpu'
                    The device (CPU/GPU) where the computations will be performed.

                Attributes:
                -----------
                TopK_ss : int
                    Number of top similarity scores to retain for each node in the similarity matrices.

                kmeans_max_K : int
                    Number of clusters for KMeans clustering.

                kmeans_TopN : int
                    Number of top clusters to keep after KMeans clustering.

                num_rels : int
                    Keeps track of the number of different relation types (edges) in the heterogeneous graph.

                device : str
                    The computing device used ('cpu' or 'gpu').

                metapath_name : dict
                    Dictionary storing the names of the different meta-paths (relation types) in the graph.

                adj_matrix : np.ndarray
                    The adjacency matrix (relationship matrix) for miRNA-disease associations, loaded from `md_file`.

                mss : np.ndarray
                    The averaged miRNA similarity matrix, concatenated from the input similarity files in `mss_list`.

                dss : np.ndarray
                    The averaged disease similarity matrix, concatenated from the input similarity files in `dss_list`.

                mss_Di : DiModel
                    KMeans-based clustering model applied to miRNA similarity data.

                dss_Di : DiModel
                    KMeans-based clustering model applied to disease similarity data.

                adj_matrix_torch : torch.Tensor
                    PyTorch tensor representing the miRNA-disease association matrix.

                mss_graph : np.ndarray
                    Graph edges representing the miRNA similarity structure.

                dss_graph : np.ndarray
                    Graph edges representing the disease similarity structure.

                src : list
                    List of source nodes for graph edges.

                dst : list
                    List of destination nodes for graph edges.

                etype : list
                    List of edge types corresponding to different meta-paths (relation types).

                num_nodes : int
                    Total number of nodes in the heterogeneous graph, including virtual nodes.

                node : dict
                    Dictionary containing information about the nodes (miRNAs, diseases, and their clusters).

                graph : dgl.DGLGraph
                    The DGL graph object representing the heterogeneous network of miRNAs, diseases, and their clusters.

                x_encode : np.ndarray
                    Encoded feature data for miRNAs and diseases if encoding is applied.
            """
            config.reset_data_name(data_name)
            self.config = config
            if md_file is None:
                self.ASS_Embedding = {

                }
            else:
                self.ASS_Embedding = {
                }

                self.num_rels = 0
                self.device = device
                self.metapath_name = {}
                self.TopK_ss = self.config.TopK_ss
                self.kmeans_max_K = self.config.kmeans_max_K
                self.kmeans_TopN = self.config.kmeans_TopN

                try:
                    self.adj_matrix = pd.read_csv(md_file, header=None, index_col=None, sep=sep).fillna(0).values
                except:
                    self.adj_matrix = pd.DataFrame(md_file).fillna(0).values
                mss_matrix_list = []
                dss_matrix_list = []
                if mss_list is None:
                    mss_matrix_list.append(randomMinMax(self.adj_matrix.shape[0]))
                else:
                    try:
                        for mss_f in mss_list:
                            mss_matrix_list.append(pd.read_csv(mss_f, header=None, index_col=None, sep=sep).fillna(0).values[np.newaxis, ...])
                    except:
                        for mss_f in mss_list:
                            mss_matrix_list.append(pd.DataFrame(mss_f).fillna(0).values[np.newaxis, ...])

                if dss_list is None:
                    mss_matrix_list.append(randomMinMax(self.adj_matrix.shape[1]))
                else:
                    try:
                        for dss_f in dss_list:
                            dss_matrix_list.append(pd.read_csv(dss_f, header=None, index_col=None, sep=sep).fillna(0).values[np.newaxis, ...])
                    except:
                        for dss_f in dss_list:
                            dss_matrix_list.append(pd.DataFrame(dss_f).fillna(0).values[np.newaxis, ...])

                self.mss_matrix_list = [ss.mean(0) for ss in mss_matrix_list]
                self.dss_matrix_list = [ss.mean(0) for ss in dss_matrix_list]
                self.mss = np.concatenate(mss_matrix_list, axis=0).mean(0) if len(mss_matrix_list) >= 2 else \
                                                                            mss_matrix_list[0][0]
                self.dss = np.concatenate(dss_matrix_list, axis=0).mean(0) if len(dss_matrix_list) >= 2 else \
                                                                                dss_matrix_list[0][0]

                if self.mss.shape[0] != self.mss.shape[-1] and self.mss.shape[1] == self.adj_matrix.shape[0]:
                    self.mss = self.mss.T

                if self.dss.shape[0] != self.dss.shape[-1] and self.dss.shape[1] == self.adj_matrix.shape[1]:
                    self.dss = self.dss.T

                assert self.mss.shape[0] == self.adj_matrix.shape[0]
                assert  self.dss.shape[0] == self.adj_matrix.shape[1]

                if self.mss.shape[0] != self.mss.shape[-1]:
                    self.mss = cosine_similarity(self.mss)

                if self.dss.shape[0] != self.dss.shape[-1]:
                    self.dss = cosine_similarity(self.dss)

                self.mss_Di = DiModel(self.kmeans_max_K, self.mss, self.kmeans_TopN)
                self.dss_Di = DiModel(self.kmeans_max_K, self.dss, self.kmeans_TopN)
                adj_matrix_mc = self.mss_Di.DiC
                adj_matrix_dc = self.dss_Di.DiC

                self.mss = np.concatenate([self.mss, np.ones((1, self.mss.shape[-1]))], axis=0)
                self.dss = np.concatenate([self.dss, np.ones((1, self.dss.shape[-1]))], axis=0)
                self.mss_graph = self.get_ss_edge(self.mss)
                self.dss_graph = self.get_ss_edge(self.dss)

                self.ASS_Embedding['edge_matrix'] = self.adj_matrix
                self.ASS_Embedding['miRNA_matrix'] = self.mss
                self.ASS_Embedding['Disease_matrix'] = self.dss

                adj_matrix = torch.tensor(self.adj_matrix)
                self.adj_matrix_torch = adj_matrix.clone().to(device)
                self.adj_matrix_mc = adj_matrix_mc = torch.tensor(adj_matrix_mc)
                self.adj_matrix_dc = adj_matrix_dc = torch.tensor(adj_matrix_dc)
                bm_numb = self.m_numb = adj_matrix.shape[0]
                bd_numb = self.d_numb = adj_matrix.shape[1]
                bcm_numb = self.cm_numb = adj_matrix_mc.shape[1]
                bcd_numb = self.cd_numb = adj_matrix_dc.shape[1]

                src, dst, etype = [], [], []
                self.node = {}

                """ m_d.csv """
                bm = 0
                bd = bm + bm_numb
                temp = adj_matrix.nonzero()
                mids, dids = temp[:, 0], temp[:, 1]
                src += (bm + mids).tolist()
                dst += (bd + dids).tolist()
                etype += [self.num_rels] * (mids.shape[0])
                self.node[0] = ['miRNA', 0, self.m_numb, self.m_numb]
                self.node[1] = ['disease', self.node[0][-1], self.node[0][-1] + self.d_numb, self.d_numb]

                self.metapath_name[self.num_rels] = ('miRNA', self.num_rels, 'disease')
                self.num_rels += 1

                src += (bd + dids).tolist()
                dst += (bm + mids).tolist()
                etype += [self.num_rels] * (mids.shape[0])
                self.num_nodes = bd + bd_numb

                self.metapath_name[self.num_rels] = ('disease', self.num_rels, 'miRNA')

                self.num_rels += 1

                """ m_c.csv """
                self.bcm = bcm = bd + bd_numb
                temp = adj_matrix_mc.nonzero()
                mids, cmids = temp[:, 0], temp[:, 1]
                src += (bm + mids).tolist()
                dst += (bcm + cmids).tolist()
                etype += [self.num_rels] * (mids.shape[0])

                self.metapath_name[self.num_rels] = ('miRNA', self.num_rels, 'DimiRNA')
                self.num_rels += 1

                src += (bcm + cmids).tolist()
                dst += (bm + mids).tolist()
                etype += [self.num_rels] * (mids.shape[0])
                self.num_nodes = bcm + bcm_numb
                self.node[2] = ['DimiRNA', self.node[1][-1], self.node[1][-1] + self.cm_numb, self.cm_numb]

                self.metapath_name[self.num_rels] = ('DimiRNA', self.num_rels, 'miRNA')
                self.num_rels += 1

                """ d_c.csv """
                self.bcd = bcd = bcm + bcm_numb
                temp = adj_matrix_dc.nonzero()
                dids, cdids = temp[:, 0], temp[:, 1]
                src += (bd + dids).tolist()
                dst += (bcd + cdids).tolist()
                etype += [self.num_rels] * (dids.shape[0])
                self.metapath_name[self.num_rels] = ('disease', self.num_rels, 'Didisease')

                self.num_rels += 1

                src += (bcd + cdids).tolist()
                dst += (bd + dids).tolist()
                etype += [self.num_rels] * (dids.shape[0])
                self.num_nodes = bcd + bcd_numb

                self.node[3] = ['Didisease', self.node[2][-1], self.node[2][-1] + self.cd_numb, self.cd_numb]
                self.metapath_name[self.num_rels] = ('DimiRNA', self.num_rels, 'miRNA')

                """ test v node """
                self.num_nodes = self.num_nodes + 2

                self.src = src
                self.dst = dst
                self.etype = etype

                self.graph = dgl.graph((self.src, self.dst), num_nodes=self.num_nodes).to(self.device)
                self.graph.edata['type'] = torch.LongTensor(self.etype).to(self.device)
                self_loop_src = torch.arange(self.num_nodes).to(self.device)
                self_loop_dst = torch.arange(self.num_nodes).to(self.device)
                self.graph.add_edges(self_loop_src, self_loop_dst)
                neg_beta = self.config.neg_beta
                neg_beta = neg_beta + 0.2
                self.x_encode, self.edge_and_label = get_y_data(self.adj_matrix, mm_list=m_encoding_list, dd_list=d_encoding_list,
                                                                x_encoding=x_encoding, neg_beta=neg_beta)
                self.graph.edata['old_type'] = self.graph.edata['type'].clone()
                self._predata = [self.adj_matrix,
                                 mss_list,
                                 dss_list,
                                 m_encoding_list,
                                 d_encoding_list,
                                 x_encoding]
                self.config.get_data_params(self)

    def resplit_Gdata(self,neg_beta=None):
        (md,
         mss_list,
         dss_list,
         m_encoding_list,
         d_encoding_list,
         x_encoding) =  self._predata
        if neg_beta is None:
            neg_beta = self.config.neg_beta
        neg_beta = neg_beta + 0.02 # used BPRloss
        self.x_encode, self.edge_and_label = get_y_data(md, mm_list=m_encoding_list, dd_list=d_encoding_list, x_encoding=x_encoding, neg_beta=neg_beta)
        self.graph.edata['old_type'] = self.graph.edata['type'].clone()
        self.config.get_data_params(self)

    def add_new_metapath(self, metapath_adj_matrix, metapath_node1, metapath_node2, metapath_name=None, bidirectional=True): # meta-path
        """
        Adds a new meta-path to the graph based on the given adjacency matrix and node names.

        Parameters:
        -----------
        metapath_adj_matrix : torch.Tensor or numpy.ndarray
            The adjacency matrix representing the meta-path relationships between metapath_node1 and metapath_node2.

        metapath_node1 : str or int
            The first node (start node) of the meta-path. Can be provided as a string (name) or an integer (ID).

        metapath_node2 : str or int
            The second node (end node) of the meta-path. Can be provided as a string (name) or an integer (ID).

        metapath_name : str, optional
            The name for the new meta-path. If not provided, no name is associated with the meta-path. Default is None.

        bidirectional : bool, optional
            Specifies whether the meta-path is bidirectional (i.e., both directions between nodes).
            If True, both the original and reverse meta-paths are added. Default is True.

        Raises:
        -------
        AssertionError:
            If the provided node names or IDs are not present in the current graph nodes.

        Updates:
        --------
        self.src : list
            Source nodes for the added meta-path.

        self.dst : list
            Target nodes for the added meta-path.

        self.etype : list
            Edge types (relationships) for the meta-paths added.

        self.metapath_name : dict
            A dictionary mapping edge types to their corresponding meta-path names.

        self.num_nodes : int
            Total number of nodes in the graph after adding the new meta-path.

        self.num_rels : int
            Total number of edge types (relationships) after adding the new meta-path.

        Notes:
        ------
        - If either `metapath_node1` or `metapath_node2` are provided as strings, the function maps these names to their
          corresponding IDs in the graph.
        - The function checks if both nodes already exist in the graph, and then adds edges between these nodes according
          to the provided adjacency matrix.
        - If the `bidirectional` flag is set to True, reverse edges between the nodes are also added.
        - New virtual nodes are created if the meta-path involves new nodes.
        - The final graph object (with the added meta-path) is constructed using the DGL (Deep Graph Library) framework.
        - Self-loops (edges from a node to itself) are added to all nodes at the end.
        """
        name1 = metapath_node1
        name2 = metapath_node2
        if type(metapath_node1) == str:
            for k,v in self.node.items():
                if v[0] == metapath_node1:
                    name1 = metapath_node1
                    metapath_node1=k
                    continue

        if type(metapath_node2) == str:
            for k,v in self.node.items():
                if v[0] == metapath_node1:
                    name2 = metapath_node2
                    metapath_node2 = k
                    continue

        assert metapath_node1 in self.node.keys() or metapath_node2 in self.node.keys()
        new_node_number = 2
        self.num_nodes = self.num_nodes-2
        if metapath_node1 in self.node.keys() and metapath_node2 in self.node.keys():
            assert metapath_adj_matrix.shape[0] == self.node[metapath_node1][-1]
            assert metapath_adj_matrix.shape[1] == self.node[metapath_node2][-1]
            temp = metapath_adj_matrix.nonzero()
            self.num_rels += 1
            sids, tids = temp[:, 0], temp[:, 1]
            self.src += (self.node[metapath_node1][1] + sids).tolist()
            self.dst += (self.node[metapath_node2][1] + tids).tolist()
            self.etype += [self.num_rels] * (sids.shape[0])
            if metapath_name != None:
                self.metapath_name[self.num_rels] = metapath_name

            if bidirectional==True:
                self.num_rels += 1
                self.src += (self.node[metapath_node2][1] + tids).tolist()
                self.dst += (self.node[metapath_node1][1] + sids).tolist()
                self.etype += [self.num_rels] * (sids.shape[0])
                if metapath_name != None:
                    self.metapath_name[self.num_rels] = metapath_name[::-1]


        elif metapath_node1 in self.node.keys():
            assert metapath_adj_matrix.shape[0] == self.node[metapath_node1][-1]
            mshape = metapath_adj_matrix.shape
            temp = metapath_adj_matrix.nonzero()
            self.num_rels += 1
            sids, tids = temp[:, 0], temp[:, 1]
            self.src += (self.node[metapath_node1][1] + sids).tolist()
            self.dst += (self.num_nodes + tids).tolist()
            self.etype += [self.num_rels] * (sids.shape[0])
            if metapath_name != None:
                self.metapath_name[self.num_rels] = metapath_name

            if bidirectional==True:
                self.num_rels += 1
                self.src += (self.num_nodes + tids).tolist()
                self.dst += (self.node[metapath_node1][1] + sids).tolist()
                self.etype += [self.num_rels] * (sids.shape[0])
                if metapath_name != None:
                    self.metapath_name[self.num_rels] = metapath_name[::-1]

            self.node[max(self.node.keys())+1] = [name2, self.num_nodes, self.num_nodes + mshape[1], mshape[1]]
            self.num_nodes += mshape[1]

            return

        elif metapath_node2 in self.node.keys():
            assert metapath_adj_matrix.shape[1] == self.node[metapath_node2][-1]
            mshape = metapath_adj_matrix.shape
            temp = metapath_adj_matrix.nonzero()
            self.num_rels += 1
            sids, tids = temp[:, 0], temp[:, 1]
            self.src += (self.num_nodes + sids).tolist()
            self.dst += (self.node[metapath_node1][1] + tids).tolist()
            self.etype += [self.num_rels] * (sids.shape[0])
            if metapath_name != None:
                self.metapath_name[self.num_rels] = metapath_name

            if bidirectional == True:
                self.num_rels += 1
                self.src += (self.node[metapath_node1][1] + tids).tolist()
                self.dst += (self.num_nodes + sids).tolist()
                self.etype += [self.num_rels] * (sids.shape[0])
                if metapath_name != None:
                    self.metapath_name[self.num_rels] = metapath_name[::-1]

            self.node[max(self.node.keys()) + 1] = [name1, self.num_nodes, self.num_nodes + mshape[0], mshape[0]]
            self.num_nodes += mshape[0]

        ''' test v node '''
        self.num_nodes = self.num_nodes + 2

        self.graph = dgl.graph((self.src, self.dst), num_nodes=self.num_nodes).to(self.device)
        self.graph.edata['type'] = torch.LongTensor(self.etype).to(self.device)

        self_loop_src = torch.arange(self.num_nodes).to(self.device)
        self_loop_dst = torch.arange(self.num_nodes).to(self.device)
        self.graph.add_edges(self_loop_src, self_loop_dst)
        self.graph.edata['old_type'] = self.graph.edata['type'].clone()


    def read_new_ss(self,mss_list=None, dss_list=None):
        mss_matrix_list = []
        dss_matrix_list = []
        if mss_matrix_list != None:
            for mss_f in mss_list:
                mss_matrix_list.append(pd.read_csv(mss_f, header=None, index_col=None).values[np.newaxis, ...])
            self.new_mss = np.concatenate(mss_matrix_list, axis=0).mean(0)
            if self.new_mss.shape[0] == self.mss.shape[-1]:
                self.new_mss = self.new_mss.T
            assert self.new_mss.shape[-1] == self.mss.shape[-1]

        if dss_list != None:
            for dss_f in dss_list:
                dss_matrix_list.append(pd.read_csv(dss_f, header=None, index_col=None).values[np.newaxis, ...])

            self.new_dss = np.concatenate(dss_matrix_list, axis=0).mean(0)

            if self.new_dss.shape[0] == self.dss.shape[-1]:
                self.new_dss = self.new_dss.T

            assert self.new_dss.shape[-1] == self.dss.shape[-1]


    def get_new_mss(self):
        for i, ss in enumerate(self.new_mss):
            self.mss_idx = i
            yield ss, i

    def get_new_dss(self):
        for i, ss in enumerate(self.new_dss):
            self.dss_idx = i
            yield ss, i

    def get_new_RNA_ass(self, idx=None):
        if idx==None:
            if self.mss_idx !=None:
                idx = self.mss_idx
            else:
                idx = 0
        return self.new_RNA_ass[idx]

    def get_new_Disease_ass(self, idx=None):
        if idx==None:
            if self.dss_idx !=None:
                idx = self.dss_idx
            else:
                idx = 0
        return self.new_Disease_ass[idx]


    def get_new_ass(self, midx=None, didx=None):
        if midx==None:
            if self.mss_idx !=None:
                midx = self.mss_idx
            else:
                midx = 0
        if didx==None:
            if self.dss_idx !=None:
                didx = self.dss_idx
            else:
                didx = 0
        return self.new_RNA_new_Disease_ass[midx, didx]


    def add_one_new_mss(self, new_mss=None):
        new_mss = new_mss.reshape((1, -1))
        assert new_mss.shape[-1] == self.mss.shape[-1]
        mss = self.mss.copy()
        mss[-1] = new_mss.reshape((1,-1))
        self.new_mss_graph = self.get_ss_edge(mss,new_ss=True)
        new_medges = torch.tensor(self.mss_Di.transform(new_mss))

        temp = new_medges.nonzero()
        mids, cmids = temp[:, 0], temp[:, 1]
        new_src = [self.num_nodes-2 for _ in range(mids.shape[0])]
        new_dst = (self.bcm + cmids).tolist()
        etype = [2] * mids.shape[0]

        new_src += (self.bcm + cmids).tolist()
        new_dst += [self.num_nodes-2 for _ in range(mids.shape[0])]
        etype += [3] * mids.shape[0]

        self.graph.add_edges(new_src, new_dst)
        self.graph.edata['type'][-len(etype):] = torch.tensor(etype).to(self.device)


    def add_one_new_dss(self, new_dss=None):
        new_dss = new_dss.reshape((1, -1))
        assert new_dss.shape[-1] == self.dss.shape[-1]
        dss = self.dss.copy()
        dss[-1] = new_dss.reshape((1,-1))
        self.new_dss_graph = self.get_ss_edge(dss, new_ss=True)
        new_dedges = torch.tensor(self.dss_Di.transform(new_dss))

        temp = new_dedges.nonzero()
        dids, cdids = temp[:, 0], temp[:, 1]
        new_src = [self.num_nodes-1 for _ in range(dids.shape[0])]
        new_dst = (self.bcd + cdids).tolist()
        etype = [4] * dids.shape[0]

        new_src += (self.bcm + cdids).tolist()
        new_dst += [self.num_nodes-1 for _ in range(dids.shape[0])]
        etype += [5] * dids.shape[0]

        self.graph.add_edges(new_src, new_dst)
        self.graph.edata['type'][-len(etype):] = torch.tensor(etype).to(self.device)


    def read_new_RNAassDisease(self, new_md_file=None):
        if new_md_file !=None:
            adj_matrix = pd.read_csv(new_md_file, header=None, index_col=None).values
            if adj_matrix.shape[0] == self.adj_matrix.shape[0]:
                adj_matrix = adj_matrix.T
            assert adj_matrix.shape[-1] == self.adj_matrix.shape[-1]
            assert adj_matrix.shape[0] == self.new_mss.shape[0]
            self.new_RNA_ass = adj_matrix


    def read_new_DiseaseassRNA(self, new_md_file=None):
        if new_md_file !=None:

            adj_matrix = pd.read_csv(new_md_file, header=None, index_col=None).values
            if adj_matrix.shape[0] == self.adj_matrix.shape[0]:
                adj_matrix = adj_matrix.T
            assert adj_matrix.shape[-1] == self.adj_matrix.shape[0]
            assert adj_matrix.shape[0] == self.new_dss.shape[0]
            self.new_Disease_ass = adj_matrix


    def read_new_RNA_new_Disease_ass(self, new_md_file=None):
        if new_md_file !=None:
            adj_matrix = pd.read_csv(new_md_file, header=None, index_col=None).values
            assert adj_matrix.shape[0] == self.new_mss.shape[0]
            assert adj_matrix.shape[1] == self.new_dss.shape[0]
            self.new_RNA_new_Disease_ass = adj_matrix

    def clear_newss(self):
        edge_ids_to_remove = []

        for node in [self.num_nodes-2, self.num_nodes-1]:
            succ = self.graph.successors(node)
            prede = self.graph.predecessors(node)
            for nei in succ:
                if node != nei:
                    edge_ids_to_remove.append(self.graph.edge_ids(node, nei))
            for nei in prede:
                if node != nei:
                    edge_ids_to_remove.append(self.graph.edge_ids(nei, node))

        if edge_ids_to_remove:
            edge_ids_to_remove = torch.cat(edge_ids_to_remove)
            self.graph = dgl.remove_edges(self.graph, edge_ids_to_remove)

        self.graph.edata['type'] = self.graph.edata['old_type'].clone()

    def get_ss_edge(self, similarity_matrix, TopK_ss=None, new_ss=False):
        if TopK_ss==None:
            TopK_ss=self.TopK_ss
        N, M = similarity_matrix.shape
        if new_ss == False:
            N = M
        src = []
        dst = []
        weights = []

        for i in range(N):
            similarities = similarity_matrix[i, :]
            if i < M:
                similarities[i] = -np.inf

            top_k_indices = np.argsort(similarities)[-(TopK_ss+1):][::-1]
            top_k_values = similarities[top_k_indices]

            for j, weight in zip(top_k_indices, top_k_values):
                src.append(i)
                dst.append(j)
                weights.append(weight)

        g = dgl.graph((src+dst,dst+src), num_nodes=similarity_matrix.shape[0]).to(self.device)
        g.edata['weight'] = torch.tensor(weights+weights, dtype=torch.float32).to(self.device)
        return g

    def copy(self):
        return copy.deepcopy(self)

    def save_ASS_Embedding(self, path):
        joblib.dump(self.ASS_Embedding, path)

    def load_ASS_Embedding(self, path):
        self.ASS_Embedding=joblib.load(path)
        return self.ASS_Embedding

    def set_rna_name(self, rna_name):
        self.rna_name = np.array(rna_name).tolist()

    def set_disease_name(self, disease_name):
        self.disease_name = np.array(disease_name).tolist()

    def get_top_mirnas_for_disease(
            self,
            given_disease_name=None,
            top_n=20
    ):
        """
        Get the top N miRNAs most associated with the specified disease and return a DataFrame.

        Args:
            given_disease_name: The name of the specified disease.
            top_n: The number of miRNAs to return, default is 20.

        Returns:
            pandas.DataFrame: A DataFrame containing the rank, miRNA name, and association score.
        """
        # Get miRNA names and disease names
        mirna_names = self.rna_name
        disease_names = self.disease_name
        edge_scores = self.ASS_Embedding['EdgeScores']
        edge = self.ASS_Embedding['edge_matrix']

        # Validate if the disease name exists
        if given_disease_name not in disease_names:
            raise ValueError(f"The disease name '{given_disease_name}' is not in the disease list.")

        # Find the index of the given disease name in the disease name list
        disease_index = disease_names.index(given_disease_name)

        # Get the column scores corresponding to this disease
        if edge_scores is None:
            raise ValueError("A valid score matrix needs to be provided.")

        column_scores = edge_scores[:, disease_index]
        top_indices = np.argsort(column_scores)[-top_n:][::-1]
        top_scores = column_scores[top_indices]

        column_scores = edge[:, disease_index]
        top_indices = np.argsort(column_scores)[-top_n:][::-1]
        top_edge = column_scores[top_indices]

        # Create a DataFrame
        data = {
            'index': range(1, top_n + 1),
            'RNA name': [mirna_names[i] for i in top_indices],
            'Association score': top_scores,
            'Association edge': top_edge,
        }

        return pd.DataFrame(data)

    def get_top_mirnas_for_multiple_diseases(
            self,
            disease_list=None,
            top_n=20
    ):
        """
        Retrieve the top N miRNAs most associated with multiple diseases and return a hierarchical DataFrame.

        Args:
            disease_list: List of specified disease names.
            top_n: Number of miRNAs to return for each disease, default is 20.

        Returns:
            pandas.DataFrame: A DataFrame with hierarchical indexing, where the outer index is the disease
                              and the inner index is the ranking information.
        """
        if not disease_list:
            raise ValueError("A list of disease names must be provided.")

        # Validate all disease names exist
        invalid_diseases = [d for d in disease_list if d not in self.disease_name]
        if invalid_diseases:
            raise ValueError(f"The following disease names are not in the list: {', '.join(invalid_diseases)}")

        # Initialize an empty list of DataFrames
        all_results = []

        # Execute the query for each disease
        for disease in disease_list:
            disease_index = self.disease_name.index(disease)
            column_scores = self.ASS_Embedding['EdgeScores'][:, disease_index]

            # Find the indices and scores of the top N miRNAs
            top_indices = np.argsort(column_scores)[-top_n:][::-1]
            top_scores = column_scores[top_indices]

            column_scores = self.ASS_Embedding['edge_matrix'][:, disease_index]
            top_indices = np.argsort(column_scores)[-top_n:][::-1]
            top_edge = column_scores[top_indices]

            # Create a DataFrame for the current disease
            disease_df = pd.DataFrame({
                'index': range(1, top_n + 1),
                'RNA name': [self.rna_name[i] for i in top_indices],
                'Association score': top_scores,
                'Association edge': top_edge,
                'disease': disease  # Add disease column for grouping
            })

            all_results.append(disease_df)

        # Combine all results
        if not all_results:
            return pd.DataFrame()

        combined_df = pd.concat(all_results, ignore_index=True)

        # Set hierarchical index
        combined_df = combined_df.set_index(['disease', 'index'])
        return combined_df

    def get_overlapping_disease_for_disease(
            self,
            given_disease_name=None,
            top_n=10,
            top_n_disease=5,
            min_score=0.5,
    ):
        """
        Retrieve the top N genes most associated with a specified disease and find the top M diseases with the highest Spearman correlation to these genes.

        Args:
            given_disease_name: Name of the specified disease
            top_n: Number of genes to return, default is 20
            top_n_disease: Number of related diseases to return, default is 5

        Returns:
            pandas.DataFrame: DataFrame containing disease, ranking, gene name, association score, and association edge
        """
        # Get gene names and disease names
        gene_names = self.rna_name
        disease_names = self.disease_name
        edge_scores = self.ASS_Embedding['EdgeScores']
        edge_matrix = self.ASS_Embedding['edge_matrix']

        # Validate if the disease name exists
        if given_disease_name not in disease_names:
            raise ValueError(f"The disease name '{given_disease_name}' is not in the disease list")

        # Find the index of the given disease name
        disease_index = disease_names.index(given_disease_name)

        # Validate the effectiveness of the score matrix
        if edge_scores is None or edge_scores.size == 0:
            raise ValueError("A valid score matrix must be provided")
        if edge_matrix is None or edge_matrix.size == 0:
            raise ValueError("A valid edge matrix must be provided")

        # Get top genes for the given disease (applying min_score filter)
        column_scores = edge_scores[:, disease_index]
        top_indices = np.argsort(column_scores)[-top_n:][::-1]
        top_genes = [gene_names[i] for i in top_indices]
        top_scores = column_scores[top_indices]
        top_edge = edge_matrix[:, disease_index][top_indices]

        # Apply min_score filtering
        valid_main = np.where(top_scores >= min_score)[0]
        top_genes = [top_genes[i] for i in valid_main]
        top_scores = top_scores[valid_main]
        top_edge = top_edge[valid_main]

        # Calculate Spearman correlation with other diseases
        spearman_corrs = {}
        all_other_diseases = [d for d in disease_names if d != given_disease_name]

        for disease in all_other_diseases:
            disease_idx = disease_names.index(disease)
            disease_scores = edge_scores[:, disease_idx]

            # Calculate Spearman correlation (need at least 3 data points)
            if len(disease_scores) >= top_n:
                corr, _ = spearmanr(column_scores, disease_scores)
                spearman_corrs[disease] = corr
            else:
                spearman_corrs[disease] = np.nan

        # Sort by correlation and get top_n_disease diseases (ignoring NaN values)
        valid_diseases = [d for d in all_other_diseases if not np.isnan(spearman_corrs[d])]
        sorted_diseases = sorted(
            valid_diseases,
            key=lambda x: spearman_corrs[x],
            reverse=True
        )[:top_n_disease]

        # Construct output DataFrame
        all_results = []

        # Add top genes data for the given disease
        main_data = {
            'disease': ['Main: ' + given_disease_name] * len(top_genes),
            'index': range(1, len(top_genes) + 1),
            'RNA name': top_genes,
            'Association score': top_scores,
            'Association edge': top_edge
        }
        all_results.append(pd.DataFrame(main_data))

        # Add related genes data for selected diseases
        for disease in sorted_diseases:
            disease_idx = disease_names.index(disease)
            disease_scores = edge_scores[:, disease_idx]
            disease_edge = edge_matrix[:, disease_idx]

            # Get genes shared between this disease and the main disease (with scores above threshold)
            shared_genes = []
            shared_scores = []
            shared_edges = []

            for gene in top_genes:
                if gene in gene_names:
                    gene_idx = gene_names.index(gene)
                    if disease_scores[gene_idx] >= min_score:
                        shared_genes.append(gene)
                        shared_scores.append(disease_scores[gene_idx])
                        shared_edges.append(disease_edge[gene_idx])

            # Sort by score in descending order
            sorted_indices = np.argsort(shared_scores)[::-1]
            shared_genes = [shared_genes[i] for i in sorted_indices]
            shared_scores = [shared_scores[i] for i in sorted_indices]
            shared_edges = [shared_edges[i] for i in sorted_indices]

            # Add correlation value to disease name
            disease_label = f"{disease} (spearman: {spearman_corrs[disease]:.4f})"

            disease_data = {
                'disease': [disease_label] * len(shared_genes),
                'index': range(1, len(shared_genes) + 1),
                'RNA name': shared_genes,
                'Association score': shared_scores,
                'Association edge': shared_edges
            }
            all_results.append(pd.DataFrame(disease_data))

        # Combine all results
        combined_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()

        # Set multi-level index
        if not combined_df.empty:
            combined_df = combined_df.set_index(['disease', 'index'])

        return combined_df

    def visualize_mirna_disease_graph(self, data_df, node1_name='Disease', node2_name='miRNA'):
        """
        Draw a relationship graph containing Disease, known miRNA, and predicted miRNA nodes
        :param data_df: DataFrame containing disease, miRNA information, and association edge data
        """
        # Create a figure
        plt.figure(figsize=(16, 12), dpi=100)

        # Create an undirected graph
        G = nx.Graph()

        # Node type and style mapping
        node_types = {
            f'{node1_name}': {'shape': 'd', 'color': '#E74C3C', 'size': 2200},
            f'Known {node2_name}': {'shape': 'o', 'color': '#3498DB', 'size': 900},
            f'Predicted {node2_name}': {'shape': 's', 'color': '#F39C12', 'size': 900}
        }

        # Edge type and style mapping
        edge_types = {
            'Known': {'color': '#3498DB', 'width': 2.0, 'style': 'solid'},
            'Predicted': {'color': '#F39C12', 'width': 1.5, 'style': 'dashed'}
        }

        # Collect all nodes
        diseases = set()
        mirnas = {}

        # Process each disease group
        for disease_name, group_df in data_df.groupby(level='disease'):
            # Ensure each word in the disease name is capitalized
            formatted_disease = disease_name.title()
            diseases.add(formatted_disease)

            # Add disease node
            G.add_node(formatted_disease, node_type='Disease')

            for idx, row in group_df.iterrows():
                mirna_name = row['RNA name']
                edge_type = 'Known' if row['Association edge'] == 1 else 'Predicted'

                # Add miRNA node (if not already added)
                if mirna_name not in G:
                    node_type = 'Known miRNA' if edge_type == 'Known' else 'Predicted miRNA'
                    G.add_node(mirna_name, node_type=node_type)
                    mirnas[mirna_name] = node_type

                # Add edge between disease and miRNA
                G.add_edge(formatted_disease, mirna_name, edge_type=edge_type)

        # Use Fruchterman-Reingold force-directed layout - more natural dispersion
        pos = nx.spring_layout(G, k=0.5, iterations=150, seed=42)

        # Calculate node positions - enhance dispersion
        # 1. Extract positions of all nodes
        x_values = [pos[node][0] for node in G.nodes]
        y_values = [pos[node][1] for node in G.nodes]

        # 2. Calculate position range
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        # 3. Expand position range to make nodes more dispersed
        scale_factor = 2.2  # Increase this value to make nodes more dispersed
        for node in pos:
            # Normalize to [0,1] range
            norm_x = (pos[node][0] - x_min) / (x_max - x_min) if x_max != x_min else 0.5
            norm_y = (pos[node][1] - y_min) / (y_max - y_min) if y_max != y_min else 0.5

            # Expand to [-scale_factor, scale_factor] range
            pos[node] = (
                (norm_x - 0.5) * 2 * scale_factor,
                (norm_y - 0.5) * 2 * scale_factor
            )

        # Draw edges
        for u, v, data in G.edges(data=True):
            edge_type = data.get('edge_type', 'Predicted')
            style = edge_types[edge_type]
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)],
                edge_color=style['color'],
                width=style['width'],
                style=style['style'],
                alpha=0.7
            )

        # Draw nodes
        for node_type, style in node_types.items():
            nodes = [n for n, attrs in G.nodes(data=True) if attrs.get('node_type') == node_type]
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes,
                node_shape=style['shape'],
                node_color=style['color'],
                node_size=style['size'],
                alpha=0.9,
                edgecolors='black',
                linewidths=1.5
            )

        # Create labels for all nodes
        labels = {}
        for node in G.nodes:
            # Ensure each word in the disease name is capitalized
            if G.nodes[node].get('node_type') == 'Disease':
                labels[node] = node
            else:
                labels[node] = node

        # Calculate label position offsets to avoid overlap
        label_pos = {}
        for node, (x, y) in pos.items():
            # Set different offsets based on node type
            if G.nodes[node].get('node_type') == 'Disease':
                # Disease node labels placed below
                label_pos[node] = (x, y)
            else:
                # miRNA node labels placed at top-right
                label_pos[node] = (x - 0.01, y)

        # Draw labels
        for node, (x, y) in label_pos.items():
            # Set different font sizes
            font_size = 13 if G.nodes[node].get('node_type') == 'Disease' else 11
            font_weight = 'bold' if G.nodes[node].get('node_type') == 'Disease' else 'normal'

            plt.text(
                x, y, labels[node],
                fontsize=font_size,
                fontweight=font_weight,
                fontfamily='sans-serif',
                bbox=dict(
                    facecolor='white',
                    alpha=0.85,
                    edgecolor='#CCCCCC',
                    boxstyle='round,pad=0.3'
                ),
                horizontalalignment='center' if G.nodes[node].get('node_type') == 'Disease' else 'left',
                verticalalignment='center'
            )

        # Create legend elements
        legend_elements = [
            Line2D([0], [0], marker='d', color='w', label='Disease',
                   markerfacecolor=node_types[f'{node1_name}']['color'], markersize=15),
            Line2D([0], [0], marker='o', color='w', label='Known miRNA',
                   markerfacecolor=node_types[f'Known {node2_name}']['color'], markersize=12),
            Line2D([0], [0], marker='s', color='w', label='Predicted miRNA',
                   markerfacecolor=node_types[f'Predicted {node2_name}']['color'], markersize=12),
            Line2D([0], [0], color=edge_types['Known']['color'],
                   linestyle='-', lw=3, label='Known Association'),
            Line2D([0], [0], color=edge_types['Predicted']['color'],
                   linestyle='--', lw=2, label='Predicted Association')
        ]

        plt.legend(
            handles=legend_elements,
            loc='lower right',
            fontsize=14,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.9
        )

        # Add title
        plt.title(f'{node2_name}-{node1_name} Association Network', fontsize=20, pad=20, y=0.93)

        # Add grid background
        plt.grid(False)
        plt.gca().set_facecolor('#F8F9F9')

        # Beautify borders
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)

        # Set margins
        plt.margins(0.15)

        # Turn off axes
        plt.axis('off')

        # Adjust layout
        plt.tight_layout()

        # Display the graph
        return plt

    def visualize_prediction_matrices(self, data_dict=None, zoom_regions=[(0.1, 0.3), (0.7, 0.9)], data_name=None):
        """
        Visualize the ground truth matrix and prediction matrix of association prediction tasks, and zoom in on specified regions

        Parameters:
        data_dict (dict): Dictionary containing prediction data
        zoom_regions (list): List of regions to zoom in on, each region is represented by (start, end)
        """
        # Extract matrices
        if data_name is None:
            data_name = self.config.data_name if hasattr(self, 'data_name') else 'Data'
        if data_dict is None:
            data_dict = self.ASS_Embedding
        true_matrix = data_dict['edge_matrix']
        pred_matrix = data_dict['EdgeScores']

        # Verify matrix shapes
        if true_matrix.shape != pred_matrix.shape:
            raise ValueError(
                f"Matrix shapes do not match: true_matrix {true_matrix.shape}, pred_matrix {pred_matrix.shape}")

        # Create custom colormap
        colors = ["white", "blue", "red"]
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

        # Create figure
        fig = plt.figure(figsize=(16, 12))

        # Create main grid layout: 2 rows (ground truth and prediction), each row contains main view and zoomed views
        gs_main = GridSpec(2, len(zoom_regions) + 1, figure=fig, width_ratios=[3] + [1] * len(zoom_regions),
                           height_ratios=[1, 1], hspace=0.3, wspace=0.2)

        # Add title
        fig.suptitle(f'{data_name}: Association Prediction Visualization', fontsize=24, fontweight='bold', y=0.94)

        # Ground truth matrix main view
        ax_true_main = fig.add_subplot(gs_main[0, 0])
        im_true = ax_true_main.imshow(true_matrix, cmap=cmap, aspect='auto', interpolation='nearest')
        ax_true_main.set_title('Ground Truth Associations', fontsize=18, pad=15)
        ax_true_main.set_xlabel('Diseases', fontsize=18)
        ax_true_main.set_ylabel('miRNAs', fontsize=18)
        ax_true_main.tick_params(axis='both', which='major', labelsize=18)

        # Prediction matrix main view
        ax_pred_main = fig.add_subplot(gs_main[1, 0])
        im_pred = ax_pred_main.imshow(pred_matrix, cmap=cmap, aspect='auto', interpolation='nearest')
        ax_pred_main.set_title('Associations', fontsize=18, pad=15)
        ax_pred_main.set_xlabel('Diseases', fontsize=18)
        ax_pred_main.set_ylabel('miRNAs', fontsize=18)
        ax_pred_main.tick_params(axis='both', which='major', labelsize=18)

        # Add zoomed views for each region
        for i, region in enumerate(zoom_regions):
            start, end = region
            x1, x2 = int(start * true_matrix.shape[1]), int(end * true_matrix.shape[1])
            y1, y2 = int(start * true_matrix.shape[0]), int(end * true_matrix.shape[0])

            # Add bounding box on ground truth main view
            rect_true = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3,
                                      edgecolor='lime', facecolor='none', zorder=10)
            ax_true_main.add_patch(rect_true)

            # Add bounding box on prediction main view
            rect_pred = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3,
                                      edgecolor='lime', facecolor='none', zorder=10)
            ax_pred_main.add_patch(rect_pred)

            # Ground truth matrix zoomed view
            ax_true_inset = fig.add_subplot(gs_main[0, i + 1])
            ax_true_inset.imshow(true_matrix[y1:y2, x1:x2], cmap=cmap, aspect='auto', interpolation='nearest')
            ax_true_inset.set_title(f'Zoom Region {i + 1}: {start * 100:.0f}%-{end * 100:.0f}%', fontsize=18)
            ax_true_inset.set_xlabel('Diseases', fontsize=18)
            # ax_true_inset.set_ylabel('miRNAs', fontsize=18)
            ax_true_inset.tick_params(axis='both', which='major', labelsize=18)

            # Set zoomed view axis labels to real values and add intermediate ticks
            self._set_zoom_ticks(ax_true_inset, x1, x2, y1, y2)

            # Prediction matrix zoomed view
            ax_pred_inset = fig.add_subplot(gs_main[1, i + 1])
            ax_pred_inset.imshow(pred_matrix[y1:y2, x1:x2], cmap=cmap, aspect='auto', interpolation='nearest')
            ax_pred_inset.set_title(f'Zoom Region {i + 1}: {start * 100:.0f}%-{end * 100:.0f}%', fontsize=18)
            ax_pred_inset.set_xlabel('Diseases', fontsize=18)
            # ax_pred_inset.set_ylabel('miRNAs', fontsize=18)
            ax_pred_inset.tick_params(axis='both', which='major', labelsize=18)

            # Set zoomed view axis labels to real values and add intermediate ticks
            self._set_zoom_ticks(ax_pred_inset, x1, x2, y1, y2)

        # Add color bar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(im_pred, cax=cbar_ax)
        cbar.set_label('Association Strength', fontsize=18)
        cbar.ax.tick_params(labelsize=18)

        # Add legend explanation
        legend_elements = [
            plt.Line2D([0], [0], color='black', marker='s', markersize=15,
                       label='No Association', markerfacecolor='white'),
            plt.Line2D([0], [0], color='black', marker='s', markersize=15,
                       label='Association', markerfacecolor='red'),
            plt.Line2D([0], [0], color='lime', lw=3, label='Zoom Region')
        ]

        fig.legend(handles=legend_elements, loc='lower center',
                   bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=18, frameon=True)

        # # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 0.92, 0.97])
        return fig

    def _set_zoom_ticks(self, ax, x1, x2, y1, y2):
        """Set ticks for zoomed subplots, showing real coordinate values"""
        # Calculate number of ticks, ensuring it doesn't exceed a reasonable range
        num_xticks = min(5, x2 - x1)  # Maximum 5 ticks
        num_yticks = min(5, y2 - y1)

        # Generate evenly distributed tick positions and labels
        xticks = np.linspace(0, x2 - x1, num_xticks, dtype=int)
        xticklabels = np.linspace(x1, x2, num_xticks, dtype=int)

        yticks = np.linspace(0, y2 - y1, num_yticks, dtype=int)
        yticklabels = np.linspace(y1, y2, num_yticks, dtype=int)

        # Set ticks
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

    def visualize_embeddings(self, data_dict=None):
        """
        Visualize miRNA and disease embedding features

        Parameters:
        data_dict (dict): Dictionary containing prediction data
        """
        # Extract data
        if data_dict is None:
            data_dict = self.ASS_Embedding
        miRNA_embedding = data_dict['miRNA_embeding']
        disease_embedding = data_dict['Disease_embeding']

        # Create figure
        fig = plt.figure(figsize=(20, 15))

        # Create grid layout
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1, 1])

        # Add main title
        fig.suptitle('Embedding Feature Analysis', fontsize=24, fontweight='bold', y=0.98)

        # 1. miRNA embedding feature visualization (PCA)
        ax1 = fig.add_subplot(gs[0, 0])
        # Dimensionality reduction
        pca = PCA(n_components=2)
        miRNA_2d = pca.fit_transform(miRNA_embedding)
        # Draw scatter plot
        ax1.scatter(miRNA_2d[:, 0], miRNA_2d[:, 1], s=50, alpha=0.8,
                    edgecolor='k', color='steelblue')
        ax1.set_title(f'miRNA Embeddings (PCA)\nExplained Variance: {pca.explained_variance_ratio_.sum():.2f}',
                      fontsize=18, pad=15)
        ax1.set_xlabel('PCA Component 1', fontsize=14)
        ax1.set_ylabel('PCA Component 2', fontsize=14)
        ax1.grid(True, alpha=0.3)

        # 2. miRNA embedding feature visualization (t-SNE)
        ax2 = fig.add_subplot(gs[0, 1])
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        miRNA_tsne = tsne.fit_transform(miRNA_embedding)
        # Draw scatter plot
        ax2.scatter(miRNA_tsne[:, 0], miRNA_tsne[:, 1], s=50, alpha=0.8,
                    edgecolor='k', color='mediumseagreen')
        ax2.set_title('miRNA Embeddings (t-SNE)', fontsize=18, pad=15)
        ax2.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax2.set_ylabel('t-SNE Dimension 2', fontsize=14)
        ax2.grid(True, alpha=0.3)

        # 3. miRNA embedding feature clustering analysis
        ax3 = fig.add_subplot(gs[0, 2])
        # Perform hierarchical clustering on miRNA embedding features
        Z = linkage(miRNA_embedding, method='ward')
        dendrogram(Z, ax=ax3, truncate_mode='lastp', p=20, show_leaf_counts=True)
        ax3.set_title('miRNA Embeddings Hierarchical Clustering', fontsize=18, pad=15)
        ax3.set_xlabel('miRNA Index', fontsize=14)
        ax3.set_ylabel('Distance', fontsize=14)
        ax3.tick_params(axis='both', which='major', labelsize=10)

        # 4. Disease embedding feature visualization (PCA)
        ax4 = fig.add_subplot(gs[1, 0])
        # Dimensionality reduction
        pca_disease = PCA(n_components=2)
        disease_2d = pca_disease.fit_transform(disease_embedding)
        # Draw scatter plot
        ax4.scatter(disease_2d[:, 0], disease_2d[:, 1], s=50, alpha=0.8,
                    edgecolor='k', color='darkorange')
        ax4.set_title(
            f'Disease Embeddings (PCA)\nExplained Variance: {pca_disease.explained_variance_ratio_.sum():.2f}',
            fontsize=18, pad=15)
        ax4.set_xlabel('PCA Component 1', fontsize=14)
        ax4.set_ylabel('PCA Component 2', fontsize=14)
        ax4.grid(True, alpha=0.3)

        # 5. Disease embedding feature visualization (t-SNE)
        ax5 = fig.add_subplot(gs[1, 1])
        # Use t-SNE for dimensionality reduction
        tsne_disease = TSNE(n_components=2, perplexity=30, random_state=42)
        disease_tsne = tsne_disease.fit_transform(disease_embedding)
        # Draw scatter plot
        ax5.scatter(disease_tsne[:, 0], disease_tsne[:, 1], s=50, alpha=0.8,
                    edgecolor='k', color='mediumpurple')
        ax5.set_title('Disease Embeddings (t-SNE)', fontsize=18, pad=15)
        ax5.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax5.set_ylabel('t-SNE Dimension 2', fontsize=14)
        ax5.grid(True, alpha=0.3)

        # 6. Disease embedding feature clustering analysis
        ax6 = fig.add_subplot(gs[1, 2])
        # Perform hierarchical clustering on disease embedding features
        Z_disease = linkage(disease_embedding, method='ward')
        dendrogram(Z_disease, ax=ax6, truncate_mode='lastp', p=20, show_leaf_counts=True)
        ax6.set_title('Disease Embeddings Hierarchical Clustering', fontsize=18, pad=15)
        ax6.set_xlabel('Disease Index', fontsize=14)
        ax6.set_ylabel('Distance', fontsize=14)
        ax6.tick_params(axis='both', which='major', labelsize=10)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.98])
        plt.subplots_adjust(hspace=0.25, wspace=0.3)

        return fig

    def visualize_disease_disease_graph(self, data_df):
        """
        Draw a relationship graph containing Disease and associated miRNA nodes, showing edge scores
        :param data_df: DataFrame containing disease, miRNA information, and associated edge data
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        # Create figure
        plt.figure(figsize=(16, 12), dpi=100)

        # Create undirected graph
        G = nx.Graph()

        # Node type and style mapping (simplified to disease and associated RNA)
        node_types = {
            'Main Disease': {'shape': 'p', 'color': '#E74C3C', 'size': 2500},  # Main disease diamond
            'Other Disease': {'shape': 'd', 'color': '#3498DB', 'size': 2200},  # Other disease square
            'Associated miRNA': {'shape': 'o', 'color': '#F39C12', 'size': 1000}  # Associated RNA circle
        }

        # Edge style (uniform dashed line)
        edge_style = {'color': '#666666', 'width': 1.2, 'style': 'dashed'}

        # Collect all nodes
        diseases = set()
        mirnas = {}

        # Process each disease group
        for disease_name, group_df in data_df.groupby(level='disease'):
            # Process disease name
            is_main_disease = 'Main:' in disease_name
            disease_type = 'Main Disease' if is_main_disease else 'Other Disease'
            formatted_disease = disease_name.replace('Main: ', '').title()
            diseases.add(formatted_disease)

            # Add disease node (set different shapes based on whether it is the main disease)
            G.add_node(formatted_disease, node_type=disease_type)

            for idx, row in group_df.iterrows():
                mirna_name = row['RNA name']
                edge_score = row['Association score']  # Get edge score

                # Add miRNA node
                if mirna_name not in G:
                    G.add_node(mirna_name, node_type='Associated miRNA')
                    mirnas[mirna_name] = 'Associated miRNA'

                # Add edge between disease and miRNA, and save edge score
                G.add_edge(formatted_disease, mirna_name, edge_score=edge_score)

        # Optimize disease node distribution (using hierarchical layout)
        pos = nx.spring_layout(G, k=0.6, iterations=200, seed=42)

        # Separate disease nodes and miRNA nodes
        disease_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] in ['Main Disease', 'Other Disease']]
        mirna_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'Associated miRNA']

        # Adjust disease node positions to make them more dispersed
        if disease_nodes:
            # Calculate bounding box of disease nodes
            disease_pos = {n: pos[n] for n in disease_nodes}
            x_coords = [p[0] for p in disease_pos.values()]
            y_coords = [p[1] for p in disease_pos.values()]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            # Expand distribution range of disease nodes
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            dist = max(x_max - x_min, y_max - y_min) * 1.5  # Expansion distance

            # Rearrange disease nodes in a circular distribution
            n_diseases = len(disease_nodes)
            for i, node in enumerate(disease_nodes):
                angle = 2 * np.pi * i / n_diseases
                x = x_center + dist * np.cos(angle)
                y = y_center + dist * np.sin(angle)
                pos[node] = (x, y)

        # Draw nodes
        for node_type, style in node_types.items():
            nodes = [n for n, attrs in G.nodes(data=True) if attrs.get('node_type') == node_type]
            nx.draw_networkx_nodes(
                G, pos, nodelist=nodes,
                node_shape=style['shape'],
                node_color=style['color'],
                node_size=style['size'],
                alpha=0.9,
                edgecolors='black',
                linewidths=1.5
            )

        # Draw edges (uniform dashed lines)
        for u, v, data in G.edges(data=True):
            edge_score = data.get('edge_score', 0)
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, v)],
                edge_color=edge_style['color'],
                width=edge_style['width'],
                style=edge_style['style'],
                alpha=0.7
            )
            # Display edge score
            if u in disease_nodes and v in mirna_nodes:  # Only display scores on edges from disease to miRNA
                edge_pos = ((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2)
                plt.text(
                    edge_pos[0], edge_pos[1], f"{edge_score:.2f}",
                    fontsize=10, color='#333333',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                    ha='center', va='center'
                )

        # Create labels for all nodes
        labels = {node: node for node in G.nodes}

        # Draw labels (enlarge disease labels, normal miRNA labels)
        for node in G.nodes:
            node_type = G.nodes[node].get('node_type')
            x, y = pos[node]

            # Adjust label position
            if node_type in ['Main Disease', 'Other Disease']:
                pass
            else:
                # Place miRNA labels to the right of the node
                plt.text(x + 0.01, y, labels[node],
                         fontsize=12, fontfamily='sans-serif',
                         fontweight='normal',  # Explicitly set to normal font
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='#DDDDDD', boxstyle='round,pad=0.3'),
                         ha='left', va='center')

        for node in G.nodes:
            node_type = G.nodes[node].get('node_type')
            x, y = pos[node]
            if node_type in ['Main Disease', 'Other Disease']:
                # Place disease labels above the node
                plt.text(x, y + 0.01, labels[node],
                         fontsize=14, fontweight='bold', fontfamily='sans-serif',
                         bbox=dict(facecolor='white', alpha=0.8, edgecolor='#DDDDDD', boxstyle='round,pad=0.3'),
                         ha='center', va='bottom')

        # Add legend
        legend_elements = [
            Line2D([0], [0], marker='p', color='w', label='Main Disease',
                   markerfacecolor=node_types['Main Disease']['color'], markersize=15),
            Line2D([0], [0], marker='d', color='w', label='Other Disease',
                   markerfacecolor=node_types['Other Disease']['color'], markersize=15),
            Line2D([0], [0], marker='o', color='w', label='Associated miRNA',
                   markerfacecolor=node_types['Associated miRNA']['color'], markersize=12),
            Line2D([0], [0], color=edge_style['color'], linestyle='--', lw=2, label='Association')
        ]

        plt.legend(
            handles=legend_elements,
            loc='lower left',
            fontsize=14,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.9
        )

        # Add title
        plt.title('Disease-Disease Association Network with Edge Scores',
                  fontsize=22, pad=25, y=0.95)

        # Beautify background
        plt.gca().set_facecolor('#F8F9F9')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)

        # Turn off axes
        plt.axis('off')

        # Adjust layout
        plt.tight_layout()

        # Display figure
        return plt