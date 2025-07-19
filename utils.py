import pickle, random
import copy
import numpy as np
from sklearn.model_selection import ParameterGrid
import datetime
import pandas as pd
import time
import os
import psutil
import torch


def set_seed(seed):
    """Set the seed for all relevant random number generators to ensure reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_attr(config, param_search):
    """Generate configurations based on the parameter grid for hyperparameter search."""
    param_grid_list = list(ParameterGrid(param_search))
    for param in param_grid_list:
        new_config = copy.deepcopy(config)
        new_config.param_search = param_search
        new_config.search_args = {'arg_name': [], 'arg_value': []}
        for key, value in param.items():
            setattr(new_config, key, value)
            new_config.search_args['arg_name'].append(key)
            new_config.search_args['arg_value'].append(value)
            print(f"{key}: {value}")
        yield new_config

def load_data(data_path):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_model(model, save_path, optimizer=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data2save = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(data2save, save_path)


def load_model(model, load_path, optimizer=None):
    data2load = torch.load(load_path, map_location='cpu')
    model.load_state_dict(data2load['state_dict'])
    if optimizer is not None and data2load['optimizer'] is not None:
        optimizer = data2load['optimizer']


def fix_random_seed_as(seed):
    random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed(seed):
    torch.manual_seed(seed)
    # This line for random search should be commented out
    # random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_metrics(real_score, predict_score):
    real_score, predict_score = real_score.flatten(), predict_score.flatten()
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN

    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]

    # np.savetxt(roc_path.format(i), ROC_dot_matrix)

    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])

    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]

    # np.savetxt(pr_path.format(i), PR_dot_matrix)

    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])

    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    print( ' auc:{:.4f} ,aupr:{:.4f},f1_score:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, specificity:{:.4f}, precision:{:.4f}'.format( auc[0, 0],aupr[0, 0], f1_score, accuracy, recall, specificity, precision))
    return [real_score, predict_score,auc[0, 0], aupr[0, 0], f1_score, accuracy, recall, specificity, precision], \
           ['y_true', 'y_score', 'auc', 'prc', 'f1_score', 'acc', 'recall', 'specificity', 'precision']


def randomMinMax(N):
    normal_matrix = np.random.normal(loc=0.0, scale=1.0, size=(N, N))
    # Linear normalization to the 0-1 interval
    min_val = np.min(normal_matrix)
    max_val = np.max(normal_matrix)
    normalized_matrix = (normal_matrix - min_val) / (max_val - min_val)
    return normalized_matrix

def get_y_data(md, mm_list=[None,], dd_list=[None,], x_encoding=False, neg_beta=5):
    try:
        adj_matrix = pd.read_csv(md, header=None, index_col=None).values
    except:
        adj_matrix = md
    edge_index_pos = np.column_stack(np.argwhere(adj_matrix != 0))
    edge_index_pos = torch.tensor(edge_index_pos, dtype=torch.long)

    edge_index_neg = np.column_stack(np.argwhere(adj_matrix == 0))
    edge_index_neg = torch.tensor(edge_index_neg, dtype=torch.long)

    num_pos_edges_number = edge_index_pos.shape[1]
    selected_neg_edge_indices = torch.randint(high=edge_index_neg.shape[1], size=(int(num_pos_edges_number*neg_beta),),
                                               dtype=torch.long)
    edge_index_neg_selected = edge_index_neg[:, selected_neg_edge_indices]
    edg_index_all = torch.cat((edge_index_pos, edge_index_neg_selected), dim=1)
    y = torch.cat((torch.ones((edge_index_pos.shape[1], 1)),
                   torch.zeros((edge_index_neg_selected.shape[1], 1))), dim=0)  # Set all y values to 1 or 0

    if x_encoding==True:
        xe_1 = []
        xe_2 = []
        if len(mm_list) + len(dd_list) > 0:
            for i in range(0, len(mm_list)):
                xe_1.append(pd.read_csv(mm_list[i], header=None, index_col=None).values)

            for j in range(0, len(dd_list)):
                xe_2.append(pd.read_csv(dd_list[i], header=None, index_col=None).values)

            xe_1 = torch.tensor(np.array(xe_1).mean(0), dtype=torch.float32)
            xe_2 = torch.tensor(np.array(xe_2).mean(0), dtype=torch.float32)
    else:
        xe_1, xe_2 = None, None

    edg_index_all[-1] = edg_index_all[-1] + adj_matrix.shape[0]
    if xe_1 == None:
        return None, {'y': y, 'y_edge': edg_index_all}
    return {'x1':xe_1,'x2':xe_2}, {'y':y, 'y_edge':edg_index_all}


def print_execution_time(start_time, epoch_interval='all'):
    """Print the execution time for a set of epochs."""
    end_time = datetime.datetime.now()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total execution time for {epoch_interval} epochs: {hours} hours, {minutes} minutes, {seconds} seconds")
    return f'{minutes} minutes {seconds} seconds'

def mask_func(edg_index_all, mask_sp='RNA', test_rate=0.05, test_numb=None, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    edg_index_copy = copy.deepcopy(edg_index_all)
    if mask_sp == 'RNA':
        drug_set = set(edg_index_copy[0].tolist())
        if test_numb != None:
            test_numb = test_numb
        else:
            test_numb = int(len(drug_set) * test_rate)

        test_drug = np.random.choice(list(drug_set), size=test_numb, replace=False)
        test_idx = np.isin(edg_index_copy[0].cpu().numpy(), test_drug)
        train_idx = ~test_idx

    elif mask_sp == 'Disease':
        gene_set = set(edg_index_copy[1].tolist())
        if test_numb != None:
            test_numb = test_numb
        else:
            test_numb = int(len(gene_set) * test_rate)
        test_gene = np.random.choice(list(gene_set), size=test_numb, replace=False)
        test_idx = np.isin(edg_index_copy[1].cpu().numpy(), test_gene)
        train_idx = ~test_idx

    elif mask_sp == 'RNA-Disease':
        drug_set = set(edg_index_copy[0].tolist())
        if test_numb != None:
            test_numb = test_numb
        else:
            test_numb = int(len(drug_set) * test_rate/2)

        test_drug = np.random.choice(list(drug_set), size=test_numb, replace=False)
        test_idx1 = np.isin(edg_index_copy[0].cpu().numpy(), test_drug)

        gene_set = set(edg_index_copy[1].tolist())
        if test_numb != None:
            test_numb = test_numb
        else:
            test_numb = int(len(gene_set) * test_rate/2)

        test_gene = np.random.choice(list(gene_set), size=test_numb, replace=False)
        test_idx2 = np.isin(edg_index_copy[1].cpu().numpy(), test_gene)
        test_idx = test_idx1 | test_idx2
        train_idx = ~test_idx
    else:
        raise ValueError("Invalid mask_sp value. Should be 'RNA' or 'Disease' or 'RNA-Disease'.")

    return train_idx, test_idx


def calculate_regularization_loss(model, lamb_reg=0.001, exclude_bn=True, clip_threshold=1.0):
    """
    Calculate the L2 regularization loss of model parameters, with numerical stability processing.

    Parameters:
    - model: PyTorch model
    - lamb_reg: Regularization strength coefficient
    - exclude_bn: Whether to exclude BatchNorm layer parameters
    - clip_threshold: Parameter value clipping threshold to prevent gradient explosion
    """
    reg_loss = 0.0

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if exclude_bn:
        bn_params = set()
        for m in model.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                bn_params.update(m.parameters())

    for param in trainable_params:
        param_data = torch.clamp(param.data, -clip_threshold, clip_threshold)
        reg_loss += param_data.pow(2).mean()
    return reg_loss * lamb_reg


def add_label_noise(y_true, noise_level=0.0):
    """
    Add noise to the labels of the training set by randomly flipping 0s and 1s.

    Parameters:
    y_true -- Original label array (numpy array or similar array structure)
    train_idx -- Training set indices (numpy array or list)
    noise_level -- Noise intensity (float between 0.0 and 1.0)

    Returns:
    A copy of the labels with added noise
    """
    if noise_level == 0:
        return y_true
    train_labels = copy.deepcopy(y_true)

    n_samples = len(train_labels)
    n_flip = int(n_samples * noise_level)

    flip_indices = np.random.choice(np.arange(n_samples), size=n_flip, replace=False)
    for idx in flip_indices:
        if train_labels[idx] != 0:
            train_labels[idx] = 0
    return train_labels

def GIP_kernel(Asso_RNA_Dis):
    def getGosiR(Asso_RNA_Dis):
        # Calculate the r in the GOsi Kernel
        nc = Asso_RNA_Dis.shape[0]
        summ = 0
        for i in range(nc):
            x_norm = np.linalg.norm(Asso_RNA_Dis[i, :])
            x_norm = np.square(x_norm)
            summ = summ + x_norm
        r = summ / nc
        return r
    # The number of rows
    nc = Asso_RNA_Dis.shape[0]
    # Initialize a matrix as the result matrix
    matrix = np.zeros((nc, nc))
    # Calculate the denominator of the GIP formula
    r = getGosiR(Asso_RNA_Dis)
    # Calculate the result matrix
    for i in range(nc):
        for j in range(nc):
            # Calculate the numerator of the GIP formula
            temp_up = np.square(np.linalg.norm(Asso_RNA_Dis[i, :] - Asso_RNA_Dis[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e ** (-temp_up / r)
    return matrix



def functional_similarity(association_matrix):
    """
    Calculate the functional similarity matrix (cosine similarity).

    Parameters:
        association_matrix: Association matrix. For microRNA functional similarity, rows are microRNAs and columns are diseases;
                            For disease functional similarity, rows are diseases and columns are microRNAs.

    Returns:
        sim_matrix: Functional similarity matrix, where sim_matrix[i][j] represents the functional similarity between the i-th and j-th entities.
    """
    # Get the number of rows (number of entities)
    n = association_matrix.shape[0]
    # Initialize the similarity matrix
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        # Get the association spectrum vector of the i-th entity
        vec_i = association_matrix[i, :]
        # Calculate the L2 norm of this vector
        norm_i = np.linalg.norm(vec_i)

        for j in range(n):
            # Get the association spectrum vector of the j-th entity
            vec_j = association_matrix[j, :]
            # Calculate the L2 norm of this vector
            norm_j = np.linalg.norm(vec_j)

            # Calculate the dot product of the two vectors
            dot_product = np.dot(vec_i, vec_j)

            # Handle the case where the norm is 0 to avoid division by zero errors
            if norm_i == 0 or norm_j == 0:
                sim_matrix[i][j] = 0.0
            else:
                # Calculate the cosine similarity
                sim_matrix[i][j] = dot_product / (norm_i * norm_j)

    return sim_matrix

def cosine_similarity(features):
    """Calculate the cosine similarity matrix of the feature matrix."""
    # Calculate the vector norms
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    # Normalize the features
    normalized_features = features / norm
    # Calculate the cosine similarity matrix (matrix multiplication)
    similarity_matrix = np.dot(normalized_features, normalized_features.T)
    return similarity_matrix

class Data_paths:
    """Class to manage data paths for different datasets."""
    def __init__(self, DATAFIELD,  md='m_d.csv', mm_list=['mss.csv'], dd_list=['dss.csv']):
        base_path = f'{DATAFIELD}/'
        if md is None:
            md = 'm_d.csv'

        self.md = base_path + md
        self.mm = [base_path + mm for mm in mm_list]
        self.dd = [base_path + dd for dd in dd_list]


class PerformanceMonitor:
    """
    Code performance monitoring tool class.

    Functions:
    - Measure the running time of a code segment.
    - Measure the total memory usage of a Python program.
    - Measure the total GPU memory usage of a Python program (multiple metrics).
    - Format and print all monitoring metrics.
    """

    def __init__(self):
        # Time monitoring variables
        self._start_time: float = None
        self._end_time: float = None

        # Memory monitoring variables
        self._process = psutil.Process(os.getpid())

        # GPU memory monitoring variables (depends on PyTorch CUDA)
        self._has_cuda = torch.cuda.is_available() if 'torch' in globals() else False
        self._gpu_process = None

        # Initialize GPU process monitoring (if CUDA is available)
        if self._has_cuda:
            try:
                import pynvml
                pynvml.nvmlInit()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
                self._gpu_process = pynvml
            except:
                print("Warning: Unable to initialize pynvml, GPU process monitoring function will be unavailable.")

    def start(self) -> None:
        """Start monitoring performance metrics."""
        self._start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop monitoring performance metrics."""
        self._end_time = time.perf_counter()

    def get_runtime(self) -> float:
        """Return the running time of the code segment (in seconds)."""
        if self._start_time is None or self._end_time is None:
            raise RuntimeError("Please call the start() and stop() methods first.")
        return self._end_time - self._start_time

    def get_memory_usage(self) -> int:
        """Return the total memory usage of the Python program (in bytes)."""
        return self._process.memory_info().rss

    def get_gpu_memory_usage(self) -> dict:
        """
        Return multiple GPU memory usage metrics (in bytes, only valid in a CUDA environment).
        The returned dictionary contains:
        - allocated: GPU memory allocated by PyTorch.
        - cached: GPU memory cached by PyTorch.
        - total_process: Total GPU memory occupied by the current process (similar to nvidia-smi).
        - total_gpu: Total GPU memory of the entire GPU.
        - free_gpu: Free GPU memory of the entire GPU.
        """
        if not self._has_cuda:
            return {}

        torch.cuda.synchronize()

        metrics = {
            'allocated': torch.cuda.memory_allocated(),
            'cached': torch.cuda.memory_reserved(),
        }

        # Try to get more comprehensive GPU usage information
        if self._gpu_process:
            try:
                pid = os.getpid()
                procs = self._gpu_process.nvmlDeviceGetComputeRunningProcesses(self._gpu_handle)
                process_memory = 0

                for proc in procs:
                    if proc.pid == pid:
                        process_memory += proc.usedGpuMemory * 1024  # nvml returns in KB

                mem_info = self._gpu_process.nvmlDeviceGetMemoryInfo(self._gpu_handle)

                metrics.update({
                    'total_process': process_memory,
                    'total_gpu': mem_info.total,
                    'free_gpu': mem_info.free,
                })
            except:
                pass

        return metrics

    def print_metrics(self, label: str = "Program Performance") -> None:
        """
        Print all monitoring metrics.

        Parameters:
            label: Prefix for the output title.
        """
        # Format the time
        runtime = self.get_runtime()
        time_str = f"{runtime:.4f} seconds"

        # Format the memory
        memory = self.get_memory_usage()
        memory_str = self._format_bytes(memory)

        # Print the results
        print(f"[{label}]")
        print(f"  Running Time: {time_str}")
        print(f"  Total Program Memory Usage: {memory_str}")

        # Print GPU memory information
        if self._has_cuda:
            gpu_metrics = self.get_gpu_memory_usage()

            if 'total_process' in gpu_metrics:
                print(f"  Total Program GPU Memory Usage (Process Level): {self._format_bytes(gpu_metrics['total_process'])}")
            print(f"  GPU Memory Allocated by PyTorch: {self._format_bytes(gpu_metrics['allocated'])}")
            print(f"  GPU Memory Cached by PyTorch: {self._format_bytes(gpu_metrics['cached'])}")

            if 'total_gpu' in gpu_metrics:
                print(f"  Total GPU Memory: {self._format_bytes(gpu_metrics['total_gpu'])}")
                print(f"  Free GPU Memory: {self._format_bytes(gpu_metrics['free_gpu'])}")
        else:
            print("  GPU Memory Usage: Not supported (no CUDA device).")

        print()

    @staticmethod
    def _format_bytes(size: int) -> str:
        """Convert bytes to human-readable units (B/KB/MB/GB)."""
        if size == 0:
            return "0 B"

        units = ["B", "KB", "MB", "GB"]
        unit_index = 0
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        return f"{size:.3f} {units[unit_index]}"

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False