import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score
from utils import *
from model import GPASS
from preprocess import BioData


def train_model(biodata, train_idx=None, test_idx=None, save_file=None, Val_test=True):
    """
    Function to train the model based on provided indices and parameters.

    Args:
        biodata: BioData object containing data and configuration.
        train_idx: Indices for training data.
        test_idx: Indices for test data.
        save_file: Path to save the model and results.
        Val_test: Flag to indicate if validation/test split is needed.
    """
    graph = biodata.graph.clone()
    param = biodata.config
    y = biodata.edge_and_label
    monitor = PerformanceMonitor()
    if hasattr(param, 'edge_number') and param.edge_number is not None:
        # Get the shape of the original matrix
        shape = biodata.adj_matrix.shape
        total_elements = np.prod(shape)
        # Ensure edge_number does not exceed the total number of matrix elements
        edge_number = min(param.edge_number, total_elements)
        # Create a matrix filled with zeros
        adj_matrix = np.zeros(shape, dtype=int)
        # Randomly select positions to set to 1
        indices = np.random.choice(total_elements, edge_number, replace=False)
        # Set the selected positions to 1
        np.put(adj_matrix, indices, 1)
        biodata_test = BioData(adj_matrix,
                               mss_list=biodata.mss_matrix_list,
                               dss_list=biodata.dss_matrix_list,
                               device=biodata.device,
                               config=biodata.config)
        biodata = biodata_test
        y = biodata.edge_and_label
        train_idx = None
        test_idx = None

    if train_idx is None:
        train_idx = np.arange(y['y'].shape[0])

    if Val_test:  # Randomly split a validation set during test training
        # Randomly select 0.1 from train_idx as test data, and the rest as training data
        num_samples = len(train_idx)
        num_test = int(num_samples * param.val_test_size)  # Calculate the size of the test set (5%)
        # Randomly shuffle the indices
        shuffled_indices = torch.randperm(num_samples)
        # Split the indices
        val_test_idx = train_idx[shuffled_indices[:num_test]]
        train_idx = train_idx[shuffled_indices[num_test:]]
        if test_idx is None:
            test_idx = val_test_idx
            _test_idx = False

    if test_idx is None:  # Recalculate Gaussian similarity if there is a mask during testing
        src_test, tgt_test = y['y_edge'][0][test_idx], y['y_edge'][1][test_idx]
        mask_adj_matrix = biodata.adj_matrix.copy()
        mask_adj_matrix[src_test, tgt_test - mask_adj_matrix.shape[0]] = 0
        mg_sim = GIP_kernel(mask_adj_matrix)
        dg_sim = GIP_kernel(mask_adj_matrix.T)
        mirna_func_sim = functional_similarity(mask_adj_matrix)
        disease_func_sim = functional_similarity(mask_adj_matrix.T)
        biodata_test = BioData(mask_adj_matrix,
                               mss_list=biodata.mss_matrix_list[:1] + [mirna_func_sim, mg_sim],
                               dss_list=biodata.dss_matrix_list[:1] + [disease_func_sim, dg_sim],
                               device=biodata.device,
                               config=biodata.config)
        biodata = biodata_test

    gmss = biodata.mss_graph
    gdss = biodata.dss_graph
    device = param.device
    _test_idx = True
    model = GPASS(param).to(device)
    if save_file is None:
        save_file = param.save_file
    if not os.path.exists(save_file):
        os.makedirs(save_file, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=param.lr, weight_decay=param.weight_decay)  # 0.0002

    src_train, tgt_train = y['y_edge'][0][train_idx], y['y_edge'][1][train_idx]
    src_test, tgt_test = y['y_edge'][0][test_idx], y['y_edge'][1][test_idx]
    true_src = y['y_edge'][0][test_idx[(y['y'][test_idx] == 1).reshape(-1)]]
    true_tgt = y['y_edge'][1][test_idx[(y['y'][test_idx] == 1).reshape(-1)]]

    # Mask the data in the test set
    for _src, _tgt in zip(true_src, true_tgt):
        graph.remove_edges((_src, _tgt))
    if _test_idx:
        val_src_test, val_tgt_test = y['y_edge'][0][val_test_idx], y['y_edge'][1][val_test_idx]
        val_true_src = y['y_edge'][0][val_test_idx[(y['y'][val_test_idx] == 1).reshape(-1)]]
        val_true_tgt = y['y_edge'][1][val_test_idx[(y['y'][val_test_idx] == 1).reshape(-1)]]
        for _src, _tgt in zip(val_true_src, val_true_tgt):
            graph.remove_edges((_src, _tgt))
    else:
        val_src_test, val_tgt_test = src_test, tgt_test
        val_true_src, val_true_tgt = true_src, true_tgt
    y_train_true = add_label_noise(y['y'][train_idx].reshape(-1, ), noise_level=param.noise_level)
    y_train_true = torch.tensor(y_train_true)
    auc_list = []
    start_time = datetime.datetime.now()
    _patience = 0
    monitor.start()
    param.print_epoch = min(param.print_epoch, param.epochs)
    for epoch in range(0, param.epochs + 1):
        optimizer.zero_grad()
        rep = model(graph, gmss, gdss).to(device)
        preds = model.predict(rep[src_train], rep[tgt_train]).to(device)
        loss = model.adaptive_mloss(preds, y_train_true.to(device))
        if model.init_parameters is False:
            model.reset_parameters()
            continue
        loss.backward()
        optimizer.step()
        if epoch > 100:
            model.Dynamic_train = False
        if epoch % param.print_epoch == 0 and not test_idx is None:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            model.eval()
            with torch.no_grad():
                rep = model(graph, gmss, gdss)
                val_preds = model.predict(rep[val_src_test], rep[val_tgt_test])
                val_out_pred = val_preds.to('cpu').detach().numpy()
                val_y_true = y['y'][val_test_idx].to('cpu').detach().numpy()
                auc = roc_auc_score(val_y_true, val_out_pred)
                print('Val AUC:', auc)
                if auc > model.best_auc:
                    _patience = 0
                    model.best_auc = auc
                    model.best_epoch = epoch
                    model.save_model(path=save_file + '/best_model.pth')
                else:
                    _patience += 1
                if _patience >= param.patience:
                    break
                preds = model.predict(rep[src_test], rep[tgt_test])
                out_pred = preds.to('cpu').detach().numpy()
                y_true = y['y'][test_idx].to('cpu').detach().numpy()
                print(out_pred[:10])
                auc_idx, auc_name = get_metrics(y_true, out_pred)
                auc_idx.extend(param.search_args['arg_value'])
                auc_idx.append(loss.item())
                auc_idx.append(epoch)
                auc_list.append(auc_idx)
                print_execution_time(start_time, epoch)
            model.train()
    monitor.stop()
    monitor.print_metrics(f'Number of edges: {biodata.adj_matrix.sum()}')
    auc_name.extend(param.search_args['arg_name'])
    auc_name += ['loss', 'epoch']
    results = pd.DataFrame(np.array(auc_list,dtype=object), columns=auc_name)
    results.to_feather(path=save_file + '/results.feather')
    model.load_model(path=save_file + '/best_model.pth')
    model.Overall_Refactoring_ASS_Embedding(biodata)

    if biodata.config.save_ASS_Embedding == True:
        biodata.save_ASS_Embedding(path=save_file + '/ASS_Embedding.emb')
    return results, model


def cold_repeat_train(biodata):
    """
    Cross-validation training setup for cold start scenario.

    Args:
        biodata: BioData object containing data and configuration.

    Returns:
        List of training results.
    """
    y = biodata.edge_and_label
    param = biodata.config
    repeat = param.repeat
    k_number = 1
    results_list = []
    for ii in range(repeat):
        train_idx, test_idx = mask_func(y['y_edge'], mask_sp=param.mask_sp)
        print(f'Running repeat {ii + 1} of {repeat}...')
        train_idx = np.arange(y['y'].shape[0])[train_idx]
        test_idx = np.arange(y['y'].shape[0])[test_idx]
        results, *_ = train_model(biodata, train_idx, test_idx, save_file=param.save_file + f'/repeat{ii}/')
        if not os.path.exists(param.save_file + f'/repeat{ii}/'):
            os.makedirs(param.save_file + f'/repeat{ii}/', exist_ok=True)
        results_list.append(results)
    return results_list


def repeat_train(biodata):
    """
    Cross-validation training setup.

    Args:
        biodata: BioData object containing data and configuration.

    Returns:
        List of training results.
    """
    y = biodata.edge_and_label
    param = biodata.config
    if not param.mask_sp is None:
        return cold_repeat_train(biodata)
    k_fold = param.kfold
    repeat = param.repeat

    k_number = 1
    results_list = []
    train_idx = np.arange(y['y'].shape[0])
    num_samples = len(train_idx)
    num_test = int(num_samples * param.test_size)  # Calculate the size of the test set (5%)
    # Randomly shuffle the indices
    shuffled_indices = torch.randperm(num_samples)
    # Split the indices
    test_idx = train_idx[shuffled_indices[:num_test]]
    train_idx = train_idx[shuffled_indices[num_test:]]
    for ii in range(repeat):
        print(f'Running repeat {len(results_list) + 1} of {repeat}...')
        auc_idx, auc_name, *_ = train_model(biodata, train_idx, test_idx,
                                            save_file=param.save_file + f'/repeat{ii}/')
        if not os.path.exists(param.save_file + f'/repeat{ii}/'):
            os.makedirs(param.save_file + f'/repeat{ii}/', exist_ok=True)
        k_number += 1
        results_list.append(auc_idx)
    return results_list


def CV_train(biodata):
    """
    Cross-validation training setup using KFold.

    Args:
        biodata: BioData object containing data and configuration.

    Returns:
        List of training results.
    """
    y = biodata.edge_and_label
    param = biodata.config
    k_fold = param.kfold
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=param.globel_random)
    results_list = []

    k_number = 0
    for train_idx, test_idx in kf.split(np.arange(y['y'].shape[0])):
        print(f'Running fold {len(results_list) + 1} of {k_fold}...')

        auc_idx, auc_name, *_ = train_model(biodata, train_idx, test_idx,
                                            save_file=param.save_file + f'/KFold{k_number}/')
        if not os.path.exists(param.save_file + f'/KFold{k_number}/'):
            os.makedirs(param.save_file + f'/KFold{k_number}/', exist_ok=True)
        k_number += 1
        results_list.append(auc_idx)
    return results_list
