from utils import *
from preprocess import BioData
from train_model import *
import setproctitle
import warnings
import multiprocessing

warnings.filterwarnings("ignore")
setproctitle.setproctitle("GPASS3")
use_multiprocessing = False

def mul_func(data_tuple):
    """Function to handle model training and evaluation for a single configuration."""
    start_time = datetime.datetime.now()
    warnings.filterwarnings('ignore', message='TypedStorage is deprecated.')
    data_name, params, biodata = data_tuple
    biodata = biodata.copy()
    params = params.copy()
    biodata.config = params
    biodata.resplit_Gdata()
    param_search = params.param_search
    set_seed(521)
    key_file = ''.join(f"{key}{getattr(params, key)}" for key in param_search.keys())
    save_file = f"{data_name}/epoch{params.epochs}_{key_file}"
    print(f'-----Starting task {save_file}-----')
    params.save_file = params.save_file + key_file
    results_list = repeat_train(biodata)  # Cross-validation training
    print(f'-----Task {save_file} completed-----')
    print_execution_time(start_time)

default_param_search = {
    'edge_emb_layer': [2,],  # 2[1, 2, 3, 4, 5, 6, 7]
    'gnn_layer': [2,],  # 4[2, 4, 6, 8]
    'n_hidden': [32,],  # ,  # emb_number # 16[4, 8, 16, 32]
    'ncf_hidden': [128,], # [4, 8, 16, 32]
}


if __name__ == '__main__':
    # data_names = ['HMDDv4.0','miRBase', 'DTI']
    data_names = ['HMDDv4.0']
    # print(best_param_search)
    for data_id in [0]:
        params_list = []
        # Load the dataset for the specified data ID
        datapath = Data_paths(f'./data/{data_names[data_id]}/')
        device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
        biodata = BioData(datapath.md,
                          mss_list=datapath.mm,
                          dss_list=datapath.dd,
                          device=device,
                          data_name=data_names[data_id],
                          sep=r'[\t ]')
        params_init = biodata.config
        params_init.epochs = 1000
        params_init.repeat = 10
        params_init.test_size = 0.2

        param_generator = set_attr(params_init, default_param_search)
        # Generate configurations from the parameter search grid
        for params in param_generator:
            params_list.append((data_names[data_id], copy.deepcopy(params), biodata))
            print(f"Configuration set {len(params_list)} prepared...")

        if use_multiprocessing:
            multiprocessing.set_start_method('spawn')
            with multiprocessing.Pool(processes=10) as pool: # min(min(len(params_list), os.cpu_count()-2), 20))
                pool.map(mul_func, params_list)
        else:
            # Process each configuration sequentially without multiprocessing
            for config in params_list:
                mul_func(config)
        print("All tasks completed.")
