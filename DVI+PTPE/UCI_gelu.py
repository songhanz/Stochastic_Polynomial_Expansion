import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os, json 
import zipfile
import requests
from io import BytesIO
from pprint import pprint

import gaussian_variables as gv
import utils
import plot_utils as pu
import bayes_layers as bnn
from bayes_models import MLP, PointMLP, AdaptedMLP
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tqdm import tqdm
# from ucimlrepo import fetch_ucirepo 
np.random.seed(42)
tf.set_random_seed(42)


nonlinearity   = "gelu"

# Define dataset URLs and names
dataset_urls = {
    "conc": "https://archive.ics.uci.edu/static/public/165/concrete+compressive+strength.zip",
    "ener": "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip",
    "kin8": "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff", 
    "nava": "https://archive.ics.uci.edu/static/public/316/condition+based+maintenance+of+naval+propulsion+plants.zip", 
    "powe": "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip",
    "prot": "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
    "wine": "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
    "yach": "https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip"
}



# Function to load each dataset
def load_dataset(name, url):
    n_repeat    = 20  # default
    hidden_size = 50  # default
    if name == "conc":
        r = requests.get(url)
        z = zipfile.ZipFile(BytesIO(r.content))
        # Let's check actual file name in the zip
        print(f"Files in conc zip: {z.namelist()}")
        excel_data = BytesIO(z.read('Concrete_Data.xls'))
        data = pd.read_excel(excel_data)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
    
    elif name == "ener":
        r = requests.get(url)
        z = zipfile.ZipFile(BytesIO(r.content))
        print(f"Files in ener zip: {z.namelist()}")
        excel_data = BytesIO(z.read('ENB2012_data.xlsx'))
        data = pd.read_excel(excel_data, engine='openpyxl', sheet_name='Φύλλο1').dropna(how='all').dropna(axis=1)
        X = data.drop(columns=['Y1', 'Y2']).values
        y = data['Y1'].values  # Change to 'Y2' for the second target variable
    
    elif name == "kin8":
        # This one is directly a CSV, no zip
        data = pd.read_csv(url)
        X = data.drop(columns=['y']).values
        y = data['y'].values
    
    elif name == "nava":
        r = requests.get(url)
        z = zipfile.ZipFile(BytesIO(r.content))
        print(f"Files in nava zip: {z.namelist()}")
        # Let's verify the path
        data = pd.read_csv(BytesIO(z.read('UCI CBM Dataset/data.txt')), sep='\s+', header=None)
        X = data.drop(columns=[16, 17]).values
        y = data[17].values
    
    elif name == "powe":
        r = requests.get(url)
        z = zipfile.ZipFile(BytesIO(r.content))
        print(f"Files in powe zip: {z.namelist()}")
        excel_data = BytesIO(z.read('CCPP/Folds5x2_pp.xlsx'))
        data = pd.read_excel(excel_data, engine='openpyxl')
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
    
    elif name == "prot":
        # Direct CSV file
        data = pd.read_csv(url)
        X = data.drop(columns=['RMSD']).values
        y = data['RMSD'].values
        n_repeat = 5
        hidden_size = 100
    
    elif name == "wine":
        r = requests.get(url)
        z = zipfile.ZipFile(BytesIO(r.content))
        print(f"Files in wine zip: {z.namelist()}")
        # Need to use BytesIO for consistent handling
        data = pd.read_csv(BytesIO(z.read('winequality-red.csv')), sep=';')
        X = data.drop(columns=['quality']).values
        y = data['quality'].values
    
    elif name == "yach":
        r = requests.get(url)
        z = zipfile.ZipFile(BytesIO(r.content))
        print(f"Files in yach zip: {z.namelist()}")
        # Need to use BytesIO for consistent handling
        data = pd.read_csv(BytesIO(z.read('yacht_hydrodynamics.data')), sep='\s+', header=None)
        X = data.drop(columns=[6]).values
        y = data[6].values
    
    elif name == "year":
        r = requests.get(url)
        z = zipfile.ZipFile(BytesIO(r.content))
        print(f"Files in year zip: {z.namelist()}")
        # Need to use BytesIO for consistent handling
        data = pd.read_csv(BytesIO(z.read('YearPredictionMSD.txt')), sep=',', header=None)
        X = data.iloc[:, 1:]
        y = data.iloc[:, 0]
        n_repeat = 1
        hidden_size = 100
    
    else:
        raise ValueError("Dataset not recognized.")

    return X, y.squeeze(), n_repeat, hidden_size
    


def make_model(hypers):
    if hypers['method'].lower().strip() == 'bayes':
        MLP_factory = MLP
        prediction  = lambda y: tf.reshape(y.mean[:, 0], [-1])
        pred_logvar = lambda y: tf.reshape(y.mean[:, 1], [-1])
        loss        = bnn.regression_loss
    else:
        MLP_factory = PointMLP
        prediction  = lambda y: tf.reshape(y.mean[:, 0], [-1])
        loss        = bnn.point_regression_loss
        
    mlp             = MLP_factory(hypers['x_dim'], hypers['y_dim'], hypers)
    mlp             = AdaptedMLP(mlp)
    mlp.make_placeholders()
    ipt             = mlp.placeholders['ipt_mean']
    y               = mlp(ipt)

    target          = tf.placeholder(tf.float32, [None])
    mlp.placeholders['target']  = target
    
    global_step     = tf.Variable(0, trainable=False, name='global_step')
    loss, logprob, all_surprise = loss(y, target, mlp, hypers, global_step)

    '''
    NOTE:

    The logprob above is the loglikelihood under the Bayesian setting: 

    \ell_{\text{norm}}(x) = -\frac{1}{2} \left[ \log(2\pi) + \log\sigma^2 + \exp\left(-\log\sigma^2 + 0.5 \, s_u\right) \left( s_{mm} + \left(m - s_{ml} - x\right)^2 \right) \right],

    which considering the uncertainty of both mean estimation and the variance estimations.
    Although the Bayesian ver. is more rigorous,
    for the purpose of benchmarking, we report the loglikelihood under point estimate approach. 

    '''

    rmse            = tf.sqrt(tf.reduce_mean(tf.square(target - prediction(y))))

    vy              = tf.exp(pred_logvar(y)) 
    loglikelihood   = tf.reduce_mean(-0.5 * ( tf.log(2 * np.pi) 
                                              + pred_logvar(y) 
                                              + tf.square(target - prediction(y)) / vy)
                                    )


    return {
        'model': mlp,
        'metrics': {
            'accuracy'      : rmse,
            'loss'          : loss,
            'logprob'       : logprob, #logprob - np.log(y_std),
            'loglikelihood' : loglikelihood,
            'all_surprise'  : all_surprise
            },
        'global_step': global_step}


def parse_data_hyper(dataset, url, n_hidden_layer):

    X, y, n_repeat, hidden_size = load_dataset(dataset, url)

    datapoint = X.shape[0]
    dimension = X.shape[1]

    print(">>>>>>>>>>>>>")
    # print(f"{dataset} | {datapoint} datapoints | {dimension} dim")
    print("{} | {} datapoints | {} dim".format(dataset, datapoint, dimension))
    print(">>>>>>>>>>>>>")

    hypers = {
        "x_dim": X.shape[1],
        "y_dim": 2,
        "hidden_dims": [hidden_size] * n_hidden_layer,
        "nonlinearity": nonlinearity,
        "method": "bayes",
        "style": "heteroskedastic",
        "prior_type": ["gaussian", "xavier", "xavier"], 
        "n_epochs": 2000,
        "early_stop_patience": 20,
        "batch_size": 64, #round(datapoint ** 0.5), #128, #,
        "learning_rate": 0.005,
        "lambda": 1.0,
        "warmup_updates": {'lambda': 14000.0},
        "anneal_updates": {'lambda': 1000.0},
        "optimizer": "adam",
        "gradient_clip": 0.1,
        "data_fraction": 1.0,
        "sections_to_run": ["train", "valid", "test"]
    }

    pprint(hypers)

    return X, y, hypers, n_repeat


def random_split(X, y, seed):
    # Split the data into training, validation, and test sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    # 80% training, 10% validation, 10% testing
    # X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=seed)
    # X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)

    X_temp, X_test, y_temp, y_test     = train_test_split(X, y, test_size=0.1, random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.1, random_state=seed)

    # X_temp, X_test, y_temp, y_test     = train_test_split(X, y, test_size=0.1)
    # X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.1)

    # normalize input
    mu              = np.mean(X_train, axis=0)
    sigma           = np.std(X_train, axis=0, ddof=0) # ddof=1 for sample variance, default is 0 for population
    sigma[sigma==0] = 1
    
    X_train         = (X_train - mu) / sigma
    X_valid         = (X_valid - mu) / sigma
    X_test          = (X_test  - mu) / sigma 

    data            = [
                       [X_train, y_train],
                       [X_valid, y_valid],
                       [X_test, y_test]
                      ]

    return data


def update_adapter(data, hypers):
    # this adapter serves as input normalization and output denormalization layers
    
    # X_train      = data[0][0]
    y_train      = data[0][1]

    # mu_X         = np.mean(X_train, axis=0)
    # sigma_X      = np.std(X_train, axis=0, ddof=0) # ddof=1 for sample variance, default is 0 for population

    mu_y         = np.mean(y_train)
    sigma2_y     = np.var(y_train, ddof=0) # ddof=1 for sample variance, default is 0 for population

    hypers["adapter"] = {
        'in' : {"scale": [[1.0]], "shift": [[0.0]]},
        'out': {"scale": [[sigma2_y, 1.0]], "shift": [[mu_y, np.log(sigma2_y)]]}        # note that the output are mean and log_var
    }

    return hypers




def store_history(storage_dataframe, accuracies, epoch_summary):

    # Extract metrics
    train_rmse    = accuracies['train']
    valid_rmse    = accuracies['valid']
    test_rmse     = accuracies['test']
    
    train_logprob = epoch_summary['train'][-1][-1]['running_logprob']
    valid_logprob = epoch_summary['valid'][-1][-1]['running_logprob']
    test_logprob  = epoch_summary['test'][-1][-1]['running_logprob']
    
    # train_KL      = epoch_summary['train'][-1][-1]['all_surprise'] / hypers['dataset_size']
    # valid_KL      = epoch_summary['valid'][-1][-1]['all_surprise'] / hypers['dataset_size']
    # test_KL       = epoch_summary['test'][-1][-1]['all_surprise'] / hypers['dataset_size']

    # Update the DataFrame
    # storage_dataframe = pd.concat([storage_dataframe, pd.DataFrame([{
    #     'epoch': epoch, 
    #     'train_rmse': train_rmse, 'train_logprob': train_logprob, 'train_KL': train_KL,
    #     'valid_rmse': valid_rmse, 'valid_logprob': valid_logprob, 'valid_KL': valid_KL,
    #     'test_rmse': test_rmse, 'test_logprob': test_logprob, 'test_KL': test_KL
    # }])], ignore_index=True)

    train_loglikelihood      = epoch_summary['train'][-1][-1]['running_loglikelihood']
    valid_loglikelihood      = epoch_summary['valid'][-1][-1]['running_loglikelihood']
    test_loglikelihood       = epoch_summary['test'][-1][-1]['running_loglikelihood']

    storage_dataframe = pd.concat([storage_dataframe, pd.DataFrame([{
        'epoch': epoch, 
        'train_rmse': train_rmse, 'train_logprob': train_logprob, 'train_loglikelihood': train_loglikelihood,
        'valid_rmse': valid_rmse, 'valid_logprob': valid_logprob, 'valid_loglikelihood': valid_loglikelihood,
        'test_rmse': test_rmse, 'test_logprob': test_logprob, 'test_loglikelihood': test_loglikelihood
    }])], ignore_index=True)

    return storage_dataframe




progress_bar   = tqdm(dataset_urls.items(), position=0)
for dataset, url in progress_bar:
    # progress_bar.set_description(f"Processing {dataset}")
    progress_bar.set_description("Processing {}".format(dataset))

    for n_hidden_layer in range(1, 2):

        X, y, hypers, n_repeat = parse_data_hyper(dataset, url, n_hidden_layer)

        save_dir  = 'UCI_results_sgd/' + dataset + '_' + nonlinearity + '_' + str(n_hidden_layer) + 'layer_adapter/'
       
        os.makedirs(save_dir, exist_ok=True)
        file_name = save_dir + "test_perf.txt"
        if os.path.exists(file_name):
            os.remove(file_name)

        # save hyperparameters
        with open(save_dir + "hyperparameters.txt", "w") as file:
            json.dump(hypers, file)

        # repeat for error bar
        run_index_desc = "Runs for {}".format(dataset)
        for run_index in tqdm(range(n_repeat), desc=run_index_desc, position=1, leave=False):
        # for run_index in tqdm(range(n_repeat), desc=f"Runs for {dataset}", position=1, leave=False):
            
            print("\n")
            tf.reset_default_graph()

            os.makedirs(save_dir + str(run_index), exist_ok=True)

            data                    = random_split(X, y, run_index)

            hypers                  = update_adapter(data, hypers)

            run_id                  = utils.start_run()

            restricted_training_set = utils.restrict_dataset_size(data[0], hypers['data_fraction'])
            hypers['dataset_size']  = len(restricted_training_set[0])

            # Initialize the DataFrame for storing history
            # history = pd.DataFrame(columns=['epoch', 'train_rmse', 'train_logprob', 'train_KL', 
            #                                 'valid_rmse', 'valid_logprob', 'valid_KL', 
            #                                 'test_rmse', 'test_logprob', 'test_KL'])
            history = pd.DataFrame(columns=['epoch', 'train_rmse', 'train_logprob', 'train_loglikelihood', 
                                            'valid_rmse', 'valid_logprob', 'valid_loglikelihood', 
                                            'test_rmse', 'test_logprob', 'test_loglikelihood'])

            device_id             = 0
            device_string         = utils.get_device_string(device_id)
            with tf.device(device_string):

                model_and_metrics = make_model(hypers)
                train_op          = utils.make_optimizer(model_and_metrics, hypers)
                sess              = utils.get_session()
                
                # save best model
                # best model is defined as the one with best validation rmse
                saver             = tf.train.Saver()
                best_test_rmse    = 0
                best_test_logprob = 0
                best_test_loglikelihood      = 0

                all_summaries     = []

                # Initialize early stopping variables
                # stop training if validation logprob not increasing for 5 epochs
                patience          = hypers["early_stop_patience"]
                best_validation   = -float('inf')
                best_model_epoch  = 0
                patience_counter  = 0
                
                for epoch in range(hypers['n_epochs']):
                    if epoch < 500:
                        # do not trigger early stopping in the first 50 epochs
                        patience_counter = 0

                    verbose = (epoch % 20 == 0)
                    if verbose:
                        n   = hypers['n_epochs']
                        # print(f"Epoch {epoch}/{n}:")
                        print("Epoch {}/{}:".format(epoch, n))

                    epoch_summary, accuracies = utils.train_valid_test(
                        {
                            'train': restricted_training_set,
                            'valid': data[1],
                            'test' : data[2]
                        },
                        sess, model_and_metrics, train_op, hypers, verbose)

                    history = store_history(history, accuracies, epoch_summary)

                    # Early stopping check
                    current_validation    = epoch_summary['valid'][-1][-1]['running_loglikelihood']
                    # current_validation    = accuracies['valid']
                    if current_validation > best_validation:
                        best_validation   = current_validation
                        best_model_epoch  = epoch
                        # Save the best model
                        # os.makedirs(f'{save_dir}{run_index}', exist_ok=True)
                        # saver.save(sess, f'{save_dir}{run_index}/best_model_checkpoint')
                        os.makedirs(save_dir + str(run_index), exist_ok=True)
                        saver.save(sess, save_dir + str(run_index) + '/best_model_checkpoint')
                        best_test_rmse    = accuracies['test']
                        best_test_logprob = epoch_summary['test'][-1][-1]['running_logprob']
                        # best_test_KL      = epoch_summary['test'][-1][-1]['all_surprise']/hypers['dataset_size']
                        best_test_loglikelihood = epoch_summary['test'][-1][-1]['running_loglikelihood']

                        patience_counter  = 0
                    else:
                        patience_counter += 1
                    
                    # Check if we should stop training
                    if patience_counter  >= patience:
                        print("Early stopping triggered! No improvement for {} epochs".format(patience))
                        break

                print("Best validation loglikelihood was {:.4f} at epoch {}".format(best_validation,best_model_epoch))
                print("Final performance")
                print(' %s RMSE = %.4f | loglikelihood (Bayes) = %.4f | loglikelihood (point) = %.4f' % 
                        ('test', best_test_rmse, best_test_logprob, best_test_loglikelihood))
            
            # Close the session after each run
            sess.close()

            history.to_csv(save_dir + str(run_index) + '/training_history.csv', index=False)


            # Write RMSE and loglikelihood to a file
            file_name = save_dir + "test_perf.txt"
            # Check if the file exists, create it if it does not
            if not os.path.exists(file_name):
                with open(file_name, "w") as file:
                    pass

            with open(file_name, "a") as file:
                # file.write(f"{best_test_rmse:.4f} \t {best_test_logprob:.4f} \n")
                file.write("{:.4f} \t {:.4f} \n".format(best_test_rmse, best_test_loglikelihood))

            print("RMSE and loglikelihood saved to " + file_name)

        
