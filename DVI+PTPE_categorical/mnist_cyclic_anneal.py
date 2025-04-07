import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os, json 
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import brier_score_loss


import gaussian_variables as gv
import utils
import plot_utils as pu
import bayes_layers as bnn
from bayes_layers import logsoftmax
from bayes_models import MLP, PointMLP, AdaptedMLP


def load_mnist():
    """Load and preprocess the MNIST dataset."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Reshape and normalize
    x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28*28).astype('float32') / 255.0
    
    # One-hot encode the labels
    y_train_oh = tf.keras.utils.to_categorical(y_train, 10)
    y_test_oh = tf.keras.utils.to_categorical(y_test, 10)
    
    print(f"Loaded {len(x_train)} training and {len(x_test)} test MNIST images.")
    return (x_train, y_train, y_train_oh), (x_test, y_test, y_test_oh)


def make_model(hypers):
    if hypers['method'].lower().strip() == 'bayes':
        MLP_factory = MLP
        loss        = bnn.categorical_loss
    else:
        MLP_factory = PointMLP
        loss        = bnn.point_catagorical_loss
        
    mlp             = MLP_factory(hypers['x_dim'], hypers['y_dim'], hypers)
    mlp.make_placeholders()
    ipt             = mlp.placeholders['ipt_mean']
    y               = mlp(ipt)

    target          = tf.placeholder(tf.float32, [None, 10])  # Changed for one-hot labels
    mlp.placeholders['target'] = target
    
    global_step     = tf.Variable(0, trainable=False, name='global_step')
    loss_val, logprob, all_surprise = loss(y, target, mlp, hypers, global_step)

    probs           = tf.exp(logsoftmax(y))

    # Calculate Brier score for classification
    target_flat     = tf.reshape(target, [-1])
    probs_flat      = tf.reshape(probs, [-1])
    brier_score     = tf.reduce_mean(tf.square(target_flat - probs_flat))
    
    # Also track accuracy for reference
    predicted_labels= tf.argmax(probs, axis=1)
    true_labels     = tf.argmax(target, axis=1)
    accuracy        = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, true_labels), tf.float32))


    return {
        'model': mlp,
        'metrics': {
            'accuracy'     : accuracy,
            'brier'        : brier_score,
            'loss'         : loss_val,
            'logprob'      : logprob,
            'all_surprise' : all_surprise
        },
        'global_step': global_step
    }


def parse_data_hyper(n_hidden_layer=2, hidden_size=512):
    # Load MNIST data
    (x_train, y_train, y_train_oh), (x_test, y_test, y_test_oh) = load_mnist()
    
    X = x_train
    y = y_train_oh
    
    datapoint = X.shape[0]
    dimension = X.shape[1]

    print(">>>>>>>>>>>>>")
    print("MNIST | {} datapoints | {} dim".format(datapoint, dimension))
    print(">>>>>>>>>>>>>")

    hypers = {
        "x_dim": X.shape[1],        # 784 for flattened MNIST
        "y_dim": 10,                # 10 classes for MNIST
        "hidden_dims": [hidden_size] * n_hidden_layer,
        "nonlinearity": "relu",
        "method": "bayes",
        "prior_type": ["gaussian", "he", "he"], 
        "n_epochs": 50,            
        "early_stop_patience": 10,
        "batch_size": 256,          # Increased batch size from original 128
        "learning_rate": 0.001,
        "lambda": 0.5,              # Default lambda value
        "warmup_updates": {'lambda': 14000.0},
        "anneal_updates": {'lambda': 1000.0},
        "optimizer": "adam",
        "gradient_clip": 0.1,
        "data_fraction": 1.0,
        "sections_to_run": ["train", "valid", "test"],
        
        # Cyclic annealing parameters
        "use_cyclic_annealing": True,
        "max_lambda": 2.0,          # Maximum lambda value in cycles
        "min_lambda": 0.5,          # Minimum lambda value in cycles
        "num_cycles": 4,            # Number of cycles to complete during training
        "cycle_ratio": 0.5          # Fraction of cycle spent increasing lambda
    }

    print("Hyperparameters:")
    for key, value in hypers.items():
        print(f"  {key}: {value}")

    return X, y, hypers


def random_split(X, y, seed):
    # Split data into training, validation, and test sets
    from sklearn.model_selection import train_test_split
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.1, random_state=seed)

    # No normalization needed as we've already normalized the data during loading
    
    data = [
        [X_train, y_train],
        [X_valid, y_valid],
        [X_test, y_test]
    ]

    return data


def store_history(storage_dataframe, accuracies, epoch_summary, epoch, lambda_value=None):
    # Extract metrics
    train_acc    = accuracies['train']
    valid_acc    = accuracies['valid']
    test_acc     = accuracies['test']
    
    train_brier  = epoch_summary['train'][-1][-1]['running_brier']
    valid_brier  = epoch_summary['valid'][-1][-1]['running_brier']
    test_brier   = epoch_summary['test'][-1][-1]['running_brier']
    
    train_logprob = epoch_summary['train'][-1][-1]['running_logprob']
    valid_logprob = epoch_summary['valid'][-1][-1]['running_logprob']
    test_logprob  = epoch_summary['test'][-1][-1]['running_logprob']
    
    data = {
        'epoch': epoch, 
        'train_acc': train_acc, 'train_brier': train_brier, 'train_logprob': train_logprob,
        'valid_acc': valid_acc, 'valid_brier': valid_brier, 'valid_logprob': valid_logprob, 
        'test_acc': test_acc, 'test_brier': test_brier, 'test_logprob': test_logprob
    }
    
    # Add lambda value if provided
    if lambda_value is not None:
        data['lambda'] = lambda_value
    
    storage_dataframe = pd.concat([storage_dataframe, pd.DataFrame([data])], ignore_index=True)

    return storage_dataframe


def cyclic_lambda_schedule(epoch, total_epochs, num_cycles, min_lambda, max_lambda, cycle_ratio=0.5):
    """
    Implements a cyclical annealing schedule for the KL weight (lambda).
    
    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs for training
        num_cycles: Number of cycles to complete during training
        min_lambda: Minimum lambda value (usually 0)
        max_lambda: Maximum lambda value (usually 1)
        cycle_ratio: Fraction of cycle spent increasing lambda (default: 0.5)
        
    Returns:
        The lambda value for the current epoch
    """
    # Calculate cycle length in epochs
    cycle_length = total_epochs / num_cycles
    
    # Determine which cycle we're in and the position within that cycle
    current_cycle = int(epoch / cycle_length)
    cycle_position = (epoch % cycle_length) / cycle_length
    
    # Implement the cyclical schedule
    if cycle_position < cycle_ratio:
        # Increasing phase - scale from min to max
        normalized_position = cycle_position / cycle_ratio
        return min_lambda + (max_lambda - min_lambda) * normalized_position
    else:
        # Decreasing phase - scale from max to min
        normalized_position = (cycle_position - cycle_ratio) / (1 - cycle_ratio)
        return max_lambda - (max_lambda - min_lambda) * normalized_position


def create_dvi_model():
    """Creates and trains a Bayesian Neural Network with cyclic lambda annealing."""
    tf.reset_default_graph()
    
    X, y, hypers = parse_data_hyper(n_hidden_layer=2, hidden_size=512)
    data = random_split(X, y, seed=42)
    
    run_id = utils.start_run()
    restricted_training_set = utils.restrict_dataset_size(data[0], hypers['data_fraction'])
    hypers['dataset_size'] = len(restricted_training_set[0])
    
    # Save original lambda for resetting later
    original_lambda = hypers['lambda']
    
    # Initialize history DataFrame
    history_columns = ['epoch', 'train_acc', 'train_brier', 'train_logprob', 
                     'valid_acc', 'valid_brier', 'valid_logprob', 
                     'test_acc', 'test_brier', 'test_logprob', 'lambda']
    history = pd.DataFrame(columns=history_columns)
    
    # Create model once
    device_id = 0
    device_string = utils.get_device_string(device_id)
    with tf.device(device_string):
        # Create the model with original lambda
        model_and_metrics = make_model(hypers)
        train_op = utils.make_optimizer(model_and_metrics, hypers)
        sess = utils.get_session()
        
        # Saver for best model
        saver = tf.train.Saver()
        best_validation = -float('inf')
        best_model_epoch = 0
        patience_counter = 0
        
        # Extract cyclic annealing parameters
        use_cyclic = hypers.get('use_cyclic_annealing', True)
        max_lambda = hypers.get('max_lambda', 1.0)
        min_lambda = hypers.get('min_lambda', 0.0)
        num_cycles = hypers.get('num_cycles', 4)
        cycle_ratio = hypers.get('cycle_ratio', 0.5)
        total_epochs = hypers['n_epochs']
        
        # Train for specified number of epochs
        for epoch in tqdm(range(total_epochs)):
            if epoch < 10:  # Don't trigger early stopping in first 10 epochs
                patience_counter = 0
                
            verbose = (epoch % 1 == 0)
            
            # Update lambda in hypers for this epoch using cyclic schedule
            if use_cyclic:
                current_lambda = cyclic_lambda_schedule(
                    epoch, total_epochs, num_cycles, min_lambda, max_lambda, cycle_ratio
                )
                hypers['lambda'] = current_lambda
                
                if verbose:
                    print(f"Epoch {epoch}/{total_epochs}: (lambda = {current_lambda:.4f})")
            
            # Use original utils.train_valid_test - no modifications needed
            epoch_summary, accuracies = utils.train_valid_test(
                {'train': restricted_training_set, 'valid': data[1], 'test': data[2]},
                sess, model_and_metrics, train_op, hypers, verbose)
            
            # Store history with current lambda
            history = store_history(history, accuracies, epoch_summary, epoch, hypers['lambda'])
            
            # Early stopping check
            current_validation = epoch_summary['valid'][-1][-1]['running_logprob']
            if current_validation > best_validation:
                best_validation = current_validation
                best_model_epoch = epoch
                patience_counter = 0
                
                # Save model checkpoint
                save_dir = 'mnist_dvi_model_cyclic_annealing3/'
                os.makedirs(save_dir, exist_ok=True)
                saver.save(sess, save_dir + 'best_model_checkpoint')
            else:
                patience_counter += 1
                
            if patience_counter >= hypers['early_stop_patience']:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break
        
        # Restore original lambda in hypers
        hypers['lambda'] = original_lambda
        
        # Print final results
        print("\nTraining completed!")
        print(f"Best validation logprob was {best_validation:.4f} at epoch {best_model_epoch}")
        print("Final performance:")
        for section in ['train', 'valid', 'test']:
            print(f"  {section}: Accuracy = {accuracies[section]:.4f}, Brier = {epoch_summary[section][-1][-1]['running_brier']:.4f}, "
                  f"LogProb = {epoch_summary[section][-1][-1]['running_logprob']:.4f} ")
        
        # Save history to CSV
        results_dir = 'mnist_dvi_results_cyclic_annealing3'
        os.makedirs(results_dir, exist_ok=True)
        history.to_csv(f'{results_dir}/training_history.csv', index=False)
        
        # Create a plot of the lambda schedule
        if use_cyclic:
            plt.figure(figsize=(10, 5))
            plt.plot(history['epoch'], history['lambda'])
            plt.title('Cyclic Lambda Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Lambda Value')
            plt.grid(True)
            plt.savefig(f'{results_dir}/cyclic_lambda_schedule.png')
            
            # Create a plot showing training accuracy vs lambda
            plt.figure(figsize=(10, 5))
            plt.scatter(history['lambda'], history['train_acc'])
            plt.title('Training Accuracy vs. Lambda')
            plt.xlabel('Lambda Value')
            plt.ylabel('Training Accuracy')
            plt.grid(True)
            plt.savefig(f'{results_dir}/accuracy_vs_lambda.png')
            
            # Create a plot showing train/valid/test metrics over epochs
            plt.figure(figsize=(15, 10))
            
            plt.subplot(3, 1, 1)
            plt.plot(history['epoch'], history['train_acc'], label='Train')
            plt.plot(history['epoch'], history['valid_acc'], label='Valid')
            plt.plot(history['epoch'], history['test_acc'], label='Test')
            plt.title('Accuracy over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(3, 1, 2)
            plt.plot(history['epoch'], history['train_logprob'], label='Train')
            plt.plot(history['epoch'], history['valid_logprob'], label='Valid')
            plt.plot(history['epoch'], history['test_logprob'], label='Test')
            plt.title('Log Probability over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Log Probability')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(3, 1, 3)
            plt.plot(history['epoch'], history['lambda'])
            plt.title('Lambda Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Lambda Value')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{results_dir}/training_metrics_with_lambda.png')
        
        # Close session to free GPU memory
        sess.close()
        
        return {
            'test_acc': accuracies['test'],
            'test_brier': epoch_summary['test'][-1][-1]['running_brier'],
            'test_logprob': epoch_summary['test'][-1][-1]['running_logprob'],
            'best_validation': best_validation,
            'best_epoch': best_model_epoch
        }


if __name__ == "__main__":
    np.random.seed(42)
    tf.set_random_seed(42)
    
    metrics = create_dvi_model()
    
    print("\nFinal test metrics:")
    print(f"  Test accuracy: {metrics['test_acc']:.4f}")
    print(f"  Test Brier score: {metrics['test_brier']:.4f}")
    print(f"  Test log probability: {metrics['test_logprob']:.4f}")
    
    print("\nDVI model training with cyclic lambda annealing completed.")