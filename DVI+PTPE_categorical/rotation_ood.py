import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import pandas as pd
import json
from sklearn.metrics import brier_score_loss, log_loss
from scipy.stats import entropy

# Import dependencies used in the original DVI model
# Assuming these modules are in the same directory as this script
import gaussian_variables as gv
import utils
import bayes_layers as bnn
from bayes_layers import logsoftmax
from bayes_models import MLP, PointMLP, AdaptedMLP
from scipy import ndimage
import pickle

def load_mnist():
    """Load and preprocess the MNIST dataset."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Process in two formats - one for the original DVI model (flattened)
    # and one for compatibility with the second script (with channel dimension)
    
    # Format for DVI model - flattened
    x_test_flat = x_test.reshape(-1, 28*28).astype('float32') / 255.0
    
    # Format for rotation (with channel dimension)
    x_test_channel = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # One-hot encode labels
    y_test_oh = tf.keras.utils.to_categorical(y_test, 10)
    
    print(f"Loaded {len(x_test)} test MNIST images.")
    return x_test_flat, x_test_channel, y_test, y_test_oh


def create_fashion_mnist_ood():
    """Load Fashion MNIST dataset as out-of-distribution data."""
    print("Loading Fashion MNIST as OOD data...")
    # Fashion MNIST dataset is the same size and format as MNIST but with clothing items
    (_, _), (fashion_x_test, fashion_y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Process in both formats
    fashion_x_test_flat = fashion_x_test.reshape(-1, 28*28).astype('float32') / 255.0
    fashion_x_test_channel = fashion_x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Use dummy labels (all zeros) for OOD data
    fashion_y_test_oh = np.zeros((len(fashion_y_test), 10), dtype='float32')
    
    print(f"Loaded {len(fashion_x_test)} Fashion MNIST images as OOD data.")
    return fashion_x_test_flat, fashion_x_test_channel, fashion_y_test_oh


def rotate_images(images, angle):
    """Rotate images by given angle in degrees using scipy instead of tf.contrib."""
    print(f"  Rotating images by {angle}°...")
    
    # Initialize output array
    rotated_images = np.zeros_like(images)
    
    # Process each image individually
    for i in range(images.shape[0]):
        # Extract the image (removing channel dimension for scipy)
        img = np.squeeze(images[i])
        
        # Rotate using scipy with bilinear interpolation
        rotated = ndimage.rotate(img, angle, reshape=False, order=1, mode='constant', cval=0.0)
        
        # Add channel dimension back and store in output array
        rotated_images[i, :, :, 0] = rotated
        
    return rotated_images


def parse_data_hyper(n_hidden_layer=2, hidden_size=512):
    """Recreate the hyperparameter configuration for loading the model."""
    hypers = {
        "x_dim": 784,                # 784 for flattened MNIST
        "y_dim": 10,                 # 10 classes for MNIST
        "hidden_dims": [hidden_size] * n_hidden_layer,
        "nonlinearity": "relu",
        "method": "bayes",
        "prior_type": ["gaussian", "he", "he"], 
        "n_epochs": 50,            
        "early_stop_patience": 10,
        "batch_size": 128,
        "learning_rate": 0.001,
        "lambda": 1.0,
        "warmup_updates": {'lambda': 14000.0},
        "anneal_updates": {'lambda': 1000.0},
        "optimizer": "adam",
        "gradient_clip": 0.1,
        "data_fraction": 1.0,
        "sections_to_run": ["train", "valid", "test"],
        "dataset_size": 50000  # Approximation, will be overwritten if training
    }

    print("Loaded hyperparameters for DVI model.")
    return hypers


def make_model(hypers):
    """Recreate the model architecture for loading the saved checkpoint."""
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

    target          = tf.placeholder(tf.float32, [None, 10])
    mlp.placeholders['target'] = target
    
    global_step     = tf.Variable(0, trainable=False, name='global_step')
    loss_val, logprob, all_surprise = loss(y, target, mlp, hypers, global_step)

    probs           = tf.exp(logsoftmax(y))

    # Calculate accuracy
    predicted_labels = tf.argmax(probs, axis=1)
    true_labels      = tf.argmax(target, axis=1)
    accuracy         = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, true_labels), tf.float32))

    # Calculate Brier score
    brier_score = tf.reduce_mean(tf.square(tf.reshape(target, [-1]) - tf.reshape(probs, [-1])))

    return {
        'model': mlp,
        'metrics': {
            'accuracy'     : accuracy,
            'brier'        : brier_score,
            'loss'         : loss_val,
            'logprob'      : logprob,
            'all_surprise' : all_surprise,
            'probs'        : probs
        },
        'global_step': global_step,
        'placeholders': {
            'input': ipt,
            'target': target
        }
    }


def load_dvi_model():
    """Load the DVI model from the saved checkpoint."""
    print("Loading DVI model from checkpoint...")
    
    # Reset TensorFlow graph
    tf.reset_default_graph()
    
    # Create session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    
    # Recreate model architecture
    hypers = parse_data_hyper(n_hidden_layer=2, hidden_size=512)
    model_and_metrics = make_model(hypers)
    
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    
    # Load weights from checkpoint
    saver = tf.train.Saver()
    checkpoint_path = 'mnist_dvi_model/best_model_checkpoint'
    
    if os.path.exists(checkpoint_path + '.meta'):
        saver.restore(sess, checkpoint_path)
        print(f"Successfully loaded DVI model from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found!")
    
    return model_and_metrics, sess, hypers


# def predict_dvi(model_and_metrics, x_data, sess, batch_size=128):
#     """
#     Get predictions from the DVI model.
    
#     Args:
#         model_and_metrics: The loaded model dictionary
#         x_data: Input data (should be flattened for DVI model)
#         sess: TensorFlow session
#         batch_size: Batch size for prediction
        
#     Returns:
#         Numpy array of predicted probabilities
#     """
#     all_probs = []
    
#     # Process in batches
#     for i in range(0, len(x_data), batch_size):
#         batch_x = x_data[i:i+batch_size]
        
#         # Use the DVI model's prediction function
#         feed_dict = {model_and_metrics['placeholders']['input']: batch_x}
#         batch_probs = sess.run(model_and_metrics['metrics']['probs'], feed_dict=feed_dict)
#         all_probs.append(batch_probs)
    
#     # Combine all batches
#     return np.vstack(all_probs)

def predict_dvi(model_and_metrics, x_data, y_data, sess, batch_size=128):
    """
    Get predictions and metrics from the DVI model.
    """
    all_metrics = {
        'probs': [],
        'accuracy': [],
        'brier': [],
        'logprob': []
    }
    
    # Process in batches
    for i in range(0, len(x_data), batch_size):
        batch_x = x_data[i:i+batch_size]
        batch_y = y_data[i:i+batch_size]
        
        # Use the DVI model to compute all metrics at once
        feed_dict = {
            model_and_metrics['placeholders']['input']: batch_x,
            model_and_metrics['placeholders']['target']: batch_y
        }
        
        # Run all metrics
        batch_results = sess.run(
            [
                model_and_metrics['metrics']['probs'],
                model_and_metrics['metrics']['accuracy'],
                model_and_metrics['metrics']['brier'],
                model_and_metrics['metrics']['logprob']
            ], 
            feed_dict=feed_dict
        )
        
        # Store results
        all_metrics['probs'].append(batch_results[0])
        all_metrics['accuracy'].append(batch_results[1])
        all_metrics['brier'].append(batch_results[2])
        all_metrics['logprob'].append(batch_results[3])
    
    # Combine all batches for probabilities
    all_metrics['probs'] = np.vstack(all_metrics['probs'])
    
    # Average batch metrics
    all_metrics['accuracy'] = np.mean(all_metrics['accuracy'])
    all_metrics['brier'] = np.mean(all_metrics['brier'])
    all_metrics['logprob'] = np.mean(all_metrics['logprob'])
    
    # Calculate additional metrics if needed
    all_metrics['confidences'] = np.max(all_metrics['probs'], axis=1)
    all_metrics['preds'] = np.argmax(all_metrics['probs'], axis=1)
    all_metrics['y_true_int'] = np.argmax(y_data, axis=1)
    all_metrics['entropies'] = -np.sum(all_metrics['probs'] * np.log(all_metrics['probs'] + 1e-10), axis=1)
    
    return all_metrics


def calculate_metrics(y_true, probs):
    """
    Calculate metrics for evaluating predictive performance.
    
    Replicates the metric calculation from the second script.
    """
    # Make sure we have the right format for y_true
    if len(y_true.shape) == 1:
        y_true_oh = tf.keras.utils.to_categorical(y_true, probs.shape[1])
    else:
        y_true_oh = y_true
    
    y_true_int = np.argmax(y_true_oh, axis=1)
    preds = np.argmax(probs, axis=1)
    
    # Accuracy
    acc = np.mean(preds == y_true_int)
    
    # Brier score (using sklearn for consistency with second script)
    brier = brier_score_loss(y_true_oh.flatten(), probs.flatten())
    
    # Log likelihood (negative log loss with small epsilon to avoid log(0))
    epsilon = 1e-10
    probs_clipped = np.clip(probs, epsilon, 1.0 - epsilon)
    log_likelihood = -log_loss(y_true_oh, probs_clipped)
    
    # Confidences
    confidences = np.max(probs, axis=1)
    
    # Entropies
    # entropies = np.array([entropy(prob + epsilon) for prob in probs])
    entropies = -np.sum(probs * np.log(probs), axis=1)
    
    return acc, brier, log_likelihood, confidences, entropies, preds, y_true_int


def run_rotation_experiment(model_and_metrics, sess, x_test_flat, x_test_channel, y_test_oh, 
                           rotation_angles, batch_size=128):
    """
    Run experiment with different rotation angles.
    
    Args:
        model_and_metrics: The loaded model and metrics
        sess: TensorFlow session
        x_test_flat: Flattened test data (for DVI model)
        x_test_channel: Test data with channel dimension (for rotation)
        y_test_oh: One-hot encoded test labels
        rotation_angles: List of angles to rotate images
        batch_size: Batch size for prediction
    
    Returns:
        List of results for each angle
    """
    results = []
    
    for angle in rotation_angles:
        print(f"Processing rotation angle: {angle}°")
        
        # Skip rotation for angle 0
        if angle == 0:
            rotated_channel = x_test_channel
        else:
            # Rotate images
            rotated_channel = rotate_images(x_test_channel, angle)
        
        # Convert back to flattened format for DVI model
        rotated_flat = rotated_channel.reshape(-1, 784)
        
        # Get predictions and metrics in one pass
        all_metrics = predict_dvi(model_and_metrics, rotated_flat, y_test_oh, sess, batch_size)
        
        # Extract key metrics for logging
        acc = all_metrics['accuracy']
        brier = all_metrics['brier']
        log_lik = all_metrics['logprob']
        
        print(f"    Accuracy: {acc:.4f}, Brier: {brier:.4f}, Log-likelihood: {log_lik:.4f}")
        
        # Create the metrics tuple in the expected format for compatibility
        metrics_tuple = (
            acc, 
            brier, 
            log_lik, 
            all_metrics['confidences'], 
            all_metrics['entropies'], 
            all_metrics['preds'], 
            all_metrics['y_true_int']
        )
        
        # Store results for this angle (as a list with one item for compatibility)
        results.append({
            'angle': angle,
            'metrics': [metrics_tuple]  # Still a list for compatibility with plotting code
        })
    
    return results

def run_ood_experiment(model_and_metrics, sess, fashion_x_test_flat, fashion_y_test_oh, 
                      batch_size=128):
    """
    Run experiment with out-of-distribution data (Fashion MNIST).
    
    Args:
        model_and_metrics: The loaded model and metrics
        sess: TensorFlow session
        fashion_x_test_flat: Flattened Fashion MNIST data
        fashion_y_test_oh: Dummy labels for Fashion MNIST
        batch_size: Batch size for prediction
    
    Returns:
        List of metrics for each run
    """
    print("Processing OOD data (Fashion MNIST)")
    
    # Get predictions and metrics in one pass
    all_metrics = predict_dvi(model_and_metrics, fashion_x_test_flat, fashion_y_test_oh, sess, batch_size)
    
    # Create the metrics tuple in the expected format for compatibility
    metrics_tuple = (
        all_metrics['accuracy'], 
        all_metrics['brier'], 
        all_metrics['logprob'], 
        all_metrics['confidences'], 
        all_metrics['entropies'], 
        all_metrics['preds'], 
        all_metrics['y_true_int']
    )
    
    # Return as a list with one item for compatibility with plotting code
    return [{
        'metrics': [metrics_tuple]  # Single run in a list for compatibility
    }]


def save_results_with_pickle(results, filename, result_type='rotation'):
    """
    Save DVI results with pickle in a format compatible with rotation_results and ood_results.
    
    Args:
        results: Results to save
        filename: Output filename (without extension)
        result_type: Type of results ('rotation' or 'ood')
    
    Returns:
        Structured results matching the expected format for rotation_results or ood_results
    """
    # Format DVI results to match the structure of rotation_results or ood_results
    if result_type == 'rotation':
        # Format: {'DVI': [[runs_for_angle0], [runs_for_angle15], ...]}
        dvi_rotation_results = {'DVI': []}
        
        for angle_result in results:
            # Get metrics (list of runs) for this angle
            angle_metrics = angle_result['metrics']
            dvi_rotation_results['DVI'].append(angle_metrics)
        
        # Create directory if it doesn't exist
        os.makedirs('mnist_dvi_results', exist_ok=True)
        
        # Save with pickle
        output_path = os.path.join('mnist_dvi_results', f"{filename}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(dvi_rotation_results, f)
        print(f"DVI rotation results saved to {output_path}")
        
        return dvi_rotation_results
    
    else:  # OOD results
        # Format: {'DVI': [[runs_for_ood]]}
        dvi_ood_results = {'DVI': []}
        
        # For OOD, there's only one entry in the outer list (just the OOD dataset)
        ood_metrics = results[0]['metrics']
        dvi_ood_results['DVI'].append(ood_metrics)
        
        # Create directory if it doesn't exist
        os.makedirs('mnist_dvi_results', exist_ok=True)
        
        # Save with pickle
        output_path = os.path.join('mnist_dvi_results', f"{filename}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(dvi_ood_results, f)
        print(f"DVI OOD results saved to {output_path}")
        
        return dvi_ood_results



def main():
    """Main function to run benchmarks for the DVI model."""
    print("Starting DVI model benchmarking")
    
    # Define parameters
    rotation_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 180]
    batch_size = 128
    
    try:
        # Load the DVI model
        model_and_metrics, sess, hypers = load_dvi_model()
        
        # Load test data
        x_test_flat, x_test_channel, y_test, y_test_oh = load_mnist()
        
        # Load OOD data
        fashion_x_test_flat, fashion_x_test_channel, fashion_y_test_oh = create_fashion_mnist_ood()
        
        # Save hyperparameters for reference
        os.makedirs('mnist_dvi_results', exist_ok=True)
        with open('mnist_dvi_results/model_hyperparameters.json', 'w') as f:
            json.dump(hypers, f, indent=2)
        
        # Run rotation experiment
        print("\nRunning rotation experiment...")
        rotation_results = run_rotation_experiment(
            model_and_metrics, sess,
            x_test_flat, x_test_channel, y_test_oh,
            rotation_angles, batch_size
        )
        
        # Save rotation results
        rotation_df = save_results_with_pickle(rotation_results, 'dvi_rotation_results', 'rotation')
        
        # Run OOD experiment
        print("\nRunning OOD experiment...")
        ood_results = run_ood_experiment(
            model_and_metrics, sess,
            fashion_x_test_flat, fashion_y_test_oh,
            batch_size
        )
        
        # Save OOD results
        ood_df = save_results_with_pickle(ood_results, 'dvi_ood_results', 'ood')
        
        print("\nAll benchmarks completed successfully!")
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close TensorFlow session
        if 'sess' in locals():
            sess.close()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.set_random_seed(42)
    main()