import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, log_loss
from scipy.stats import entropy
import os
import pickle

np.random.seed(42)
tf.set_random_seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load MNIST dataset
def load_mnist():
    """Load and preprocess the MNIST dataset."""
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Reshape to have a channel dimension, but no need for special preprocessing for MLP
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train_oh = tf.keras.utils.to_categorical(y_train, 10)
    y_test_oh = tf.keras.utils.to_categorical(y_test, 10)
    print(f"Loaded {len(x_train)} training and {len(x_test)} test MNIST images.")
    return (x_train, y_train, y_train_oh), (x_test, y_test, y_test_oh)

# Create OOD dataset using Fashion MNIST
def create_fashion_mnist_ood():
    """Load Fashion MNIST dataset as out-of-distribution data."""
    print("Loading Fashion MNIST as OOD data...")
    # Fashion MNIST dataset is the same size and format as MNIST but with clothing items
    (_, _), (fashion_x_test, fashion_y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Process exactly like MNIST for consistency
    fashion_x_test = fashion_x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Use dummy labels (all zeros) since we don't care about actual classes for OOD
    fashion_y_test_oh = np.zeros((len(fashion_y_test), 10), dtype='float32')
    
    print(f"Loaded {len(fashion_x_test)} Fashion MNIST images as OOD data.")
    return fashion_x_test, fashion_y_test_oh

# Image transformation functions
def rotate_images(images, angle):
    """Rotate images by given angle in degrees using spline interpolation."""
    print(f"  Rotating images by {angle}°...")
    # Create TF session
    with tf.Session() as sess:
        # Convert angle to radians
        angle_rad = angle * np.pi / 180.0
        
        # Create tensor for rotation
        images_tensor = tf.convert_to_tensor(images)
        rotated_tensor = tf.contrib.image.rotate(
            images_tensor, 
            angle_rad,
            interpolation='BILINEAR'  # Use BILINEAR for spline interpolation
        )
        
        # Run session to get rotated images
        rotated_images = sess.run(rotated_tensor)
        
    return rotated_images

# Model architectures and creation functions
def create_vanilla_model():
    """Create a basic MLP model with two hidden layers of 1024 units each for MNIST classification."""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax', name='output')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_dropout_model(dropout_rate=0.5):
    """Create an MLP model with two hidden layers of 1024 units each and dropout."""
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation='softmax', name='output')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_ensemble_models(n_models=10):
    """Create an ensemble of n independently initialized models."""
    models = []
    for i in range(n_models):
        model = create_vanilla_model()
        models.append(model)
    return models

# Prediction functions
def predict_vanilla(model, x, batch_size=128):
    """Standard prediction for vanilla models."""
    # Make sure the session is initialized
    with tf.keras.backend.get_session().as_default():
        return model.predict(x, batch_size=batch_size)

def predict_dropout(model, x, n_samples=10, batch_size=128):
    """Monte Carlo dropout prediction with n_samples forward passes.

    Implements the method from Gal & Ghahramani (2016) by keeping dropout
    active during inference to obtain model uncertainty estimates.
    """
    
    # Create the dropout prediction function once
    dropout_predict = tf.keras.backend.function(
        [model.input, tf.keras.backend.learning_phase()],
        [model.output]
    )
    
    # Initialize list to store predictions
    preds = []
    
    # Run n_samples predictions
    for _ in range(n_samples):
        # Get predictions with dropout active (training=1)
        pred = dropout_predict([x, 1])[0]
        preds.append(pred)
    
    # Return mean prediction (numpy array)
    return np.mean(np.array(preds), axis=0)

def predict_ensemble(models, x, batch_size=128):
    """Ensemble prediction by averaging predictions from all models."""
    preds = []
    for model in models:
        # Make sure the session is initialized
        with tf.keras.backend.get_session().as_default():
            pred = model.predict(x, batch_size=batch_size)
        preds.append(pred)
    return np.mean(preds, axis=0)

# Metric calculation
def calculate_metrics(y_true, probs):
    """Calculate accuracy, Brier score, log likelihood, confidences, and entropies."""
    # Make sure we have the right format for y_true
    y_true_oh = y_true
    if len(y_true.shape) == 1:
        y_true_oh = tf.keras.utils.to_categorical(y_true, probs.shape[1])
    
    y_true_int = np.argmax(y_true_oh, axis=1)
    preds = np.argmax(probs, axis=1)
    
    # Accuracy
    acc = np.mean(preds == y_true_int)
    
    # Brier score 
    # Flatten the one-hot encoded labels and predictions
    brier = brier_score_loss(y_true_oh.flatten(), probs.flatten())
    
    # Log likelihood (negative log loss)
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    probs_clipped = np.clip(probs, epsilon, 1.0 - epsilon)
    log_likelihood = -log_loss(y_true_oh, probs_clipped)
    
    # Confidences
    confidences = np.max(probs, axis=1)
    
    # Entropies
    entropies = np.array([entropy(prob + epsilon) for prob in probs])
    
    return acc, brier, log_likelihood, confidences, entropies, preds, y_true_int

# Experiment functions
def run_rotation_experiment(models_dict, rotation_angles, x_test, y_test_oh, n_runs=5):
    """Run experiment with different rotation angles."""
    results = {model_name: [] for model_name in models_dict.keys()}
    
    for angle in rotation_angles:
        print(f"Processing rotation angle: {angle}°")
        
        # Create rotated test set
        rotated_images = rotate_images(x_test, angle)
        
        for model_name, model_info in models_dict.items():
            print(f"  Evaluating model: {model_name}")
            model = model_info['model']
            predict_fn = model_info['predict_fn']
            
            model_runs = []
            for run in range(n_runs):
                probs = predict_fn(model, rotated_images)
                metrics = calculate_metrics(y_test_oh, probs)
                model_runs.append(metrics)
                print(f"    Run {run+1}/{n_runs} - Accuracy: {metrics[0]:.4f}, Brier: {metrics[1]:.4f}, LogLik: {metrics[2]:.4f}")
            
            results[model_name].append(model_runs)
    
    return results

def run_ood_experiment(models_dict, ood_images, ood_labels, n_runs=5):
    """Run experiment with OOD data."""
    results = {model_name: [] for model_name in models_dict.keys()}
    
    for model_name, model_info in models_dict.items():
        print(f"Evaluating OOD performance for model: {model_name}")
        model = model_info['model']
        predict_fn = model_info['predict_fn']
        
        model_runs = []
        for run in range(n_runs):
            probs = predict_fn(model, ood_images)
            metrics = calculate_metrics(ood_labels, probs)
            model_runs.append(metrics)
            print(f"  Run {run+1}/{n_runs} completed")
        
        results[model_name].append(model_runs)
    
    return results

# Plotting functions
def plot_brier(ax, results, shift_values, title, x_label):
    """Plot only Brier score against shift intensity."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for i, (model_name, model_results) in enumerate(results.items()):
        color = colors[i]
        
        # Extract Brier scores
        briers = [[run[1] for run in model_result] for model_result in model_results]
        mean_brier = [np.mean(brier_runs) for brier_runs in briers]
        std_brier = [np.std(brier_runs) for brier_runs in briers]
        
        # Plot Brier score
        ax.plot(shift_values, mean_brier, label=model_name, color=color)
        ax.fill_between(
            shift_values,
            [m - s for m, s in zip(mean_brier, std_brier)],
            [m + s for m, s in zip(mean_brier, std_brier)],
            alpha=0.2,
            color=color
        )
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel('Brier Score')
    ax.set_title(f'{title}')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis limits from 0 to 180
    ax.set_xlim(0, 180)
    ax.legend(loc='lower right')

        

def plot_log_likelihood(ax, results, shift_values, title, x_label):
    """Plot log likelihood against shift intensity."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for i, (model_name, model_results) in enumerate(results.items()):
        color = colors[i]
        
        # Extract Log Likelihood scores
        log_liks = [[run[2] for run in model_result] for model_result in model_results]
        mean_log_lik = [np.mean(log_lik_runs) for log_lik_runs in log_liks]
        std_log_lik = [np.std(log_lik_runs) for log_lik_runs in log_liks]
        
        # Plot Log Likelihood
        ax.plot(shift_values, mean_log_lik, label=model_name, color=color)
        ax.fill_between(
            shift_values,
            [m - s for m, s in zip(mean_log_lik, std_log_lik)],
            [m + s for m, s in zip(mean_log_lik, std_log_lik)],
            alpha=0.2,
            color=color
        )
    
    # Set labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel('Log Likelihood')
    ax.set_title(f'{title}')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis limits from 0 to 180
    ax.set_xlim(0, 180)

def plot_count_vs_confidence_rotation(ax, results, angle_idx, title):
    """Plot count vs confidence threshold."""
    # Use slightly less than 1.0 as the maximum threshold to avoid issues at exactly 1.0
    tau_values = np.linspace(0.0, 0.99, 50)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for i, (model_name, model_results) in enumerate(results.items()):
        color = colors[i]
        
        # Get all runs for this angle
        angle_results = model_results[angle_idx]
        
        # Count samples above different confidence thresholds
        all_counts = []
        for run in angle_results:
            confidences = run[3]  # Confidences from the run
            
            counts_at_tau = []
            for tau in tau_values:
                count = np.sum(confidences >= tau)
                counts_at_tau.append(count)
                
            all_counts.append(counts_at_tau)
        
        # Average across runs
        mean_counts = np.mean(all_counts, axis=0)
        
        # Plot count vs confidence threshold
        ax.plot(tau_values, mean_counts, label=model_name, color=color)
    
    # Set labels and title
    ax.set_xlabel("Confidence Threshold (τ)")
    ax.set_ylabel("Number of examples with confidence ≥ τ")
    ax.set_title(f'{title}')
    ax.grid(True, linestyle='--', alpha=0.7)

def plot_count_vs_confidence_ood(ax, results, title):
    """Plot count vs confidence threshold."""
    # Use slightly less than 1.0 as the maximum threshold to avoid issues at exactly 1.0
    tau_values = np.linspace(0.0, 0.99, 50)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for i, (model_name, model_results) in enumerate(results.items()):
        color = colors[i]
        
        # Count samples above different confidence thresholds
        all_counts = []
        for run in model_results:
            confidences = run[0][3]  # Confidences from the run
            
            counts_at_tau = []
            for tau in tau_values:
                count = np.sum(confidences >= tau)
                counts_at_tau.append(count)
                
            all_counts.append(counts_at_tau)
        
        # Average across runs
        mean_counts = np.mean(all_counts, axis=0)
        
        # Plot count vs confidence threshold
        ax.plot(tau_values, mean_counts, label=model_name, color=color)
    
    # Set labels and title
    ax.set_xlabel("Confidence Threshold (τ)")
    ax.set_ylabel("Number of examples with confidence ≥ τ")
    ax.set_title(f'{title}')
    ax.grid(True, linestyle='--', alpha=0.7)

def plot_entropy_distribution(ax, results, title):
    """Plot entropy distribution for OOD data."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    # Define bins for histogram
    bins = np.linspace(0, 2.5, 50) # max entropy about 2.3
    
    for i, (model_name, model_results) in enumerate(results.items()):
        color = colors[i]
        
        # Extract entropies from all runs
        all_entropies = []
        for run in model_results[0]:  # Only one entry for OOD experiment
            entropies = run[4]  # Entropies from the run
            all_entropies.extend(entropies)
        
        # Plot histogram as density (normalized) 
        ax.hist(all_entropies, bins=bins, density=False, histtype='step', 
                label=model_name, color=color, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel("Entropy (Nats)")
    ax.set_ylabel("Number of examples")
    ax.set_title(f'{title}')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

def plot_acc_vs_confidence(ax, results, angle_idx, title):
    """Plot accuracy vs confidence threshold."""
    tau_values = np.linspace(0.0, 0.99, 50)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for i, (model_name, model_results) in enumerate(results.items()):
        color = colors[i]
        
        # Get all runs for this angle
        angle_results = model_results[angle_idx]
        
        # Calculate accuracy at different confidence thresholds
        all_accs = []
        for run in angle_results:
            confidences = run[3]  # Confidences from the run
            preds = run[5]  # Predicted labels
            true_labels = run[6]  # True labels
            
            accs_at_tau = []
            for tau in tau_values:
                # Filter by confidence threshold
                mask = confidences >= tau
                if np.sum(mask) > 0:
                    acc = np.mean(preds[mask] == true_labels[mask])
                else:
                    # Use NaN when no samples meet the threshold
                    acc = float('nan')  
                accs_at_tau.append(acc)
            
            all_accs.append(accs_at_tau)
        
        # Average across runs
        mean_accs = np.mean(all_accs, axis=0)
        
        # Plot confidence vs accuracy
        ax.plot(tau_values, mean_accs, label=model_name, color=color)
    
    # Set labels and title
    ax.set_xlabel("τ")
    ax.set_ylabel("Accuracy on examples y(p) ≥ τ")
    ax.set_title(f'{title}')
    ax.grid(True, linestyle='--', alpha=0.7)



# Functions to save and load models
def save_vanilla_model(model):
    """Save vanilla model to disk."""
    print("Saving vanilla model...")
    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/vanilla_model")
    print("Vanilla model saved.")
    return True

def save_dropout_model(model):
    """Save dropout model to disk."""
    print("Saving dropout model...")
    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/dropout_model")
    print("Dropout model saved.")
    return True

def save_ensemble_models(models):
    """Save ensemble models to disk."""
    print(f"Saving {len(models)} ensemble models...")
    os.makedirs("saved_models", exist_ok=True)
    for i, model in enumerate(models):
        model.save(f"saved_models/ensemble_model_{i}")
    print(f"All {len(models)} ensemble models saved.")
    return True

def load_vanilla_model():
    """Load vanilla model from disk if available."""
    try:
        if os.path.exists("saved_models/vanilla_model"):
            print("Loading vanilla model from disk...")
            model = tf.keras.models.load_model("saved_models/vanilla_model")
            print("Vanilla model loaded successfully.")
            return model
        else:
            print("No saved vanilla model found.")
            return None
    except Exception as e:
        print(f"Error loading vanilla model: {e}")
        return None

def load_dropout_model():
    """Load dropout model from disk if available."""
    try:
        if os.path.exists("saved_models/dropout_model"):
            print("Loading dropout model from disk...")
            model = tf.keras.models.load_model("saved_models/dropout_model")
            print("Dropout model loaded successfully.")
            return model
        else:
            print("No saved dropout model found.")
            return None
    except Exception as e:
        print(f"Error loading dropout model: {e}")
        return None

def load_ensemble_models(ensemble_size=10):
    """Load ensemble models from disk if available."""
    try:
        # Check if all ensemble models exist
        all_exist = True
        for i in range(ensemble_size):
            if not os.path.exists(f"saved_models/ensemble_model_{i}"):
                all_exist = False
                print(f"Ensemble model {i} not found, can't load complete ensemble.")
                break
        
        if all_exist:
            print(f"Loading {ensemble_size} ensemble models from disk...")
            models = []
            for i in range(ensemble_size):
                model = tf.keras.models.load_model(f"saved_models/ensemble_model_{i}")
                models.append(model)
            print(f"All {ensemble_size} ensemble models loaded successfully.")
            return models
        else:
            return None
    except Exception as e:
        print(f"Error loading ensemble models: {e}")
        return None

# Main function
def main():
    # Set up experiment parameters
    rotation_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 180]
    n_runs = 1  
    ensemble_size = 10  # Number of models in ensemble
    
    try:
        # Initialize global session for TensorFlow
        sess = tf.Session()
        tf.keras.backend.set_session(sess)
        
        # Load datasets
        print("Loading datasets...")
        (x_train, y_train, y_train_oh), (x_test, y_test, y_test_oh) = load_mnist()
        ood_images, ood_labels = create_fashion_mnist_ood()
        
        # Create output directory
        os.makedirs('results', exist_ok=True)
        
        # Try to load vanilla model
        vanilla_model = load_vanilla_model()
        if vanilla_model is None:
            print("Training vanilla model from scratch...")
            vanilla_model = create_vanilla_model()
            vanilla_model.fit(x_train, y_train_oh, epochs=10, batch_size=128, verbose=1)
            save_vanilla_model(vanilla_model)
        
        # Try to load dropout model
        dropout_model = load_dropout_model()
        if dropout_model is None:
            print("Training dropout model from scratch...")
            dropout_model = create_dropout_model()
            dropout_model.fit(x_train, y_train_oh, epochs=20, batch_size=128, verbose=1)
            save_dropout_model(dropout_model)
        
        # Try to load ensemble models
        ensemble_models = load_ensemble_models(ensemble_size)
        if ensemble_models is None:
            print("Training ensemble models from scratch...")
            ensemble_models = create_ensemble_models(n_models=ensemble_size)
            for i, model in enumerate(ensemble_models):
                print(f"Training ensemble model {i+1}/{ensemble_size}...")
                model.fit(x_train, y_train_oh, epochs=10, batch_size=128, verbose=1)
            save_ensemble_models(ensemble_models)
        
        # Create models dictionary with prediction functions
        models_dict = {
            'Vanilla': {'model': vanilla_model, 'predict_fn': predict_vanilla},
            'Dropout': {'model': dropout_model, 'predict_fn': predict_dropout},
            'Ensemble': {'model': ensemble_models, 'predict_fn': predict_ensemble}
        }
        
        # Run experiment with all rotation angles
        print("Running rotation experiment...")
        rotation_results = run_rotation_experiment(models_dict, rotation_angles, x_test, y_test_oh, n_runs)
        
        print("Running OOD experiment...")
        ood_results = run_ood_experiment(models_dict, ood_images, ood_labels, n_runs)



        # Load the DVI results
        print("Loading DVI model results...")
        with open('./mnist_dvi_results/dvi_rotation_results.csv.pkl', 'rb') as f:
            dvi_rotation_results = pickle.load(f)
        with open('./mnist_dvi_results/dvi_ood_results.csv.pkl', 'rb') as f:
            dvi_ood_results = pickle.load(f)



        # Combine the results if DVI results were loaded successfully
        if dvi_rotation_results is not None:
            print("Integrating DVI rotation results...")
            # Merge dictionaries to include DVI in rotation results
            rotation_results = {**rotation_results, **dvi_rotation_results}
            
        if dvi_ood_results is not None:
            print("Integrating DVI OOD results...")
            # Merge dictionaries to include DVI in OOD results
            ood_results = {**ood_results, **dvi_ood_results}


        
        # Create figure
        print("Creating figure...")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot results - ONLY BRIER SCORE in first figure
        plot_brier(axes[0, 0], rotation_results, rotation_angles, 
                 '(a) Brier Score, Rotated MNIST', 'Intensity of Shift')
        
        # Plot LOG LIKELIHOOD in the second figure
        plot_log_likelihood(axes[0, 1], rotation_results, rotation_angles,
                          '(b) Log Likelihood, Rotated MNIST', 'Intensity of Shift')
        
        # Plot accuracy vs confidence for 90° rotation (replaced with count vs confidence)
        angle_90_idx = rotation_angles.index(90)
        plot_acc_vs_confidence(axes[0, 2], rotation_results, angle_90_idx, 
                                '(c) Confidence vs Count, Rotated 90°')
        
        # Plot count vs confidence for 90° rotation
        plot_count_vs_confidence_rotation(axes[1, 0], rotation_results, angle_90_idx, 
                               '(d) Confidence vs Count, Rotated 90°')
        
        # Plot OOD results
        plot_entropy_distribution(axes[1, 1], ood_results, '(e) Entropy Distribution on FashionMNIST (OOD)')
        plot_count_vs_confidence_ood(axes[1, 2], ood_results, '(f) Confidence vs Count, FashionMNIST (OOD)')
        
        # Adjust layout and save figure
        plt.tight_layout()
        plt.savefig('results/uncertainty_estimation_results.pdf', dpi=300, bbox_inches='tight')
        plt.savefig('results/uncertainty_estimation_results.png', dpi=300, bbox_inches='tight')
        print("Figure saved to results/uncertainty_estimation_results.pdf and .png")
        
        # Display figure
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()