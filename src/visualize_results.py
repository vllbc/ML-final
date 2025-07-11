import numpy as np
import matplotlib.pyplot as plt
import os

def plot_predictions(model_name, horizon):
    """
    Loads predictions and ground truth, then plots them for a given model and horizon.
    """
    preds_path = f'results/{model_name}_preds_horizon_{horizon}.npy'
    labels_path = f'results/{model_name}_labels_horizon_{horizon}.npy'
    
    if not os.path.exists(preds_path) or not os.path.exists(labels_path):
        print(f"Prediction or label file not found for {model_name}, horizon {horizon}. Skipping.")
        return

    predictions = np.load(preds_path)
    labels = np.load(labels_path)

    # Plot the first sample from the test set
    plt.figure(figsize=(15, 6))
    plt.plot(labels[0, :], label='Ground Truth', color='blue', marker='.')
    plt.plot(predictions[0, :], label='Prediction', color='red', linestyle='--', marker='.')
    
    plt.title(f'{model_name} Prediction vs. Ground Truth (Horizon: {horizon} days)')
    plt.xlabel('Time (Days)')
    plt.ylabel('Global Active Power')
    plt.legend()
    plt.grid(True)
    
    # Ensure the output directory exists
    os.makedirs('plots', exist_ok=True)
    
    plot_filename = f'plots/{model_name}_prediction_horizon_{horizon}.png'
    plt.savefig(plot_filename)
    print(f"Saved plot to {plot_filename}")
    plt.close()

def main():
    """
    Main function to generate plots for all models and horizons.
    """
    models = ['LSTM', 'Transformer', 'HTFN']
    horizons = [90, 365]

    for model in models:
        for horizon in horizons:
            plot_predictions(model, horizon)

if __name__ == '__main__':
    main() 