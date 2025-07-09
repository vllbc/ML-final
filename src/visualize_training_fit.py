import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_fit(model_name, horizon):
    """
    Plots the model's predictions on the training data against the actual values.
    """
    preds_path = f'results/{model_name}_train_preds_horizon_{horizon}.npy'
    labels_path = f'results/{model_name}_train_labels_horizon_{horizon}.npy'
    
    if not os.path.exists(preds_path) or not os.path.exists(labels_path):
        print(f"Training prediction files for {model_name} with horizon {horizon} not found. Skipping plot.")
        return

    predictions = np.load(preds_path)
    labels = np.load(labels_path)

    # Since predictions can be multi-dimensional (for multi-step forecasts),
    # we can plot the first forecasted step against the true value for simplicity.
    # We also flatten the arrays to make them 1D for plotting.
    preds_to_plot = predictions[:, 0].flatten()
    labels_to_plot = labels[:, 0].flatten()
    
    plt.figure(figsize=(15, 7))
    plt.plot(labels_to_plot, label='Actual Values (Training Set)')
    plt.plot(preds_to_plot, label='Predicted Values (Training Set)', alpha=0.7)
    plt.title(f'{model_name} - Training Data Fit (Horizon: {horizon} days)')
    plt.xlabel('Time Steps')
    plt.ylabel('Global Active Power')
    plt.legend()
    
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # Save the plot
    plot_path = f'plots/{model_name}_training_fit_horizon_{horizon}.png'
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training fit plot to {plot_path}")

def main():
    models = ["LSTM", "Transformer", "Custom"]
    horizons = [90, 365]
    
    for model in models:
        for h in horizons:
            plot_training_fit(model, h)

if __name__ == '__main__':
    main() 