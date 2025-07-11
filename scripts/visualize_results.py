import numpy as np
import matplotlib.pyplot as plt
import os

def plot_predictions(model_name, horizon):
    preds_path = f'results/{model_name}_preds_horizon_{horizon}.npy'
    labels_path = f'results/{model_name}_labels_horizon_{horizon}.npy'
    assert os.path.exists(preds_path) and os.path.exists(labels_path), f"Prediction or label file not found for {model_name}, horizon {horizon}."

    predictions = np.load(preds_path)
    labels = np.load(labels_path)

    plt.figure(figsize=(15, 6))
    plt.plot(labels[0, :], label='Ground Truth', color='blue', marker='.')
    plt.plot(predictions[0, :], label='Prediction', color='red', linestyle='--', marker='.')
    
    plt.title(f'{model_name} Prediction vs. Ground Truth (Horizon: {horizon} days)')
    plt.xlabel('Time (Days)')
    plt.ylabel('Global Active Power')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('plots', exist_ok=True)
    
    plot_filename = f'plots/{model_name}_prediction_horizon_{horizon}.png'
    plt.savefig(plot_filename)
    print(f"Saved plot to {plot_filename}")
    plt.close()

def main():
    models = ['LSTM', 'Transformer', 'HTFN']
    horizons = [90, 365]

    for model in models:
        for horizon in horizons:
            plot_predictions(model, horizon)

if __name__ == '__main__':
    main() 