import torch
import torch.optim as optim
import itertools
import pandas as pd

from model_definitions import LSTMModel, TransformerModel, CustomModel
from training_pipeline import get_data_loaders, train_model, evaluate_model

def tune_hyperparameters():
    """
    Performs a grid search for hyperparameters for all model types.
    """
    # Use the shorter horizon for faster tuning
    input_window = 90
    output_window = 90
    
    # Use larger batch for tuning. Note: get_data_loaders no longer returns val_loader
    train_loader, _, _, input_dim = get_data_loaders(input_window, output_window, batch_size=64) 
    
    param_grid = {
        'LSTM': {
            'hidden_dim': [32, 64],
            'num_layers': [1, 2],
            'learning_rate': [0.001, 0.0005]
        },
        'Transformer': {
            'd_model': [32, 64, 128, 256, 512],
            'nhead': [2, 4, 8, 16],
            'num_encoder_layers': [2, 3, 4, 5],
            'learning_rate': [0.001, 0.0005]
        },
        'Custom': {
            'd_model': [32, 64],
            'nhead': [2, 4],
            'cnn_out_channels': [32, 64],
            'learning_rate': [0.001, 0.0005]
        }
    }

    best_params_overall = {}

    for model_name, grid in param_grid.items():
        print(f"\n--- Tuning Hyperparameters for {model_name} ---")
        best_train_loss = float('inf')
        best_params_for_model = {}
        
        # Generate all combinations of hyperparameters
        keys, values = zip(*grid.items())
        hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        for i, params in enumerate(hyperparameter_combinations):
            print(f"  Testing combination {i+1}/{len(hyperparameter_combinations)}: {params}")

            # Pop learning_rate as it's not a model parameter
            lr = params.pop('learning_rate')

            # Instantiate model with current params
            if model_name == 'LSTM':
                model = LSTMModel(input_dim=input_dim, output_dim=output_window, **params)
            elif model_name == 'Transformer':
                model = TransformerModel(input_dim=input_dim, dim_feedforward=256, output_dim=output_window, **params)
            elif model_name == 'Custom':
                model = CustomModel(input_dim=input_dim, num_encoder_layers=3, dim_feedforward=256, output_dim=output_window, **params)

            # Add it back for logging
            params['learning_rate'] = lr

            # Create optimizer and scheduler
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
            
            # Train for fewer epochs for speed, get the final training loss
            _, loss = train_model(model, train_loader, optimizer, scheduler, num_epochs=20)
            
            if loss < best_train_loss:
                best_train_loss = loss
                best_params_for_model = params

        best_params_overall[model_name] = best_params_for_model
        print(f"  Best Training Loss for {model_name}: {best_train_loss:.4f}")
        print(f"  Best Hyperparameters for {model_name}: {best_params_for_model}")

    print("\n--- Best Hyperparameters Found ---")
    for model_name, params in best_params_overall.items():
        print(f"  {model_name}: {params}")
        
    # Save best params to a file
    pd.DataFrame(best_params_overall).to_json("results/best_hyperparameters.json", indent=4)
    print("\nBest hyperparameters saved to results/best_hyperparameters.json")


if __name__ == '__main__':
    tune_hyperparameters() 