import optuna
import torch
import torch.nn as nn
import numpy as np
import json
import os

from model_definitions import LSTMModel, TransformerModel, HTFN
from training_pipeline import get_data_loaders, evaluate_model

# --- Global Settings ---
N_TRIALS = 50 # Number of trials for Optuna to run
EPOCHS = 50
EVAL_INTERVAL = 10

def objective(trial, model_name, train_loader, test_loader, power_scaler, input_dim, horizon):
    """
    Optuna objective function.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Define Hyperparameter Search Space ---
    if model_name == "LSTM":
        hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128, 256])
        num_layers = trial.suggest_int("num_layers", 1, 4)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        model = LSTMModel(input_dim, hidden_dim, num_layers, horizon).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    elif model_name == "Transformer":
        d_model = trial.suggest_categorical("d_model", [32, 64, 128, 256])
        nhead = trial.suggest_categorical("nhead", [2, 4, 8])
        num_encoder_layers = trial.suggest_int("num_encoder_layers", 1, 6)
        dim_feedforward = trial.suggest_categorical("dim_feedforward", [128, 256, 512, 1024])
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, horizon, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    elif model_name == "HTFN":
        gru_hidden_dim = trial.suggest_categorical("gru_hidden_dim", [32, 64, 128])
        gru_num_layers = trial.suggest_int("gru_num_layers", 1, 3)
        cross_attention_heads = trial.suggest_categorical("cross_attention_heads", [2, 4, 8])
        transformer_d_model = trial.suggest_categorical("transformer_d_model", [64, 128, 256])
        transformer_nhead = trial.suggest_categorical("transformer_nhead", [2, 4, 8])
        transformer_num_layers = trial.suggest_int("transformer_num_layers", 1, 4)
        transformer_dim_feedforward = trial.suggest_categorical("transformer_dim_feedforward", [256, 512, 1024])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        
        model = HTFN(
            input_dim=input_dim,
            output_dim=horizon,
            gru_hidden_dim=gru_hidden_dim,
            gru_num_layers=gru_num_layers,
            cross_attention_heads=cross_attention_heads,
            transformer_d_model=transformer_d_model,
            transformer_nhead=transformer_nhead,
            transformer_num_layers=transformer_num_layers,
            transformer_dim_feedforward=transformer_dim_feedforward,
            dropout=dropout
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    else:
        raise ValueError("Unknown model name")

    # --- 2. Training and Evaluation Loop ---
    criterion = nn.MSELoss()
    test_metrics = []

    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % EVAL_INTERVAL == 0:
            mse, mae, preds, labels = evaluate_model(model, test_loader)
            preds_inv = power_scaler.inverse_transform(preds)
            labels_inv = power_scaler.inverse_transform(labels)
            mae_inv = np.mean(np.abs(preds_inv - labels_inv))
            test_metrics.append(mae_inv)
            # Shortened print statement for clarity during tuning runs
            # print(f"Trial {trial.number}, Ep {epoch+1}, MAE: {mae_inv:.4f}")

    if not test_metrics:
        return float('inf')
        
    return np.mean(test_metrics)


def main():
    windows_to_tune = [90, 365]
    all_best_params = {}

    for window in windows_to_tune:
        print(f"\n{'='*30}\nStarting tuning for INPUT_WINDOW={window}, HORIZON={window}\n{'='*30}")
        all_best_params[f"window_{window}"] = {}
        
        # --- Load Data ---
        train_loader, test_loader, power_scaler, input_dim = get_data_loaders(
            input_window=window,
            output_window=window
        )

        if not test_loader:
            print(f"Test loader is empty for window size {window}. Cannot run hyperparameter tuning. Skipping.")
            continue
            
        for model_name in ["LSTM", "Transformer", "HTFN"]:
            print(f"\n--- Tuning for {model_name} ---")
            
            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda trial: objective(trial, model_name, train_loader, test_loader, power_scaler, input_dim, horizon=window),
                n_trials=N_TRIALS
            )

            print(f"\nBest trial for {model_name} (Window: {window}):")
            print(f"  Value (Avg MAE): {study.best_value:.4f}")
            print("  Params: ")
            for key, value in study.best_params.items():
                print(f"    {key}: {value}")
            
            all_best_params[f"window_{window}"][model_name] = {
                "value": study.best_value,
                "params": study.best_params
            }

    # --- Save all results to a single JSON file ---
    output_path = 'results/best_hyperparameters.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_best_params, f, indent=4)
        
    print(f"\n\nAll tuning complete. Best hyperparameters saved to {output_path}")


if __name__ == '__main__':
    main() 