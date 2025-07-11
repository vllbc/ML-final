import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import random
from model_definitions import LSTMModel, TransformerModel, HTFN
import json

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def create_sliding_windows(data, input_window_size, output_window_size):
    X, y = [], []
    for i in range(len(data) - input_window_size - output_window_size + 1):
        input_slice = data[i:i + input_window_size]
        output_slice = data[i + input_window_size:i + input_window_size + output_window_size, 0] # Target is Global_active_power
        X.append(input_slice)
        y.append(output_slice)
    return np.array(X), np.array(y)

def get_data_loaders(input_window, output_window, batch_size=16):
    """
    Creates and returns train and test data loaders.
    The scaler is fit ONLY on the training data.
    """
    train_df = pd.read_csv('data/processed_train.csv', index_col='DateTime', parse_dates=True)
    test_df = pd.read_csv('data/processed_test.csv', index_col='DateTime', parse_dates=True)
    
    # Fit scaler ONLY on training data
    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train_df)
    
    # Transform test data with the same scaler
    if not test_df.empty:
        scaled_test_data = scaler.transform(test_df)
    else:
        scaled_test_data = np.array([])

    # Create a separate scaler for the target variable for inverse transforming
    power_scaler = MinMaxScaler()
    power_scaler.fit(train_df[['Global_active_power']])
    
    # Create sliding windows
    X_train, y_train = create_sliding_windows(scaled_train_data, input_window, output_window)
    X_test, y_test = create_sliding_windows(scaled_test_data, input_window, output_window)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if len(test_dataset) > 0 else None
    
    return train_loader, test_loader, power_scaler, X_train.shape[2]


def train_model(model, train_loader, optimizer, scheduler=None, num_epochs=200, test_dataloader=None, power_scaler=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    
    model.train()
    avg_loss = 0
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if scheduler:
            scheduler.step()
            
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 ==   0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}')
            if test_dataloader:
                mse, mae, preds, labels = evaluate_model(model, test_dataloader)
                model.train()
                preds_inv = power_scaler.inverse_transform(preds)
                labels_inv = power_scaler.inverse_transform(labels) 
                
                mse_inv = np.mean((preds_inv - labels_inv)**2)
                mae_inv = np.mean(np.abs(preds_inv - labels_inv))
                print(f"Test MSE: {mse_inv:.4f}, Test MAE: {mae_inv:.4f}")
            
        
    return model, avg_loss

def evaluate_model(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    mse = np.mean((preds - labels)**2)
    mae = np.mean(np.abs(preds - labels))
    return mse, mae, preds, labels

def run_training_session(horizon):
    # --- 1. Load and Prepare Data ---
    input_window = 90
    output_window = horizon
    train_loader, test_loader, power_scaler, input_dim = get_data_loaders(input_window, output_window)

    results = {}

    for model_name in ["HTFN", "LSTM", "Transformer"]:
        print(f"--- Training {model_name} for {horizon}-day horizon ---")
        
        results[model_name] = {"MSE": [], "MAE": []}
        
        for i in range(5): # 5 runs
            scheduler = None
            if model_name == "LSTM":
                run_model = LSTMModel(input_dim=input_dim, hidden_dim=128, num_layers=3, output_dim=output_window)
                optimizer = torch.optim.Adam(run_model.parameters(), lr=0.0001)
            elif model_name == "Transformer":
                run_model = TransformerModel(input_dim=input_dim, d_model=128, nhead=4, num_encoder_layers=3, dim_feedforward=512, output_dim=output_window)
                optimizer = torch.optim.Adam(run_model.parameters(), lr=0.0001)
            elif model_name == "HTFN":
                run_model = HTFN(
                    input_dim=input_dim,
                    output_dim=output_window,
                    short_kernel_size=3,
                    mid_kernel_size=7,
                    long_kernel_size=15,
                    gru_hidden_dim=128,
                    gru_num_layers=3,
                    cross_attention_heads=2,
                    transformer_d_model=64,
                    transformer_nhead=4,
                    transformer_num_layers=4,
                    transformer_dim_feedforward=1024,
                    dropout=0.2
                )
                optimizer = torch.optim.AdamW(run_model.parameters(), lr=5e-3, weight_decay=1e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)

            print(f"Run {i+1}/5")
            trained_model, _ = train_model(run_model, train_loader, optimizer, scheduler=scheduler, test_dataloader=test_loader, power_scaler=power_scaler)
            
            if test_loader:
                mse, mae, preds, labels = evaluate_model(trained_model, test_loader)
            else:
                print("Test loader is empty, skipping evaluation.")
                mse, mae = float('nan'), float('nan')
            
            preds_inv = power_scaler.inverse_transform(preds)
            labels_inv = power_scaler.inverse_transform(labels) 
            
            mse_inv = np.mean((preds_inv - labels_inv)**2)
            mae_inv = np.mean(np.abs(preds_inv - labels_inv))
            # print(f"MSE: {mse_inv:.4f}, MAE: {mae_inv:.4f}")
            if i == 4 and test_loader: # Save results from the last run for plotting
                # --- Save Model ---
                os.makedirs('models', exist_ok=True)
                torch.save(trained_model.state_dict(), f'models/{model_name}_horizon_{horizon}.pth')

                # --- Save Test Set Predictions ---
                os.makedirs('results', exist_ok=True)
                
                # # Inverse transform predictions and labels
                # preds_inv = power_scaler.inverse_transform(preds)
                # labels_inv = power_scaler.inverse_transform(labels)

                np.save(f'results/{model_name}_preds_horizon_{horizon}.npy', preds_inv)
                np.save(f'results/{model_name}_labels_horizon_{horizon}.npy', labels_inv)

                # --- Evaluate and Save Training Set Predictions for Fit Analysis ---
                _, _, train_preds, train_labels = evaluate_model(trained_model, train_loader)
                train_preds_inv = power_scaler.inverse_transform(train_preds)
                train_labels_inv = power_scaler.inverse_transform(train_labels)
                np.save(f'results/{model_name}_train_preds_horizon_{horizon}.npy', train_preds_inv)
                np.save(f'results/{model_name}_train_labels_horizon_{horizon}.npy', train_labels_inv)


            results[model_name]["MSE"].append(float(mse_inv))
            results[model_name]["MAE"].append(float(mae_inv))
        results[model_name]["MSE_std"] = float(np.std(results[model_name]["MSE"]))
        results[model_name]["MAE_std"] = float(np.std(results[model_name]["MAE"]))
        results[model_name]["MSE_mean"] = float(np.mean(results[model_name]["MSE"]))
        results[model_name]["MAE_mean"] = float(np.mean(results[model_name]["MAE"]))
        
        print(f"Results for {model_name} (Horizon: {horizon}):")
        print(f"  MSE: {results[model_name]['MSE_mean']:.4f} (+/- {results[model_name]['MSE_std']:.4f})")
        print(f"  MAE: {results[model_name]['MAE_mean']:.4f} (+/- {results[model_name]['MAE_std']:.4f})\n")

    return results


def main():
    seed_everything()
    print("="*20)
    print("Running for 90-day horizon")
    print("="*20)
    results_90 = run_training_session(horizon=90)
    
    print("\n" + "="*20)
    print("Running for 365-day horizon")
    print("="*20)
    results_365 = run_training_session(horizon=365)

    with open('results/results_90.json', 'w') as f:
        json.dump(results_90, f)

    with open('results/results_365.json', 'w') as f:
        json.dump(results_365, f)

if __name__ == '__main__':
    main() 