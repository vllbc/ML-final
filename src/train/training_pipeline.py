import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import os
import random

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
        output_slice = data[i + input_window_size:i + input_window_size + output_window_size, 0]
        X.append(input_slice)
        y.append(output_slice)
    return np.array(X), np.array(y)

def get_data_loaders(input_window, output_window, train_path, test_path, batch_size):
    train_df = pd.read_csv(train_path, index_col='DateTime', parse_dates=True)
    test_df = pd.read_csv(test_path, index_col='DateTime', parse_dates=True)
    
    scaler = MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train_df)
    
    scaled_test_data = scaler.transform(test_df) if not test_df.empty else np.array([])

    power_scaler = MinMaxScaler()
    power_scaler.fit(train_df[['Global_active_power']])
    
    X_train, y_train = create_sliding_windows(scaled_train_data, input_window, output_window)
    X_test, y_test = create_sliding_windows(scaled_test_data, input_window, output_window)
    
    train_dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if len(test_dataset) > 0 else None
    
    return train_loader, test_loader, power_scaler, X_train.shape[2]


def train_model(model, train_loader, optimizer, scheduler=None, test_dataloader=None, power_scaler=None,model_name=None, warmup_epochs=0, peak_lr=None, initial_lr=0, clip_grad_norm=None, num_epochs=200, eval_interval=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    
    if not (model_name == "HTFN"):
        warmup_epochs = 0
        peak_lr = None
        initial_lr = 0
        clip_grad_norm = None

    model.train()
    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            lr = initial_lr + (peak_lr - initial_lr) * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            if clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
                
            optimizer.step()
            total_loss += loss.item()
        
        if scheduler and epoch >= warmup_epochs:
            scheduler.step()
            
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % eval_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')
            if test_dataloader:
                mse_inv, mae_inv = evaluate_model(model, test_dataloader, power_scaler)
                print(f"Test MSE: {mse_inv:.4f}, Test MAE: {mae_inv:.4f}")
                model.train()
            
    return model

def evaluate_model(model, data_loader, power_scaler, return_preds=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    preds_inv = power_scaler.inverse_transform(preds)
    labels_inv = power_scaler.inverse_transform(labels) 
    
    mse_inv = np.mean((preds_inv - labels_inv)**2)
    mae_inv = np.mean(np.abs(preds_inv - labels_inv))

    if return_preds:
        return mse_inv, mae_inv, preds_inv, labels_inv
        
    return mse_inv, mae_inv

def save_results(model_name, horizon, trained_model, train_loader, test_loader, power_scaler):
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    torch.save(trained_model.state_dict(), f'models/{model_name}_horizon_{horizon}.pth')

    _, _, test_preds_inv, test_labels_inv = evaluate_model(trained_model, test_loader, power_scaler, return_preds=True)
    np.save(f'results/{model_name}_preds_horizon_{horizon}.npy', test_preds_inv)
    np.save(f'results/{model_name}_labels_horizon_{horizon}.npy', test_labels_inv)

    _, _, train_preds_inv, train_labels_inv = evaluate_model(trained_model, train_loader, power_scaler, return_preds=True)
    np.save(f'results/{model_name}_train_preds_horizon_{horizon}.npy', train_preds_inv)
    np.save(f'results/{model_name}_train_labels_horizon_{horizon}.npy', train_labels_inv)
    
    print(f"\nSaved model and predictions for {model_name} with horizon {horizon}.") 