import hydra
from omegaconf import DictConfig
import torch
from hydra.utils import instantiate
import numpy as np
import json

from src.train.training_pipeline import (
    seed_everything,
    get_data_loaders,
    train_model,
    evaluate_model,
    save_results,
)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.trainer.seed:
        seed_everything(cfg.trainer.seed)

    model_name = cfg.model._target_.split(".")[-1]
    print(f"--- trainer {model_name} for {cfg.data.output_window_size}-day horizon ---")

    train_loader, test_loader, power_scaler, input_dim = get_data_loaders(
        input_window=cfg.data.input_window_size,
        output_window=cfg.data.output_window_size,
        train_path=cfg.data.train_path,
        test_path=cfg.data.test_path,
        batch_size=cfg.data.batch_size,
    )

    cfg.model.input_dim = input_dim
    cfg.model.output_dim = cfg.data.output_window_size
    results = {"mse": [], "mae": []}
    for i in range(cfg.trainer.num_experiments):
        print(f"--- Experiment {i + 1}/{cfg.trainer.num_experiments} ---")
        model = instantiate(cfg.model)

        if model_name == "HTFN" and "htfn_optimizer" in cfg.trainer:
            print("Using special AdamW optimizer for HTFN.")
            optimizer = instantiate(
                cfg.trainer.htfn_optimizer, params=model.parameters()
            )
            scheduler = instantiate(cfg.trainer.scheduler, optimizer=optimizer)
        else:
            optimizer = instantiate(cfg.trainer.optimizer, params=model.parameters())
            scheduler = None

        trained_model = train_model(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            test_dataloader=test_loader,
            power_scaler=power_scaler,
            model_name=model_name,
            warmup_epochs=cfg.trainer.warmup_epochs,
            peak_lr=cfg.trainer.peak_lr,
            initial_lr=cfg.trainer.initial_lr,
            clip_grad_norm=cfg.trainer.clip_grad_norm,
            num_epochs=cfg.trainer.num_epochs,
            eval_interval=cfg.trainer.eval_interval,
        )

        print("\n--- Final Evaluation ---")
        final_mse, final_mae = evaluate_model(trained_model, test_loader, power_scaler)
        print(f"Final Test MSE: {final_mse:.4f}, Final Test MAE: {final_mae:.4f}")

        save_results(
            model_name=model_name,
            horizon=cfg.data.output_window_size,
            trained_model=trained_model,
            train_loader=train_loader,
            test_loader=test_loader,
            power_scaler=power_scaler,
        )
        results["mse"].append(float(final_mse))
        results["mae"].append(float(final_mae))

    results["mse_mean"] = float(np.mean(results["mse"]))
    results["mae_mean"] = float(np.mean(results["mae"]))
    results["mse_std"] = float(np.std(results["mse"]))
    results["mae_std"] = float(np.std(results["mae"]))

    with open(
        f"results/{model_name}_{cfg.data.output_window_size}_results.json", "w"
    ) as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
