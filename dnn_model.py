from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from feature_pipeline import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    FeatureConfig,
    extract_raw_features,
    fit_transform_feature_fold,
)
from ga_utils import (
    CV_FOLDS,
    GA_CROSSOVER_RATE,
    GA_GENERATIONS,
    GA_MUTATION_RATE,
    GA_POP_SIZE,
    prepare_descriptor_cv_folds,
    run_discrete_ga,
)
from model_utils import finalize_model, regression_metrics

TRAIN_PATH = Path(DEFAULT_TRAIN_PATH)
TEST_PATH = Path(DEFAULT_TEST_PATH)
OUTPUT_DIR = Path(__file__).resolve().parent / 'outputs' / 'dnn'

RANDOM_STATE = 42
MAX_EPOCHS = 250
PATIENCE = 25
MIN_DELTA = 1e-4

GA_SEARCH_SPACE: dict[str, tuple] = {
    'hidden_dim': (128, 256, 512),
    'num_layers': (2, 3, 4),
    'dropout': (0.0, 0.1, 0.2, 0.3),
    'learning_rate': (3e-4, 5e-4, 1e-3, 2e-3),
    'weight_decay': (0.0, 1e-5, 1e-4, 1e-3),
    'batch_size': (16, 32, 64),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DescriptorMLP(nn.Module):
    """全连接回归网络：L 层 (Linear -> ReLU -> Dropout) + 线性回归头。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            blocks.append(nn.Linear(in_dim, hidden_dim))
            blocks.append(nn.ReLU())
            if dropout > 0.0:
                blocks.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).view(-1)


def prepare_scaled_fold_tensors(folds: list[dict], device: torch.device) -> list[dict]:
    scaled = []
    for f in folds:
        # 特征已在折内完成 Z-score 标准化；此处仅做标签折内标准化
        X_tr = f['X_train'].astype(np.float32)
        X_va = f['X_val'].astype(np.float32)
        y_mean = float(np.mean(f['y_train']))
        y_std = float(np.std(f['y_train'], ddof=0))
        y_std = y_std if y_std > 1e-12 else 1.0
        y_tr_norm = ((f['y_train'] - y_mean) / y_std).astype(np.float32)
        scaled.append({
            'fold': f['fold'],
            'X_tr_t': torch.from_numpy(X_tr).to(device),
            'y_tr_norm_t': torch.from_numpy(y_tr_norm).to(device),
            'X_va_t': torch.from_numpy(X_va).to(device),
            'y_va_t': torch.from_numpy(f['y_val'].astype(np.float64)).to(device),
            'y_va': f['y_val'],
            'y_mean': y_mean,
            'y_std': y_std,
            'input_dim': X_tr.shape[1],
        })
    return scaled


def train_dnn_fold(
    fold_t: dict, params: dict, device: torch.device, return_predictions: bool = False,
) -> dict:
    fold_seed = RANDOM_STATE + fold_t['fold'] * 101
    seed_everything(fold_seed)
    rng = torch.Generator(device=device)
    rng.manual_seed(fold_seed)

    model = DescriptorMLP(
        input_dim=fold_t['input_dim'],
        hidden_dim=int(params['hidden_dim']),
        num_layers=int(params['num_layers']),
        dropout=float(params['dropout']),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params['learning_rate']),
        weight_decay=float(params['weight_decay']),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    X_tr = fold_t['X_tr_t']
    y_tr = fold_t['y_tr_norm_t']
    n_tr = X_tr.size(0)
    bs = int(params['batch_size'])

    best_rmse = math.inf
    best_epoch = 1
    best_state = None
    stalled = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_tr, generator=rng, device=device)
        for start in range(0, n_tr, bs):
            idx = perm[start:start + bs]
            optimizer.zero_grad(set_to_none=True)
            pred = model(X_tr[idx])
            loss = F.mse_loss(pred, y_tr[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.inference_mode():
            va_pred_c = (
                model(fold_t['X_va_t']).to(torch.float64) * fold_t['y_std']
                + fold_t['y_mean']
            )
            val_rmse = float(torch.sqrt(torch.mean((va_pred_c - fold_t['y_va_t']) ** 2)).item())
        if not math.isfinite(val_rmse):
            return {'best_rmse': 1e6, 'best_epoch': epoch}
        if val_rmse < best_rmse - MIN_DELTA:
            best_rmse = val_rmse
            best_epoch = epoch
            if return_predictions:
                best_state = copy.deepcopy(model.state_dict())
            stalled = 0
        else:
            stalled += 1
        if stalled >= PATIENCE:
            break

    out = {'best_rmse': best_rmse, 'best_epoch': best_epoch}
    if return_predictions and best_state is not None:
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            pred = (
                model(fold_t['X_va_t']).to(torch.float64) * fold_t['y_std']
                + fold_t['y_mean']
            ).cpu().numpy()
        out['metrics'] = regression_metrics(fold_t['y_va'], pred)
    return out


def main() -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'DNN 使用设备：{device}')
    seed_everything(RANDOM_STATE)

    cfg = FeatureConfig(random_state=RANDOM_STATE)
    raw = extract_raw_features(TRAIN_PATH, TEST_PATH, cfg)
    folds = prepare_descriptor_cv_folds(raw, cfg, n_splits=CV_FOLDS, seed=RANDOM_STATE)
    fold_tensors = prepare_scaled_fold_tensors(folds, device)

    def eval_fn(params: dict) -> dict:
        fold_rmses, fold_epochs = [], []
        for ft in fold_tensors:
            res = train_dnn_fold(ft, params, device, return_predictions=False)
            fold_rmses.append(res['best_rmse'])
            fold_epochs.append(res['best_epoch'])
        return {
            'mean_cv_rmse': float(np.mean(fold_rmses)),
            'fold_rmses': fold_rmses,
            'extra': {'fold_epochs': fold_epochs},
        }

    best_params, _, ga_evals_df = run_discrete_ga(
        model_tag='DNN',
        search_space=GA_SEARCH_SPACE,
        eval_fn=eval_fn,
        output_dir=OUTPUT_DIR,
        pop_size=GA_POP_SIZE,
        generations=GA_GENERATIONS,
        crossover_rate=GA_CROSSOVER_RATE,
        mutation_rate=GA_MUTATION_RATE,
        seed=RANDOM_STATE,
    )
    print(f'\nDNN GA 最优超参数：{json.dumps(best_params, ensure_ascii=False)}')

    cv_rows = []
    best_epochs = []
    for ft in fold_tensors:
        res = train_dnn_fold(ft, best_params, device, return_predictions=True)
        m = res['metrics']
        best_epochs.append(res['best_epoch'])
        cv_rows.append({
            'fold': ft['fold'] + 1,
            'best_epoch': res['best_epoch'],
            'val_R2': m['R2'],
            'val_RMSE_C': m['RMSE_C'],
            'val_MAE_C': m['MAE_C'],
        })
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(OUTPUT_DIR / 'cv_best_fold_metrics.csv', index=False, encoding='utf-8-sig')
    retrain_epochs = max(1, int(round(float(np.mean(best_epochs)))))
    print(f'5 折平均：R²={cv_df["val_R2"].mean():.4f}，'
          f'RMSE={cv_df["val_RMSE_C"].mean():.3f} ℃，重训轮数={retrain_epochs}')

    # 在完整训练集上重新拟合特征筛选与 Z-score 标准化，并重训
    all_train_idx = np.arange(len(raw.y_train))
    final_features = fit_transform_feature_fold(
        raw, train_idx=all_train_idx, config=cfg, use_test=True, verbose=True
    )
    X_train = final_features.X_train.astype(np.float32)
    X_test = final_features.X_test.astype(np.float32)
    y_mean = float(np.mean(final_features.y_train))
    y_std = float(np.std(final_features.y_train, ddof=0))
    y_std = y_std if y_std > 1e-12 else 1.0
    y_train_norm = ((final_features.y_train - y_mean) / y_std).astype(np.float32)

    seed_everything(RANDOM_STATE)
    rng = torch.Generator(device=device)
    rng.manual_seed(RANDOM_STATE)
    model = DescriptorMLP(
        input_dim=X_train.shape[1],
        hidden_dim=int(best_params['hidden_dim']),
        num_layers=int(best_params['num_layers']),
        dropout=float(best_params['dropout']),
    ).to(device)
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(best_params['learning_rate']),
        weight_decay=float(best_params['weight_decay']),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(retrain_epochs, 1)
    )
    X_tr_t = torch.from_numpy(X_train).to(device)
    y_tr_t = torch.from_numpy(y_train_norm).to(device)
    X_te_t = torch.from_numpy(X_test).to(device)
    bs = int(best_params['batch_size'])

    history = []
    for epoch in range(1, retrain_epochs + 1):
        model.train()
        perm = torch.randperm(X_tr_t.size(0), generator=rng, device=device)
        total_loss = torch.zeros((), device=device)
        for start in range(0, X_tr_t.size(0), bs):
            idx = perm[start:start + bs]
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.detach() * idx.numel()
        scheduler.step()
        history.append({
            'epoch': epoch,
            'train_mse_standardized': float((total_loss / X_tr_t.size(0)).item()),
            'lr': float(optimizer.param_groups[0]['lr']),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / 'training_history.csv', index=False, encoding='utf-8-sig'
    )

    model.eval()
    with torch.inference_mode():
        train_pred = (model(X_tr_t).to(torch.float64) * y_std + y_mean).cpu().numpy()
        test_pred = (model(X_te_t).to(torch.float64) * y_std + y_mean).cpu().numpy()

    torch.save({
        'model_class': 'DescriptorMLP',
        'model_state_dict': model.state_dict(),
        'input_dim': int(X_train.shape[1]),
        'hyperparameters': best_params,
        'retrain_epochs': retrain_epochs,
        'trainable_params': trainable_params,
        'y_scaler': {'mean': y_mean, 'std': y_std},
        'random_state': RANDOM_STATE,
    }, OUTPUT_DIR / 'dnn_model.pt')

    finalize_model(
        'DNN', OUTPUT_DIR,
        final_features.y_train, train_pred,
        final_features.y_test, test_pred,
        final_features,
        extra_metadata={
            'ga_config': {
                'cv_folds': CV_FOLDS,
                'population_size': GA_POP_SIZE,
                'generations': GA_GENERATIONS,
                'crossover_rate': GA_CROSSOVER_RATE,
                'mutation_rate': GA_MUTATION_RATE,
                'fitness_metric': '5-fold CV mean RMSE (C)',
                'unique_configurations_evaluated': int(len(ga_evals_df)),
            },
            'ga_search_space': {k: list(v) for k, v in GA_SEARCH_SPACE.items()},
            'hyperparameters': best_params,
            'trainable_params': trainable_params,
            'activation': 'ReLU',
            'loss': 'MSE',
            'optimizer': 'AdamW',
            'retrain_epochs': retrain_epochs,
            'max_epochs': MAX_EPOCHS,
            'patience': PATIENCE,
            'cv_5fold_summary': {
                'mean_R2': float(cv_df['val_R2'].mean()),
                'mean_RMSE_C': float(cv_df['val_RMSE_C'].mean()),
                'mean_MAE_C': float(cv_df['val_MAE_C'].mean()),
                'folds': cv_rows,
            },
            'test_used_for_optimization': False,
            'random_state': RANDOM_STATE,
            'device': str(device),
        },
    )


if __name__ == '__main__':
    main()
