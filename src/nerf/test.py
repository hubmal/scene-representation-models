import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
import torch.nn.functional as F
import pandas as pd
import numpy as np
import warnings
import gc

from tqdm import tqdm
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

from thop import profile

from angio_3dmatch.model.ensemble import GeoCoroEnsemble
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from torchmetrics.functional.classification import multiclass_stat_scores
from torchmetrics.functional.classification import binary_auroc


def _register_safe_globals_for_torch_load():
    try:
        from monai.utils.enums import TraceKeys
        from angio_3dmatch.model.modular_lightning import GeoCoroConfig
        torch.serialization.add_safe_globals([TraceKeys, GeoCoroConfig])
    except Exception:
        pass


def test(model, dm, log_name, log_version, best_path=None):
    torch.set_float32_matmul_precision('medium')
    model.eval()
    _register_safe_globals_for_torch_load()
    logger = CSVLogger(
        save_dir="logs", 
        name=log_name, 
        version=log_version
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_true",
        enable_progress_bar=True,
        logger=logger
    )
    trainer.test(model, datamodule=dm)



warnings.filterwarnings(
    "ignore", 
    message="the ground truth of class .* is all 0", 
    category=UserWarning
)
warnings.filterwarnings(
    "ignore", 
    message="the prediction class .* is all 0", 
    category=UserWarning
)

IMPORTANCE_GROUPS = {
    "f1_critical": [9, 10, 1, 17],
    "f1_high": [11, 2, 13, 18, 21, 22, 3],
    "f1_low": [4, 5, 6, 7, 8, 12, 14, 15, 16, 19, 20, 23, 24, 25]
}

def compute_95_ci(data, n_bootstraps=1000):
    data = np.array(data)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return np.nan, np.nan
    
    bootstrapped_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstraps)]
    mean_val = np.mean(data)
    ci_margin = 1.96 * np.std(bootstrapped_means)
    return mean_val, ci_margin

def compute_diameter_error(pred_mask, gt_mask):
    pred_b, gt_b = pred_mask > 0.5, gt_mask > 0.5
    if not np.any(gt_b): return np.nan
    skeleton = skeletonize(gt_b)
    if not np.any(skeleton): return np.nan
    gt_radius_map = distance_transform_edt(gt_b)
    pred_radius_map = distance_transform_edt(pred_b)
    gt_diameters = 2 * gt_radius_map[skeleton]
    pred_diameters = 2 * pred_radius_map[skeleton]
    return np.mean(np.abs(gt_diameters - pred_diameters))

def compute_cldice(pred_mask, gt_mask):
    pred_b, gt_b = pred_mask > 0.5, gt_mask > 0.5
    if not np.any(gt_b) or not np.any(pred_b): 
        return np.nan
    skel_pred = skeletonize(pred_b)
    skel_gt = skeletonize(gt_b)
    t_prec = np.sum(skel_pred & gt_b) / (np.sum(skel_pred) + 1e-8)
    t_sens = np.sum(skel_gt & pred_b) / (np.sum(skel_gt) + 1e-8)

    if t_prec + t_sens == 0: 
        return 0.0

    return 2 * (t_prec * t_sens) / (t_prec + t_sens)

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def evaluate_checkpoints(checkpoints, dm, model_class, num_classes=27, device="cuda", mc_passes=5, ensemble_voting_mode="soft_voting", suffix=""):
    dm.setup(stage="test")
    dl_test = dm.test_dataloader()
    
    dice_metric = DiceMetric(include_background=False, reduction="none")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")

    all_aggregate_results = []

    for task_name, ckpt_path in checkpoints:
        is_ensemble = isinstance(ckpt_path, (list, tuple))

        if is_ensemble:
          model = GeoCoroEnsemble(ckpt_path, model_class, device=device)
        else:
            model = model_class.load_from_checkpoint(ckpt_path, map_location="cpu")
            model.to(device)
        model.eval()

        sample_results = []
        flops_calculated = False
        
        gflops = np.nan
        params_M = np.nan

        with torch.no_grad():
            for batch in tqdm(dl_test, desc="Testing Samples"):
                batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                dir_paths = batch["dir_path"] 
                base_model = model
                x, y = base_model._prepare_input(batch)

                if profile is not None and not flops_calculated:
                    macs, params = profile(base_model, inputs=(x,), verbose=False)
                    gflops = (macs * 2) / 1e9
                    params_M = params / 1e6
                    print(f"\nModel Complexity -> Params: {params_M:.2f}M | GFLOPs: {gflops:.2f}\n")
                    flops_calculated = True

                model.eval()
                if is_ensemble:
                    preds = model(x, mode=ensemble_voting_mode)
                else:
                    logits = model(x)
                    if isinstance(logits, tuple): logits = logits[0]
                    
                    preds = F.softmax(logits, dim=1)
                pred_indices = torch.argmax(preds, dim=1)
                y_indices = torch.argmax(y, dim=1)
                
                preds_onehot = F.one_hot(pred_indices, num_classes=num_classes).permute(0, 3, 1, 2).float()
                
                dice_batch = dice_metric(y_pred=preds_onehot, y=y) 
                hd95_batch = hd95_metric(y_pred=preds_onehot, y=y) 

                enable_dropout(model)
                mc_preds_list = []
                for _ in range(mc_passes):
                    mc_logits = model(x)
                    if isinstance(mc_logits, tuple): mc_logits = mc_logits[0]
                    mc_preds_list.append(F.softmax(mc_logits, dim=1).cpu())
                mean_mc_preds = torch.stack(mc_preds_list, dim=0).mean(dim=0).to(device)                
                uncertainty_map = -torch.sum(mean_mc_preds * torch.log(mean_mc_preds + 1e-8), dim=1)
                error_map = (pred_indices != y_indices).long()

                
                for i in range(x.size(0)):
                    sample_id = Path(dir_paths[i]).name
                    
                    sample_dice = dice_batch[i].cpu().numpy()
                    sample_hd95 = hd95_batch[i].cpu().numpy()
                    
                    fg_mean_dice = np.nanmean(sample_dice) 
                    fg_mean_hd95 = np.nanmean(sample_hd95)

                    s_uq = uncertainty_map[i].flatten()
                    auc_uq_classes = []
                    
                    for cls_idx in range(1, num_classes):
                        s_err_c = ((pred_indices[i] == cls_idx) != (y_indices[i] == cls_idx)).flatten().long()
                        if torch.sum(s_err_c) > 0 and torch.sum(1 - s_err_c) > 0:
                            auc_uq_classes.append(binary_auroc(s_uq, s_err_c).item())
                        else:
                            auc_uq_classes.append(np.nan)
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        auc_uq = np.nanmean(auc_uq_classes)

                    stats = multiclass_stat_scores(
                        pred_indices[i:i+1], 
                        y_indices[i:i+1], 
                        num_classes=num_classes, 
                        average=None
                    )
                    tp, fp, fn = stats[:, 0].float(), stats[:, 1].float(), stats[:, 3].float()
                    denominator = (2 * tp) + fp + fn
                    
                    sample_f1_all_classes = torch.where(
                        denominator > 0, 
                        (2 * tp) / denominator, 
                        torch.tensor(float('nan'), device=device)
                    ).cpu().numpy()
                    
                    fg_mean_f1 = np.nanmean(sample_f1_all_classes[1:])

                    sample_pred_np = preds_onehot[i].cpu().numpy() 
                    sample_gt_np = y[i].cpu().numpy()              
                    
                    fg_pred_mask = np.sum(sample_pred_np[1:], axis=0) > 0
                    fg_gt_mask = np.sum(sample_gt_np[1:], axis=0) > 0
                    
                    dia_error = compute_diameter_error(fg_pred_mask, fg_gt_mask)
                    cldice_score = compute_cldice(fg_pred_mask, fg_gt_mask)


                    importance_metrics = {}
                    for group_name, indices in IMPORTANCE_GROUPS.items():
                        group_f1s = [sample_f1_all_classes[idx] for idx in indices]
                        importance_metrics[group_name] = np.nanmean(group_f1s)


                    row = {
                        "task": task_name,
                        "sample_id": sample_id,
                        "dir_path": dir_paths[i],
                        "fg_mean_f1": fg_mean_f1,
                        "fg_mean_dice": fg_mean_dice,
                        "fg_mean_hd95": fg_mean_hd95,
                        "diameter_error": dia_error,
                        "cldice": cldice_score,
                        "auc_uq": auc_uq,
                        "gflops": gflops,
                        "params_M": params_M,
                        **importance_metrics
                    }
                    
                    for cls_idx in range(1, num_classes):
                        row[f"class_{cls_idx}_f1"] = sample_f1_all_classes[cls_idx]
                        row[f"class_{cls_idx}_dice"] = sample_dice[cls_idx - 1]
                        row[f"class_{cls_idx}_hd95"] = sample_hd95[cls_idx - 1] 

                    sample_results.append(row)
        
        dice_metric.reset()
        hd95_metric.reset()

        df_task = pd.DataFrame(sample_results)
        task_csv_name = f"results_{task_name.replace('+', '').replace('/', '_')}.csv"
        df_task.to_csv(task_csv_name, index=False)
        print(f"Saved sample metrics to {task_csv_name}")

        # Add static complexity metrics to the aggregate row
        agg_row = {
            "task": task_name,
            "gflops": gflops,
            "params_M": params_M
        }
        target_metrics = [
            "fg_mean_f1", "f1_critical", "f1_high", "f1_low", "class_26_f1",
            "fg_mean_dice", "fg_mean_hd95", "diameter_error", "cldice", "auc_uq"
        ]
        for metric in target_metrics:
            mean_val, ci_val = compute_95_ci(df_task[metric].dropna())
            agg_row[f"{metric}_mean"] = mean_val
            agg_row[f"{metric}_ci95"] = ci_val
            
        all_aggregate_results.append(agg_row)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    df_agg = pd.DataFrame(all_aggregate_results)
    df_agg.to_csv(f"master_aggregate_results{suffix}.csv", index=False)
    print("\nSaved master aggregate metrics to master_aggregate_results.csv")
    print(df_agg[["task", "gflops", "params_M", "fg_mean_dice_mean", "cldice_mean"]].to_string(index=False))