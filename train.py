"""
CoVaL Training Script (simplified)
- Siamese VMamba encoder + LCVD + VPL
- CE + Lovasz + Progressive + Edge + Commonality loss
- AdamW + CosineAnnealingLR + AMP
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from configs.config import get_config
from datasets.make_data_loader import make_data_loader
from utils.metrics import Evaluator
from models.coval import CoVaLModel
from losses.edge_loss import EdgeLoss
import losses.lovasz_loss as lovasz_loss


def main():
    args = _parse_args()
    config = get_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Seeds ──
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # ── Model ──
    cfg = config.MODEL.VSSM
    model = CoVaLModel(
        pretrained=args.pretrained_weight_path,
        decoder_output_stride=args.decoder_output_stride,
        decoder_dims=args.decoder_dims,
        edge_stages=args.edge_stages,
        use_commonality=args.commonality_loss_weight > 0.0,
        patch_size=cfg.PATCH_SIZE, in_chans=cfg.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        depths=cfg.DEPTHS, dims=cfg.EMBED_DIM,
        ssm_d_state=cfg.SSM_D_STATE, ssm_ratio=cfg.SSM_RATIO,
        ssm_rank_ratio=cfg.SSM_RANK_RATIO,
        ssm_dt_rank=("auto" if cfg.SSM_DT_RANK == "auto" else int(cfg.SSM_DT_RANK)),
        ssm_act_layer=cfg.SSM_ACT_LAYER, ssm_conv=cfg.SSM_CONV,
        ssm_conv_bias=cfg.SSM_CONV_BIAS, ssm_drop_rate=cfg.SSM_DROP_RATE,
        ssm_init=cfg.SSM_INIT, forward_type=cfg.SSM_FORWARDTYPE,
        mlp_ratio=cfg.MLP_RATIO, mlp_act_layer=cfg.MLP_ACT_LAYER,
        mlp_drop_rate=cfg.MLP_DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        patch_norm=cfg.PATCH_NORM, norm_layer=cfg.NORM_LAYER,
        downsample_version=cfg.DOWNSAMPLE, patchembed_version=cfg.PATCHEMBED,
        gmlp=cfg.GMLP, use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total / 1e6:.2f}M")

    # ── Data ──
    def _load_list(root, name):
        return (Path(root) / "list" / name).read_text().splitlines()

    args.train_dataset_path = args.dataset_path
    args.train_data_name_list = _load_list(args.dataset_path, "train.txt")
    args.shuffle = True
    args.type = "train"
    train_loader = make_data_loader(args)

    val_name = "val.txt" if args.validation_split_mode == "val" else "test.txt"
    import copy
    val_args = copy.deepcopy(args)
    val_args.train_data_name_list = _load_list(args.dataset_path, val_name)
    val_args.shuffle = False
    val_args.type = "val"
    val_args.max_iters = None
    val_args.batch_size = args.val_batch_size or args.batch_size
    val_loader = make_data_loader(val_args)

    # ── Optimizer ──
    enc_params, dec_params = [], []
    for name, param in model.named_parameters():
        (enc_params if "encoder" in name else dec_params).append(param)

    optimiser = optim.AdamW([
        {"params": enc_params, "lr": args.learning_rate * args.encoder_lr_ratio},
        {"params": dec_params, "lr": args.learning_rate},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.max_iters, eta_min=1e-6)

    # ── Losses ──
    edge_criterion = EdgeLoss(bce_weight=1.0, dice_weight=1.0, pos_weight=10.0)
    evaluator = Evaluator(num_class=2)
    scaler = GradScaler(enabled=args.use_amp)

    # ── Training ──
    best_f1, step = 0.0, 0
    save_dir = Path(args.model_param_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    while step < args.max_iters:
        model.train()
        for pre_img, post_img, labels, edge_labels, _ in train_loader:
            if step >= args.max_iters:
                break
            step += 1

            pre_img = pre_img.to(device, dtype=torch.float32)
            post_img = post_img.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            edge_labels = edge_labels.to(device, dtype=torch.float32)
            if pre_img.dim() == 3:
                pre_img = pre_img.unsqueeze(0)
                post_img = post_img.unsqueeze(0)
                labels = labels.unsqueeze(0)
                edge_labels = edge_labels.unsqueeze(0)

            optimiser.zero_grad()
            with autocast(enabled=args.use_amp):
                outputs = model(pre_img, post_img, return_aux=True, return_edge=True)
                preds = outputs["change"]

                ce = F.cross_entropy(preds, labels, ignore_index=255)
                lovasz = lovasz_loss.lovasz_softmax(F.softmax(preds, dim=1), labels, ignore=255)
                change_loss = args.ce_weight * ce + args.lovasz_weight * lovasz

                progressive_loss = torch.tensor(0.0, device=device)
                progressive_preds = outputs.get("progressive")
                if progressive_preds is not None:
                    for pp in progressive_preds:
                        if pp.shape[-2:] != labels.shape[-2:]:
                            pp = F.interpolate(pp, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                        progressive_loss += (
                            args.ce_weight * F.cross_entropy(pp, labels, ignore_index=255)
                            + args.lovasz_weight * lovasz_loss.lovasz_softmax(F.softmax(pp, dim=1), labels, ignore=255))
                    progressive_loss = progressive_loss / len(progressive_preds)

                if step <= 8000:       ew = 1.0
                elif step <= 15000:    ew = 0.3
                elif step <= 25000:    ew = 0.1
                else:                  ew = 0.05
                ew = ew * (args.edge_loss_weight or 0.0)

                edge_pred = outputs.get("edge")
                edge_loss = edge_criterion(edge_pred, edge_labels.unsqueeze(1)) if edge_pred is not None else torch.tensor(0.0, device=device)

                commonality_loss = torch.tensor(0.0, device=device)
                commonality_preds = outputs.get("commonality")
                if commonality_preds is not None and args.commonality_loss_weight > 0.0:
                    inverse_labels = labels.clone()
                    valid = inverse_labels != 255
                    inverse_labels[valid] = 1 - inverse_labels[valid]
                    for cp in commonality_preds:
                        if cp.shape[-2:] != labels.shape[-2:]:
                            cp = F.interpolate(cp, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                        commonality_loss += (
                            args.ce_weight * F.cross_entropy(cp, inverse_labels, ignore_index=255)
                            + args.lovasz_weight * lovasz_loss.lovasz_softmax(F.softmax(cp, dim=1), inverse_labels, ignore=255))
                    commonality_loss = commonality_loss / len(commonality_preds)

                loss = (change_loss
                        + args.progressive_loss_weight * progressive_loss
                        + ew * edge_loss
                        + args.commonality_loss_weight * commonality_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()

            if step % args.log_interval == 0:
                print(f"[{step}/{args.max_iters}] loss={loss.item():.4f} "
                      f"loc={change_loss.item():.4f} prog={progressive_loss.item():.4f} "
                      f"edge={edge_loss.item():.4f} com={commonality_loss.item():.4f} "
                      f"lr={scheduler.get_last_lr()[1]:.6f}")

            if step % args.val_interval == 0 or step == args.max_iters:
                model.eval()
                evaluator.reset()
                with torch.no_grad():
                    for pre_img, post_img, labels, _, _ in val_loader:
                        pre_img = pre_img.to(device, dtype=torch.float32)
                        post_img = post_img.to(device, dtype=torch.float32)
                        labels = labels.to(device, dtype=torch.long)
                        if pre_img.dim() == 3:
                            pre_img = pre_img.unsqueeze(0)
                            post_img = post_img.unsqueeze(0)
                            labels = labels.unsqueeze(0)
                        preds = model(pre_img, post_img)["change"]
                        evaluator.add_batch(labels.cpu().numpy(),
                                            torch.argmax(preds, dim=1).cpu().numpy())

                f1 = evaluator.Pixel_F1_score()
                print(f"Val @ {step}: F1={f1:.4f} IoU={evaluator.Intersection_over_Union():.4f} OA={evaluator.Pixel_Accuracy():.4f}")
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save({"model": model.state_dict(), "step": step, "f1": f1},
                               save_dir / f"best_model_f1_{f1:.4f}.pth")
                    print(f"  -> saved best (F1={f1:.4f})")

    print(f"Done. Best F1: {best_f1:.4f}")


def _parse_args():
    p = argparse.ArgumentParser(description="CoVaL Training")
    p.add_argument("--cfg", type=str, required=True, help="YAML config path")
    p.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    p.add_argument("--pretrained_weight_path", type=str, default="")
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--dataset", type=str, default="LEVIR-CD-256")
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--val_batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--encoder_lr_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_iters", type=int, default=50000)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--decoder_dims", type=int, nargs=4, default=None)
    p.add_argument("--decoder_output_stride", type=int, default=1, choices=[1, 4])
    p.add_argument("--edge_stages", type=str, default="234")
    p.add_argument("--ce_weight", type=float, default=0.5)
    p.add_argument("--lovasz_weight", type=float, default=1.0)
    p.add_argument("--progressive_loss_weight", type=float, default=1.5, help="λ_aux: progressive deep supervision weight")
    p.add_argument("--edge_loss_weight", type=float, default=0.1, help="λ_edge: edge-aware structural loss weight")
    p.add_argument("--commonality_loss_weight", type=float, default=0.2, help="λ_com: commonality preservation loss weight")
    p.add_argument("--model_param_path", type=str, default="saved_models/CoVaL_run")
    p.add_argument("--log_interval", type=int, default=20)
    p.add_argument("--val_interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--validation_split_mode", type=str, default="test", choices=["test", "val"])
    return p.parse_args()


if __name__ == "__main__":
    main()

