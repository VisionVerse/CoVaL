
import argparse
from pathlib import Path

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import get_config
from datasets.make_data_loader import ChangeDetectionDatset
from utils.metrics import Evaluator
from utils.post_processing import batch_post_process
from models.coval import CoVaLModel


def parse_args():
    p = argparse.ArgumentParser(description="CoVaL Inference")
    p.add_argument("--cfg", type=str, required=True)
    p.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    p.add_argument("--resume", type=str, required=True)
    p.add_argument("--dataset", type=str, default="LEVIR-CD-256")
    p.add_argument("--test_dataset_path", type=str, required=True)
    p.add_argument("--test_data_list_path", type=str, required=True)
    p.add_argument("--result_saved_path", type=str, default="./results/CoVaL")
    p.add_argument("--decoder_dims", type=int, nargs=4, default=None)
    p.add_argument("--upsample_mode", type=str, default="bilinear", choices=["bilinear", "transpose", "nearest", "bicubic"])
    p.add_argument("--decoder_output_stride", type=int, default=1, choices=[1, 4])
    p.add_argument("--edge_stages", type=str, default="234")
    p.add_argument("--use_post_processing", action="store_true")
    p.add_argument("--post_min_area", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    config = get_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vssm_cfg = config.MODEL.VSSM
    model = CoVaLModel(
        pretrained=None,
        upsample_mode=args.upsample_mode,
        decoder_output_stride=args.decoder_output_stride,
        decoder_dims=args.decoder_dims,
        edge_stages=args.edge_stages,
        use_commonality=False,
        patch_size=vssm_cfg.PATCH_SIZE,
        in_chans=vssm_cfg.IN_CHANS,
        num_classes=config.MODEL.NUM_CLASSES,
        depths=vssm_cfg.DEPTHS,
        dims=vssm_cfg.EMBED_DIM,
        ssm_d_state=vssm_cfg.SSM_D_STATE,
        ssm_ratio=vssm_cfg.SSM_RATIO,
        ssm_rank_ratio=vssm_cfg.SSM_RANK_RATIO,
        ssm_dt_rank=("auto" if vssm_cfg.SSM_DT_RANK == "auto" else int(vssm_cfg.SSM_DT_RANK)),
        ssm_act_layer=vssm_cfg.SSM_ACT_LAYER,
        ssm_conv=vssm_cfg.SSM_CONV,
        ssm_conv_bias=vssm_cfg.SSM_CONV_BIAS,
        ssm_drop_rate=vssm_cfg.SSM_DROP_RATE,
        ssm_init=vssm_cfg.SSM_INIT,
        forward_type=vssm_cfg.SSM_FORWARDTYPE,
        mlp_ratio=vssm_cfg.MLP_RATIO,
        mlp_act_layer=vssm_cfg.MLP_ACT_LAYER,
        mlp_drop_rate=vssm_cfg.MLP_DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        patch_norm=vssm_cfg.PATCH_NORM,
        norm_layer=vssm_cfg.NORM_LAYER,
        downsample_version=vssm_cfg.DOWNSAMPLE,
        patchembed_version=vssm_cfg.PATCHEMBED,
        gmlp=vssm_cfg.GMLP,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    ).to(device)

    resume_path = Path(args.resume)
    if not resume_path.is_file():
        raise FileNotFoundError(f"No checkpoint found at '{args.resume}'")
    checkpoint = torch.load(resume_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    if any("fused_conv" in k for k in state_dict.keys()) and hasattr(model, "switch_to_deploy"):
        model.switch_to_deploy()
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    result_dir = Path(args.result_saved_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    change_map_dir = result_dir / "change_map"
    change_map_dir.mkdir(parents=True, exist_ok=True)

    with open(args.test_data_list_path, "r", encoding="utf-8") as f:
        data_list = [line.strip() for line in f if line.strip()]

    dataset = ChangeDetectionDatset(args.test_dataset_path, data_list, args.crop_size, None, "test")
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)

    evaluator = Evaluator(num_class=2)
    use_pp = args.use_post_processing

    torch.cuda.empty_cache()
    with torch.no_grad():
        for data in tqdm(loader, desc="Inferring"):
            pre_imgs, post_imgs, labels, _, names = data
            pre_imgs = pre_imgs.to(device, dtype=torch.float32)
            post_imgs = post_imgs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(pre_imgs, post_imgs)
            pred_labels = torch.argmax(outputs["change"], dim=1).cpu().numpy()

            if use_pp:
                pred_labels = batch_post_process(pred_labels, min_area=args.post_min_area, morph_kernel_size=3, open_iterations=1, close_iterations=1)

            pred_labels = pred_labels.astype(np.int64)
            gt_labels = labels.cpu().numpy().astype(np.int64)
            evaluator.add_batch(gt_labels, pred_labels)

            for i, name in enumerate(names):
                if isinstance(name, (list, tuple)):
                    name = name[0]
                image_name = Path(str(name)).stem + ".png"
                binary_map = (pred_labels[i] * 255).astype(np.uint8)
                imageio.imwrite(str(change_map_dir / image_name), binary_map, format="png")

    rec = evaluator.Pixel_Recall_Rate()
    pre = evaluator.Pixel_Precision_Rate()
    oa = evaluator.Pixel_Accuracy()
    f1 = evaluator.Pixel_F1_score()
    iou = evaluator.Intersection_over_Union()
    kc = evaluator.Kappa_coefficient()

    summary_path = result_dir / "summary_metrics.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Recall: {rec:.4f}\nPrecision: {pre:.4f}\nOA: {oa:.4f}\nF1: {f1:.4f}\nIoU: {iou:.4f}\nKappa: {kc:.4f}\n")

    print(f"Recall: {rec:.4f} | Precision: {pre:.4f} | OA: {oa:.4f} | F1: {f1:.4f} | IoU: {iou:.4f} | Kappa: {kc:.4f}")


if __name__ == "__main__":
    main()
