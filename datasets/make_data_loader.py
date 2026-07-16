from pathlib import Path

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import cv2

import datasets.imutils as imutils


def generate_edge_label(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    mask_u8 = (mask * 255).astype(np.uint8)
    if mask_u8.max() == 0:
        return np.zeros_like(mask, dtype=np.float32)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    edge = np.zeros_like(mask, dtype=np.float32)

    for i in range(1, num_labels):
        obj_mask = (labels == i).astype(np.uint8) * 255
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 100:
            k = 1
        elif area < 500:
            k = 2
        else:
            k = max(2, kernel_size)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        eroded = cv2.erode(obj_mask, kernel, iterations=1)
        edge += cv2.subtract(obj_mask, eroded).astype(np.float32) / 255.0

    return np.clip(edge, 0, 1)


def img_loader(path: str) -> np.ndarray:
    try:
        img = np.array(imageio.imread(path), dtype=np.float32)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        elif img.ndim == 3:
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            elif img.shape[2] >= 4:
                img = img[:, :, :3]
        else:
            raise ValueError(f"Unsupported image shape: {img.shape} for {path}")
        return img
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {path}")
    except Exception as e:
        raise IOError(f"Failed to load image {path}: {e}")


class ChangeDetectionDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size,
                 max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = Path(dataset_path)
        self.data_list = list(data_list)
        self.loader = data_loader
        self.data_pro_type = type
        self.crop_size = crop_size
        self._ext_candidates = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        self._split_prefixes = self._build_split_prefixes(type)
        if max_iters is not None:
            self.data_list = (self.data_list * int(np.ceil(max_iters / len(self.data_list))))[:max_iters]

    def _build_split_prefixes(self, split_type: str):
        split = (split_type or '').lower()
        if split == 'train':
            return ('train', 'trn', 'trainset')
        if split == 'val':
            return ('val', 'valid', 'validation', 'dev')
        if split == 'test':
            return ('test', 'tst')
        return ('train', 'val', 'test')

    def _resolve_file(self, folder: str, name: str) -> Path:
        base_dir = self.dataset_path / folder
        p = base_dir / name
        if p.is_file():
            return p

        stem = Path(name).stem
        suffix = Path(name).suffix.lower()

        def _try_candidates(candidate_stem: str):
            if not candidate_stem:
                return None
            candidate_stem = Path(str(candidate_stem)).name
            direct = base_dir / candidate_stem
            if direct.is_file():
                return direct
            if not Path(candidate_stem).suffix:
                for ext in self._ext_candidates:
                    cand = base_dir / f"{candidate_stem}{ext}"
                    if cand.is_file():
                        return cand
            matches = sorted(base_dir.glob(f"{candidate_stem}.*"))
            if matches:
                exact = [m for m in matches if m.stem == candidate_stem]
                return exact[0] if exact else matches[0]
            matches = sorted(base_dir.glob(f"*{candidate_stem}.*"))
            if matches:
                return matches[0]
            return None

        resolved = _try_candidates(stem or name)
        if resolved is not None:
            return resolved

        normalized = []
        if stem:
            normalized.append(stem)
        if suffix:
            normalized.append(name)
        for prefix in ('train_', 'test_', 'val_', 'trn_', 'tst_'):
            if stem.startswith(prefix):
                normalized.append(stem[len(prefix):])
        if '_' in stem:
            normalized.append(stem.split('_', 1)[-1])
        seen = set()
        normalized = [x for x in normalized if not (x in seen or seen.add(x))]

        for cand_stem in normalized:
            resolved = _try_candidates(cand_stem)
            if resolved is not None:
                return resolved

        raise FileNotFoundError(f"Image file not found: {p}")

    def _transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_color_jitter(pre_img, post_img, label)
        pre_img = np.transpose(imutils.normalize_img(pre_img), (2, 0, 1))
        post_img = np.transpose(imutils.normalize_img(post_img), (2, 0, 1))
        return pre_img, post_img, label

    def __getitem__(self, index):
        name = self.data_list[index]
        pre_img = self.loader(str(self._resolve_file('A', name)))
        post_img = self.loader(str(self._resolve_file('B', name)))
        label = self.loader(str(self._resolve_file('label', name)))
        if label.ndim == 3:
            label = label[:, :, 0]
        label = (label / 255.0).astype(np.float32)
        aug = 'train' in self.data_pro_type
        pre_img, post_img, label = self._transforms(aug, pre_img, post_img, label)
        if not aug:
            label = np.asarray(label)
        edge_label = generate_edge_label(label, kernel_size=3)
        return pre_img, post_img, label, edge_label, name

    def __len__(self):
        return len(self.data_list)


class DamageAssessmentDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size,
                 max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = Path(dataset_path)
        self.data_list = list(data_list)
        self.loader = data_loader
        self.data_pro_type = type
        self.crop_size = crop_size
        if max_iters is not None:
            self.data_list = (self.data_list * int(np.ceil(max_iters / len(self.data_list))))[:max_iters]

    def _transforms(self, aug, pre_img, post_img, loc_label, clf_label):
        if aug:
            pre_img, post_img, loc_label, clf_label = imutils.random_crop_bda(
                pre_img, post_img, loc_label, clf_label, self.crop_size)
            pre_img, post_img, loc_label, clf_label = imutils.random_fliplr_bda(
                pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_flipud_bda(
                pre_img, post_img, loc_label, clf_label)
            pre_img, post_img, loc_label, clf_label = imutils.random_rot_bda(
                pre_img, post_img, loc_label, clf_label)
        pre_img = np.transpose(imutils.normalize_img(pre_img), (2, 0, 1))
        post_img = np.transpose(imutils.normalize_img(post_img), (2, 0, 1))
        return pre_img, post_img, loc_label, clf_label

    def __getitem__(self, index):
        name = self.data_list[index]
        root = self.dataset_path
        aug = 'train' in self.data_pro_type
        if aug:
            parts = name.rsplit('_', 2)
            pre_name  = f"{parts[0]}_pre_disaster_{parts[1]}_{parts[2]}.png"
            post_name = f"{parts[0]}_post_disaster_{parts[1]}_{parts[2]}.png"
        else:
            pre_name  = f"{name}_pre_disaster.png"
            post_name = f"{name}_post_disaster.png"
        pre_img   = self.loader(str(root / 'images' / pre_name))
        post_img  = self.loader(str(root / 'images' / post_name))
        loc_label = self.loader(str(root / 'masks'  / pre_name))[:, :, 0]
        clf_label = self.loader(str(root / 'masks'  / post_name))[:, :, 0]
        pre_img, post_img, loc_label, clf_label = self._transforms(
            aug, pre_img, post_img, loc_label, clf_label)
        if aug:
            clf_label[clf_label == 0] = 255
        else:
            loc_label = np.asarray(loc_label)
            clf_label = np.asarray(clf_label)
        return pre_img, post_img, loc_label, clf_label, name

    def __len__(self):
        return len(self.data_list)


class MultimodalDamageAssessmentDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size,
                 max_iters=None, type='train', data_loader=img_loader, suffix='.tif'):
        self.dataset_path = Path(dataset_path)
        self.data_list = list(data_list)
        self.loader = data_loader
        self.data_pro_type = type
        self.crop_size = crop_size
        self.suffix = suffix
        if max_iters is not None:
            self.data_list = (self.data_list * int(np.ceil(max_iters / len(self.data_list))))[:max_iters]

    def _transforms(self, aug, pre_img, post_img, label):
        if aug:
            pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)
        pre_img = np.transpose(imutils.normalize_img(pre_img), (2, 0, 1))
        post_img = np.transpose(imutils.normalize_img(post_img), (2, 0, 1))
        return pre_img, post_img, label

    def __getitem__(self, index):
        name = self.data_list[index]
        root = self.dataset_path
        pre_folder  = 'time1' if 'SYSU' in str(root) else 'A'
        post_folder = 'time2' if 'SYSU' in str(root) else 'B'
        pre_img = self.loader(str(root / pre_folder  / name))
        post_img = self.loader(str(root / post_folder / name))
        label = self.loader(str(root / 'label' / name)) / 255.0
        aug = 'train' in self.data_pro_type
        pre_img, post_img, label = self._transforms(aug, pre_img, post_img, label)
        if not aug:
            label = np.asarray(label)
        return pre_img, post_img, label, name

    def __len__(self):
        return len(self.data_list)


class SemanticChangeDetectionDatset(Dataset):
    def __init__(self, dataset_path, data_list, crop_size,
                 max_iters=None, type='train', data_loader=img_loader):
        self.dataset_path = Path(dataset_path)
        self.data_list = list(data_list)
        self.loader = data_loader
        self.data_pro_type = type
        self.crop_size = crop_size
        if max_iters is not None:
            self.data_list = (self.data_list * int(np.ceil(max_iters / len(self.data_list))))[:max_iters]

    def _transforms(self, aug, pre_img, post_img, cd_label, t1_label, t2_label):
        if aug:
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_crop_mcd(
                pre_img, post_img, cd_label, t1_label, t2_label, self.crop_size)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_fliplr_mcd(
                pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_flipud_mcd(
                pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_rot_mcd(
                pre_img, post_img, cd_label, t1_label, t2_label)
        pre_img = np.transpose(imutils.normalize_img(pre_img), (2, 0, 1))
        post_img = np.transpose(imutils.normalize_img(post_img), (2, 0, 1))
        return pre_img, post_img, cd_label, t1_label, t2_label

    def __getitem__(self, index):
        name = self.data_list[index]
        root = self.dataset_path
        aug = 'train' in self.data_pro_type
        suffix = f"{name}.png" if aug else name
        pre_img = self.loader(str(root / 'T1' / suffix))
        post_img = self.loader(str(root / 'T2' / suffix))
        t1_label = self.loader(str(root / 'GT_T1' / suffix))
        t2_label = self.loader(str(root / 'GT_T2' / suffix))
        cd_label = self.loader(str(root / 'GT_CD' / suffix)) / 255.0
        pre_img, post_img, cd_label, t1_label, t2_label = self._transforms(
            aug, pre_img, post_img, cd_label, t1_label, t2_label)
        if not aug:
            cd_label = np.asarray(cd_label)
            t1_label = np.asarray(t1_label)
            t2_label = np.asarray(t2_label)
        return pre_img, post_img, cd_label, t1_label, t2_label, name

    def __len__(self):
        return len(self.data_list)


def make_data_loader(args, **kwargs) -> DataLoader:
    num_workers = getattr(args, 'num_workers', 8)

    def _loader(dataset_cls):
        ds = dataset_cls(
            args.train_dataset_path,
            args.train_data_name_list,
            args.crop_size,
            args.max_iters,
            args.type,
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=args.shuffle,
                         num_workers=num_workers, drop_last=False, **kwargs)

    name = args.dataset
    if any(k in name for k in ('SYSU', 'LEVIR-CD', 'WHU', 'CDD')):
        return _loader(ChangeDetectionDatset)
    elif 'xBD' in name:
        return _loader(DamageAssessmentDatset)
    elif 'SECOND' in name:
        return _loader(SemanticChangeDetectionDatset)
    elif 'BRIGHT' in name:
        ds = MultimodalDamageAssessmentDatset(
            args.train_dataset_path,
            args.train_data_name_list,
            args.crop_size,
            args.max_iters,
            args.type,
        )
        batch_size = getattr(args, 'train_batch_size', args.batch_size)
        return DataLoader(ds, batch_size=batch_size, shuffle=args.shuffle,
                         num_workers=num_workers, drop_last=False, **kwargs)
    else:
        raise NotImplementedError(f"Dataset '{name}' is not supported")
