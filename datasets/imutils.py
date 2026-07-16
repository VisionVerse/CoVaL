import random
import numpy as np
from PIL import Image, ImageEnhance


def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    img_array = np.asarray(img)
    normalized_img = np.empty_like(img_array, np.float32)
    for i in range(3):
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
    return normalized_img


def random_fliplr(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.fliplr(label)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)
    return pre_img, post_img, label


def random_fliplr_bda(pre_img, post_img, label_1, label_2):
    if random.random() > 0.5:
        label_1 = np.fliplr(label_1)
        label_2 = np.fliplr(label_2)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)
    return pre_img, post_img, label_1, label_2


def random_fliplr_mcd(pre_img, post_img, label_cd, label_1, label_2):
    if random.random() > 0.5:
        label_cd = np.fliplr(label_cd)
        label_1 = np.fliplr(label_1)
        label_2 = np.fliplr(label_2)
        pre_img = np.fliplr(pre_img)
        post_img = np.fliplr(post_img)
    return pre_img, post_img, label_cd, label_1, label_2


def random_flipud(pre_img, post_img, label):
    if random.random() > 0.5:
        label = np.flipud(label)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)
    return pre_img, post_img, label


def random_flipud_bda(pre_img, post_img, label_1, label_2):
    if random.random() > 0.5:
        label_1 = np.flipud(label_1)
        label_2 = np.flipud(label_2)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)
    return pre_img, post_img, label_1, label_2


def random_flipud_mcd(pre_img, post_img, label_cd, label_1, label_2):
    if random.random() > 0.5:
        label_cd = np.flipud(label_cd)
        label_1 = np.flipud(label_1)
        label_2 = np.flipud(label_2)
        pre_img = np.flipud(pre_img)
        post_img = np.flipud(post_img)
    return pre_img, post_img, label_cd, label_1, label_2


def random_rot(pre_img, post_img, label):
    k = random.randrange(3) + 1
    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label = np.rot90(label, k).copy()
    return pre_img, post_img, label


def random_rot_bda(pre_img, post_img, label_1, label_2):
    k = random.randrange(3) + 1
    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label_1 = np.rot90(label_1, k).copy()
    label_2 = np.rot90(label_2, k).copy()
    return pre_img, post_img, label_1, label_2


def random_rot_mcd(pre_img, post_img, label_cd, label_1, label_2):
    k = random.randrange(3) + 1
    pre_img = np.rot90(pre_img, k).copy()
    post_img = np.rot90(post_img, k).copy()
    label_1 = np.rot90(label_1, k).copy()
    label_2 = np.rot90(label_2, k).copy()
    label_cd = np.rot90(label_cd, k).copy()
    return pre_img, post_img, label_cd, label_1, label_2


def random_color_jitter(pre_img, post_img, label,
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1):
    if random.random() > 0.5:
        brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
        saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)

        pre_pil = Image.fromarray(pre_img.astype(np.uint8))
        post_pil = Image.fromarray(post_img.astype(np.uint8))

        for enhancer_cls, factor in [
            (ImageEnhance.Brightness, brightness_factor),
            (ImageEnhance.Contrast, contrast_factor),
            (ImageEnhance.Color, saturation_factor),
        ]:
            pre_pil = enhancer_cls(pre_pil).enhance(factor)
            post_pil = enhancer_cls(post_pil).enhance(factor)

        pre_img = np.array(pre_pil, dtype=np.float32)
        post_img = np.array(post_pil, dtype=np.float32)

    return pre_img, post_img, label


def _get_random_cropbox(pad_label, H, W, crop_size, ignore_index=255, cat_max_ratio=0.75):
    for _ in range(10):
        H_start = random.randrange(0, H - crop_size + 1, 1)
        W_start = random.randrange(0, W - crop_size + 1, 1)
        H_end = H_start + crop_size
        W_end = W_start + crop_size
        temp_label = pad_label[H_start:H_end, W_start:W_end]
        index, cnt = np.unique(temp_label, return_counts=True)
        cnt = cnt[index != ignore_index]
        if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
            break
    return H_start, H_end, W_start, W_end


def random_crop_new(pre_img, post_img, label, crop_size, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label.shape
    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre = np.zeros((H, W, 3), dtype=np.float32)
    pad_post = np.zeros((H, W, 3), dtype=np.float32)
    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index

    for c in range(3):
        pad_pre[:, :, c] = mean_rgb[c]
        pad_post[:, :, c] = mean_rgb[c]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))
    pad_pre[H_pad:H_pad + h, W_pad:W_pad + w, :] = pre_img
    pad_post[H_pad:H_pad + h, W_pad:W_pad + w, :] = post_img
    pad_label[H_pad:H_pad + h, W_pad:W_pad + w] = label

    H_start, H_end, W_start, W_end = _get_random_cropbox(pad_label, H, W, crop_size, ignore_index)
    return (
        pad_pre[H_start:H_end, W_start:W_end, :],
        pad_post[H_start:H_end, W_start:W_end, :],
        pad_label[H_start:H_end, W_start:W_end],
    )


def random_crop_bda(pre_img, post_img, loc_label, clf_label, crop_size,
                    mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = loc_label.shape
    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre = np.zeros((H, W, 3), dtype=np.float32)
    pad_post = np.zeros((H, W, 3), dtype=np.float32)
    pad_loc = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_clf = np.ones((H, W), dtype=np.float32) * ignore_index

    for c in range(3):
        pad_pre[:, :, c] = mean_rgb[c]
        pad_post[:, :, c] = mean_rgb[c]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))
    pad_pre[H_pad:H_pad + h, W_pad:W_pad + w, :] = pre_img
    pad_post[H_pad:H_pad + h, W_pad:W_pad + w, :] = post_img
    pad_loc[H_pad:H_pad + h, W_pad:W_pad + w] = loc_label
    pad_clf[H_pad:H_pad + h, W_pad:W_pad + w] = clf_label

    H_start, H_end, W_start, W_end = _get_random_cropbox(pad_loc, H, W, crop_size, ignore_index)
    return (
        pad_pre[H_start:H_end, W_start:W_end, :],
        pad_post[H_start:H_end, W_start:W_end, :],
        pad_loc[H_start:H_end, W_start:W_end],
        pad_clf[H_start:H_end, W_start:W_end],
    )


def random_crop_mcd(pre_img, post_img, label_cd, label_1, label_2, crop_size,
                    mean_rgb=[0, 0, 0], ignore_index=255):
    h, w = label_1.shape
    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_pre = np.zeros((H, W, 3), dtype=np.float32)
    pad_post = np.zeros((H, W, 3), dtype=np.float32)
    pad_cd = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_l1 = np.ones((H, W), dtype=np.float32) * ignore_index
    pad_l2 = np.ones((H, W), dtype=np.float32) * ignore_index

    for c in range(3):
        pad_pre[:, :, c] = mean_rgb[c]
        pad_post[:, :, c] = mean_rgb[c]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))
    pad_pre[H_pad:H_pad + h, W_pad:W_pad + w, :] = pre_img
    pad_post[H_pad:H_pad + h, W_pad:W_pad + w, :] = post_img
    pad_cd[H_pad:H_pad + h, W_pad:W_pad + w] = label_cd
    pad_l1[H_pad:H_pad + h, W_pad:W_pad + w] = label_1
    pad_l2[H_pad:H_pad + h, W_pad:W_pad + w] = label_2

    H_start, H_end, W_start, W_end = _get_random_cropbox(pad_l1, H, W, crop_size, ignore_index)
    return (
        pad_pre[H_start:H_end, W_start:W_end, :],
        pad_post[H_start:H_end, W_start:W_end, :],
        pad_cd[H_start:H_end, W_start:W_end],
        pad_l1[H_start:H_end, W_start:W_end],
        pad_l2[H_start:H_end, W_start:W_end],
    )
