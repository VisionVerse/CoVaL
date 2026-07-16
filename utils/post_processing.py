import numpy as np
import cv2


def post_process_predictions(
    pred_mask: np.ndarray,
    min_area: int = 50,
    morph_kernel_size: int = 3,
    open_iterations: int = 1,
    close_iterations: int = 1
) -> np.ndarray:
    mask = pred_mask.astype(np.uint8)
    if mask.max() == 0:
        return mask.astype(np.float32)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))

    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iterations)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    cleaned = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1

    closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=close_iterations)
    return closed.astype(np.float32)


def batch_post_process(pred_masks: np.ndarray, min_area: int = 50, **kwargs) -> np.ndarray:
    cleaned = np.zeros_like(pred_masks, dtype=np.float32)
    for i in range(pred_masks.shape[0]):
        cleaned[i] = post_process_predictions(pred_masks[i], min_area=min_area, **kwargs)
    return cleaned


def adaptive_post_process(pred_mask: np.ndarray, image_size: int = 256) -> np.ndarray:
    if image_size <= 256:
        min_area, kernel_size = 50, 3
    elif image_size <= 512:
        min_area, kernel_size = 100, 5
    else:
        min_area, kernel_size = 200, 7
    return post_process_predictions(pred_mask, min_area=min_area, morph_kernel_size=kernel_size)
