import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.longlong)

    def Pixel_Accuracy(self):
        return np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-7)
        return np.nanmean(Acc), Acc

    def Pixel_Precision_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        TP = self.confusion_matrix[1, 1]
        FP = self.confusion_matrix[0, 1]
        denom = FP + TP
        return float(TP / denom) if denom != 0 else 0.0

    def Pixel_Recall_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        TP = self.confusion_matrix[1, 1]
        FN = self.confusion_matrix[1, 0]
        denom = FN + TP
        return float(TP / denom) if denom != 0 else 0.0

    def Pixel_F1_score(self):
        assert self.confusion_matrix.shape[0] == 2
        rec = self.Pixel_Recall_Rate()
        pre = self.Pixel_Precision_Rate()
        denom = rec + pre
        return 2 * rec * pre / denom if denom != 0 else 0.0

    def _per_class_metrics(self):
        TPs = np.diag(self.confusion_matrix)[1:]
        FNs = np.sum(self.confusion_matrix, axis=1)[1:] - TPs
        FPs = np.sum(self.confusion_matrix, axis=0)[1:] - TPs
        return TPs, FNs, FPs

    def calculate_per_class_metrics(self):
        return self._per_class_metrics()

    def Damage_F1_socore(self):
        TPs, FNs, FPs = self._per_class_metrics()
        precisions = TPs / (TPs + FPs + 1e-7)
        recalls = TPs / (TPs + FNs + 1e-7)
        return 2 * (precisions * recalls) / (precisions + recalls + 1e-7)

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
            + 1e-7
        )
        return np.nanmean(MIoU)

    def Intersection_over_Union(self):
        TP = self.confusion_matrix[1, 1]
        FP = self.confusion_matrix[0, 1]
        FN = self.confusion_matrix[1, 0]
        union = FP + FN + TP
        return float(TP / union) if union != 0 else 0.0

    def Kappa_coefficient(self):
        num_total = np.sum(self.confusion_matrix)
        if num_total == 0:
            return 0.0
        observed = np.trace(self.confusion_matrix) / num_total
        expected = np.sum(
            np.sum(self.confusion_matrix, axis=0) / num_total
            * np.sum(self.confusion_matrix, axis=1) / num_total
        )
        return (observed - expected) / (1 - expected)

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1)
            + np.sum(self.confusion_matrix, axis=0)
            - np.diag(self.confusion_matrix)
        )
        return (freq[freq > 0] * iu[freq > 0]).sum()

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int64') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        return count.reshape(self.num_class, self.num_class)

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
