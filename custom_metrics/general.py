import torch

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryAUROC,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

class GeneralMetrics():
    _accuracy_func = None
    _f1_score_func = None
    _prec_func = None
    _recall_func = None
    _roc_auc_func = None
    _inference_time_func = None
    _storage_size_func = None

    _acc_val = None
    _f1_score_val = None
    _prec_val = None
    _recall_val = None
    _roc_auc_val = None
    _inference_time_val = None
    _storage_size_val = None

    def __init__(self, index: str, framework: str, num_outputs: int, device):
        self._index = index
        self._framework = framework
        self._num_outputs = num_outputs
        self._device = device

        if self._framework == "sklearn":
            self._accuracy_func = accuracy_score
            self._f1_score_func = f1_score
            self._prec_func = precision_score
            self._recall_func = recall_score
            self._roc_auc_func = roc_auc_score
        elif self._framework == "pytorch":
            if self._num_outputs > 1:
                self._accuracy_func = MulticlassAccuracy(num_classes=self._num_outputs)
                self._f1_score_func = MulticlassF1Score(num_classes=self._num_outputs)
                self._prec_func = MulticlassPrecision(num_classes=self._num_outputs)
                self._recall_func = MulticlassRecall(num_classes=self._num_outputs)
                self._roc_auc_func = MulticlassAUROC(num_classes=self._num_outputs)
            else:
                self._accuracy_func = BinaryAccuracy()
                self._f1_score_func = BinaryF1Score()
                self._prec_func = BinaryPrecision()
                self._recall_func = BinaryRecall()
                self._roc_auc_func = BinaryAUROC()
            # Move data to the device that is being used
            self._accuracy_func.to(self._device)
            self._f1_score_func.to(self._device)
            self._prec_func.to(self._device)
            self._recall_func.to(self._device)
            self._roc_auc_func.to(self._device)
        else:
            raise KeyError(f"Selected framework: {self._framework} is NOT available!")

    def update(self, y_pred, y_true):
        if self._framework == "pytorch":
                self._accuracy_func.update(y_pred.detach(), y_true)
                self._f1_score_func.update(y_pred.detach(), y_true)
                self._prec_func.update(y_pred.detach(), y_true)
                self._recall_func.update(y_pred.detach(), y_true)
                if self._num_outputs == 6:
                    self._roc_auc_func.update(y_pred.detach(), torch.argmax(y_true, dim=1))
                else:
                    self._roc_auc_func.update(y_pred.detach(), y_true)

    def compute(self):
        if self._framework == "sklearn":
            self._acc_val = self._accuracy_func(y_true, y_pred)
            self._f1_score_val = self._f1_score_func(y_true, y_pred)
            self._prec_val = self._prec_func(y_true, y_pred)
            self._recall_val = self._recall_func(y_true, y_pred)
            self._roc_auc_val = self._roc_auc_func(y_true, y_pred_prob[:, 1])
        elif self._framework == "pytorch":
            self._acc_val = self._accuracy_func.compute().cpu().numpy()
            self._f1_score_val = self._f1_score_func.compute().cpu().numpy()
            self._prec_val = self._prec_func.compute().cpu().numpy()
            self._recall_val = self._recall_func.compute().cpu().numpy()
            self._roc_auc_val = self._roc_auc_func.compute().cpu().numpy()


    def get_as_list(self):
        return [
            self._index,
            self._acc_val,
            self._f1_score_val,
            self._prec_val,
            self._recall_val,
            self._roc_auc_val,
            self._inference_time_val,
            self._storage_size_val
        ]
