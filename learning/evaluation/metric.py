import torch
from typing import Optional
from torchmetrics import Metric
from torchmetrics.functional.classification.average_precision \
        import binary_average_precision


class GraphAveragePrecision(Metric):
    """ Compute the average precision for the whole graph. """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_ap", default = torch.tensor(0.0), dist_reduce_fx = "sum")
        self.add_state("total_graph", default = torch.tensor(0), dist_reduce_fx = "sum")

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        graph_ap = binary_average_precision(pred, target, thresholds = None)
        if torch.isnan(graph_ap).any():
            return

        graph_ap.to(device = self.device)
        self.sum_ap += graph_ap
        self.total_graph += torch.tensor(1)

    def compute(self):
        return self.sum_ap / self.total_graph
