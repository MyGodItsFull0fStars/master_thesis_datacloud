class PenaltyMSELoss(torch.nn.Module):
    penalty: float
    
    def __init__(self, penalty: float = 0.05):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.penalty = penalty
        
    def forward(self, pred: torch.Tensor, actual: torch.Tensor):
        under_alloc_indices = pred < actual
        pred[under_alloc_indices] = pred[under_alloc_indices] - self.penalty
        return self.mse(pred, actual)