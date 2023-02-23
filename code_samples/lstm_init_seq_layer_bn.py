def init_sequential_layer_batchnorm(self, hidden_size: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(hidden_size),
        torch.nn.Linear(hidden_size, 512),
        
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(512),
        torch.nn.Linear(512, 128),
        
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(128),
        torch.nn.Linear(128, 1),
    ).to(device)