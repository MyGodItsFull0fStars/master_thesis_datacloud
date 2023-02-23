def get_hidden_internal_state(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    hidden_state = torch.zeros(self.num_layers, input.size(0), self.hidden_size).requires_grad_().to(device)
    internal_state = torch.zeros(self.num_layers, input.size(0), self.hidden_size).requires_grad_().to(device)

    return (hidden_state, internal_state)