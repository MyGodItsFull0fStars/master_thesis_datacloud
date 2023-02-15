class UtilizationLSTM(torch.nn.Module):

    def __init__(
        self, 
        num_classes: int, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int = 1, 
        generalization: str = 'batch', 
        lstm_mode: str = 'full'
        ) -> None:
        
        super(UtilizationLSTM, self).__init__()
        self.num_classes: int = num_classes
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_layers = num_layers
        self.lstm_mode = lstm_mode
        self.device = device
        
        # if dataset is split, remove two columns not used in the lstm model
        split_decrement: int = 2 if lstm_mode == 'split' else 0

        # long-short term memory layer to predict cpu usage
        self.cpu_lstm = torch.nn.LSTM(
            input_size=input_size - split_decrement,
            hidden_size=hidden_size,
            num_layers=self.num_layers, 
            batch_first=True,
        ).to(device)

        # long-short term memory layer to predict memory usage
        self.mem_lstm = torch.nn.LSTM(
            input_size=input_size - split_decrement,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        ).to(device)

        if generalization == 'dropout':
            self.cpu_lstm_seq = self.init_sequential_layer_dropout(hidden_size)
            self.mem_lstm_seq = self.init_sequential_layer_dropout(hidden_size)
            
        elif generalization == 'batch':
            self.cpu_lstm_seq = self.init_sequential_layer_batchnorm(hidden_size)
            self.mem_lstm_seq = self.init_sequential_layer_batchnorm(hidden_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.lstm_mode == 'full':
            cpu_input, mem_input = input, input
        elif self.lstm_mode == 'split':
            cpu_input, mem_input = self.split_input(input)

        # Propagate input through LSTM
        _, (cpu_ht, _) = self.cpu_lstm(cpu_input,
                                       self.get_hidden_internal_state(input))
        _, (mem_ht, _) = self.mem_lstm(mem_input,
                                       self.get_hidden_internal_state(input))
               
        # Reshaping the data for the Dense layer
        cpu_ht = cpu_ht.view(-1, self.hidden_size)
        mem_ht = mem_ht.view(-1, self.hidden_size)

        cpu_out: torch.Tensor = self.cpu_lstm_seq(cpu_ht)
        mem_out: torch.Tensor = self.mem_lstm_seq(mem_ht)

        # Concat the two tensors column-wise
        output = torch.cat([cpu_out, mem_out], dim=1)
        
        # Only use the last stacked lstm layer as output
        output = output[(self.num_layers - 1) * input.size(0):]

        # don't allow negative values
        output = torch.abs(output)
        
        return output
        # return torch.abs(output)
        
    def get_hidden_internal_state(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_state = torch.zeros(self.num_layers, input.size(0), self.hidden_size).requires_grad_().to(device)
        internal_state = torch.zeros(self.num_layers, input.size(0), self.hidden_size).requires_grad_().to(device)

        return (hidden_state, internal_state)

    def split_input(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if input.size(dim=2) < 4:
            return torch.empty(1), torch.empty(1)
        
        cpu_columns, mem_columns = [0, 1], [2, 3]
        
        if input.size(dim=2) > 4:
            cpu_columns = cpu_columns + [x for x in range(4, input.size(dim=2))]
            mem_columns = mem_columns + [x for x in range(4, input.size(dim=2))]

        cpu_input, mem_input = input[:, :, cpu_columns], input[:, :, mem_columns]
        
        return (cpu_input, mem_input)

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


    def init_sequential_layer_dropout(self, hidden_size: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(hidden_size, 512),
            
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512, 128),
            
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(128, 1),
        ).to(device)
