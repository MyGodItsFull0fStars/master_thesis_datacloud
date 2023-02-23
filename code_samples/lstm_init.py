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