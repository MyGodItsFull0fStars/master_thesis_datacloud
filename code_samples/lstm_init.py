class UtilizationLSTM(torch.nn.Module):

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int = 1, 
        ) -> None:
        
        super(UtilizationLSTM, self).__init__()
        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.num_layers = num_layers
        self.device = device

        # long-short term memory layer to predict cpu usage
        self.cpu_lstm = torch.nn.LSTM(
            input_size=input_size
            hidden_size=hidden_size,
            num_layers=self.num_layers, 
            batch_first=True,
        ).to(self.device)

        # long-short term memory layer to predict memory usage
        self.mem_lstm = torch.nn.LSTM(
            input_size=input_size
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        ).to(self.device)

        self.cpu_lstm_seq = self.init_sequential_layer_batchnorm(hidden_size)
        self.mem_lstm_seq = self.init_sequential_layer_batchnorm(hidden_size)