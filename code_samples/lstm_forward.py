class UtilizationLSTM(torch.nn.Module):

    def __init__():
        ...

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