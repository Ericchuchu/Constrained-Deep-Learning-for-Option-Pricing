import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List, Tuple

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, dropout_rate=0.1):
        super(ConvLSTMCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 5
        self.padding = int((kernel_size - 1) / 2)
        
        # Add layer normalization for hidden states
        self.layer_norm = nn.LayerNorm([self.num_features])
        
        self.Wxi = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x.requires_grad_(True)
        h.requires_grad_(True) 
        c.requires_grad_(True)
        
        # Normalize hidden state
        h = self.layer_norm(h)
        
        # Gates with regularization
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        
        # Output with residual connection and normalization
        ch = co * torch.tanh(cc) + h
        ch = self.layer_norm(ch)
        
        return ch, cc

    def init_hidden(self, batch_size, hidden, dim):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, dim))
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, dim))
            self.Wco = nn.Parameter(torch.zeros(1, hidden, dim))
        else:
            assert dim == self.Wci.size()[2], 'Input Dim Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, dim), requires_grad=True),
                Variable(torch.zeros(batch_size, hidden, dim), requires_grad=True))
    
class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, in_dim, out_dim, step=10, dropout_rate=0.1):
        super(ConvLSTM, self).__init__()
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)
        self.step = step
        self._all_layers = []
        
        # Batch normalization for ConvLSTM layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels[i]) 
            for i in range(len(hidden_channels))
        ])
        
        # Enhanced fully connected layers
        self.fc1 = nn.Linear(in_features=in_dim, out_features=in_dim*2)
        self.bn1 = nn.BatchNorm1d(in_dim*2)
        self.fc2 = nn.Linear(in_features=in_dim*2, out_features=out_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i], self.kernel_size)
            setattr(self, name, cell)
            self._all_layers.append(cell)

    def forward(self, input):
        # input: (N, C, T, D_in)
        bsize, _, seq_len, _ = input.size()
        device = input.device
        
        # initialize hidden states of all layers
        hidden_states = []
        for i in range(self.num_layers):
            cell = getattr(self, f'cell{i}')
            h, c = cell.init_hidden(bsize, cell.hidden_channels, input.size(-1))
            hidden_states.append((h.to(device), c.to(device)))

        # process time steps
        outputs = []
        for t in range(seq_len):
            x = input[:, :, t, :]
            layer_output = []
            for i in range(self.num_layers):
                h, c = hidden_states[i]
                h_new, c_new = getattr(self, f'cell{i}')(x, h, c)
                # Apply batch normalization
                h_new = self.batch_norms[i](h_new)
                hidden_states[i] = (h_new, c_new)
                x = h_new
            outputs.append(x)

        # Stack outputs and get final prediction
        stacked_outputs = torch.stack(outputs, dim=1)
        final_hidden = stacked_outputs[:, -1, :].squeeze()
        
        # Enhanced prediction layers with batch norm
        x = self.fc1(final_hidden)
        x = self.bn1(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Ensure positive output for option prices using softplus activation
        # Softplus: f(x) = ln(1 + exp(x)) is a smoother alternative to ReLU
        # It ensures non-negative outputs while being more stable than exponential
        return F.softplus(x).squeeze()

# test
if __name__ == '__main__':
    input = torch.normal(0, 1, size=(64, 3, 10, 5))
    model = ConvLSTM(input_channels=3, hidden_channels=[3, 1], kernel_size=3, in_dim=5, out_dim=1, step=10)
    output = model(input)
    print(output.shape)
