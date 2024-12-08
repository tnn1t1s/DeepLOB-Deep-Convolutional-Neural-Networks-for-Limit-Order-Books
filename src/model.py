import torch
import torch.nn as nn
import torch.nn.functional as F

class deeplob(nn.Module):
    """
    Implementation of the DeepLOB model as described in the paper:
    'DeepLOB: Deep Convolutional Neural Networks for Limit Order Books'

    This model:
    - Takes input of shape (B, 1, T, 40), where B is batch size, T is sequence length.
    - Passes data through three convolutional blocks to extract local temporal patterns.
    - Uses inception modules to further process the representation.
    - Applies an LSTM layer on the feature dimension.
    - Outputs softmax probabilities for the specified number of classes (y_len).

    Args:
        y_len (int): Number of output classes.
    """
    def __init__(self, y_len):
        super(deeplob, self).__init__()
        self.y_len = y_len
        
        # First convolution block
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )

        # Second convolution block
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,2), stride=(1,2)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.Tanh(),
            nn.BatchNorm2d(32),
        )

        # Third convolution block
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(32),
        )
        
        # Inception modules
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3,1), stride=(1,1), padding=(1,0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm2d(64),
        )
        
        # LSTM + fully connected
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.y_len)

    def forward(self, x):
        # x shape: (B, 1, T, 40)
        device = x.device
        batch_size = x.size(0)

        # Initialize LSTM hidden states on the same device as input x
        h0 = torch.zeros(1, batch_size, 64, device=device)
        c0 = torch.zeros(1, batch_size, 64, device=device)
    
        # Pass through convolution blocks
        x = self.conv1(x)  # shape changes as per conv layers
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Pass through inception modules
        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        # Concatenate along channel dimension (dim=1)
        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)
        
        # x shape now: (B, 192, T', 1) after inception
        # We want to feed into LSTM: LSTM expects (B, seq_len, features)
        # permute to (B, T', channels, 1) -> (B, T', channels)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, x.shape[1], x.shape[2])  # (B, T', 192)

        # LSTM expects input: (B, seq_len, input_size=192)
        x, _ = self.lstm(x, (h0, c0))

        # Take the last time step's output
        x = x[:, -1, :]  # (B, 64)

        x = self.fc1(x)  # (B, y_len)
        forecast_y = F.softmax(x, dim=1)
        
        return forecast_y

