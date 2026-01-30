from torch import nn

class BestNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 398),
            nn.GELU(),             
      
            nn.Linear(398, 440),
            nn.Tanh(),            
            
            nn.Linear(440, num_classes),
        )

    def forward(self, x):
        return self.net(x)
    
# Best solution according to optuna : lr = 0.0016, batch_size = 64, num_epoochs = 150