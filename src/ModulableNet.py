from torch import nn

# Dictionnaire des activations possibles
activation_dict = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
}

class ModulableNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims, activations):
        """
        input_dim : dimension de l'entrée
        num_classes : nombre de classes en sortie
        hidden_dims : liste des tailles des couches cachées
        activations : liste des activations pour chaque couche cachée (doit correspondre à hidden_dims)
        """
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hid_dim, act_name in zip(hidden_dims, activations):
            layers.append(nn.Linear(prev_dim, hid_dim))
            layers.append(activation_dict[act_name]())
            prev_dim = hid_dim

        # Couche finale
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)