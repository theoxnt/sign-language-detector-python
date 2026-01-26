import pickle
import torch
from src.ModulableNet import ModulableNet
from src.core import splitting_dataset, epoch_trainer
import optuna
from torch.utils.data import DataLoader

def run_optuna(
        n_trials=30, 
        min_lr=1e-5, 
        max_lr=1e-1, 
        batch_size_list=[32, 64, 128], 
        min_epochs=5,
        max_epochs=250,
        min_numbers_layer=1,
        max_numbers_layer=7):
    """
    Run Optuna hyperparameter optimization for the ModulableNet model.

    Args:
        n_trials (int, optional): Number of Optuna trials to perform. Default is 30.
        min_lr (float, optional): Minimum learning rate to explore. Default is 1e-5.
        max_lr (float, optional): Maximum learning rate to explore. Default is 1e-1.
        batch_size_list (list[int], optional): List of batch sizes to choose from. Default is [32, 64, 128].
        min_epochs (int, optional): Minimum number of epochs. Default is 5.
        max_epochs (int, optional): Maximum number of epochs. Default is 250.
        min_numbers_layer (int, optional): Minimum number of hidden layers in the model. Default is 1.
        max_numbers_layer (int, optional): Maximum number of hidden layers in the model. Default is 7.

    Returns:
        tuple: A pair containing:
            - accuracy (float): Accuracy of the best trial on the test dataset.
            - params (dict): Dictionary of hyperparameters for the best trial.
    """
    
    def objective(trial):
        """Objective function for Optuna."""
        learning_rate = trial.suggest_loguniform('learning_rate', min_lr, max_lr)
        batch_size = trial.suggest_categorical('batch_size', batch_size_list)
        num_epochs = trial.suggest_int('num_epochs', min_epochs, max_epochs)
        n_layers = trial.suggest_int("n_layers", min_numbers_layer, max_numbers_layer)

        hidden_dims = []
        activations = []

        for i in range(n_layers):
            hidden_dims.append(
                trial.suggest_int(f"hidden_dim_{i}", 64, 512)
            )
            activations.append(
                trial.suggest_categorical(
                    f"activation_{i}",
                    ["relu", "leaky_relu", "gelu", "elu", "tanh"]
                )
            )

        model = ModulableNet(input_dim=42, num_classes=25,
                             hidden_dims=hidden_dims, activations=activations)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        data_dict = pickle.load(open('./src/data_pickle/data_25.pickle', 'rb'))
        train_dataset, test_dataset = splitting_dataset(data_dict)
        model.train()
        loss = []

        for epoch in range(num_epochs):
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            total_loss = epoch_trainer(model, optimizer, train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")
            loss.append(total_loss)

        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)
        model.eval()
        true_preds = 0
        predictions = []
        y_true = []
        for x, y in test_loader:
            preds = torch.argmax(model(x), dim=1)
            true_preds_iter = (preds == y).sum().item()
            predictions.extend(preds.tolist())
            y_true.extend(y.tolist())
            true_preds += true_preds_iter
        accuracy = true_preds / len(test_dataset)
        return accuracy

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value}")
    print(f"  Params: {trial.params}")
    return (trial.value, trial.params)

if __name__ == "__main__":
    run_optuna()
