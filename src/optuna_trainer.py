import pickle
import torch
from src.ModulableNet import ModulableNet
from src.core import splitting_dataset, epoch_trainer
import optuna
from torch.utils.data import DataLoader

def run_optuna(n_trials=30):
    """Run Optuna hyperparameter optimization for the ModulableNet."""
    
    def objective(trial):
        """Objective function for Optuna."""
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
        num_epochs = trial.suggest_int('num_epochs', 5, 250)
        n_layers = trial.suggest_int("n_layers", 1, 7)

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

if __name__ == "__main__":
    run_optuna()
