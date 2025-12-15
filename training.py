from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn 

"""
  Training model functions
"""
def train_model(model, opt, data):
    """
      Train the model on the data using the optimizer opt
    """
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for train_data, train_label in data:
        opt.zero_grad()
        predicted_labels = model(train_data)
        loss = criterion(predicted_labels, train_label)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss

def training(model, batch_size, learning_rate, nb_epochs, train_dataset):
  """
    Train the model using the hyperparameters provided as arguments
  """
  opt = optim.SGD(model.parameters(), lr=learning_rate)
  loss = []
  for epoch in range(nb_epochs):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    total_loss = train_model(model, opt, train_loader)
    loss.append(total_loss)
    print(f"epoch {epoch + 1} : loss = {total_loss}")
  return loss