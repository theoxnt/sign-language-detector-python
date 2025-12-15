import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from training import training
from datasetAnalyse import create_dataset
from SimpleNet import SimpleNet
import torch
from torch.utils.data import DataLoader

data_dict = pickle.load(open('./data.pickle', 'rb'))
train_dataset, test_dataset = create_dataset(data_dict)

#42 because it's the number of landmarks on each image (I think) 
#and 3 because for the moment we just predict 3 letters (A, B and L)
model = SimpleNet(42, 3) 

training(model, 128, 0.1, 50, train_dataset)

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
print(f'Accuracy on test set: {accuracy * 100}%')
# score = accuracy_score(y_predict, y_test)

# print('{}% of samples were classified correctly !'.format(score * 100))

# f = open('model.p', 'wb')
# pickle.dump({'model': model}, f)
# f.close()

f = open('modelMachineLearning.p', 'wb')
pickle.dump({'model': model}, f)
f.close()

