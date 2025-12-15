import pickle
import torch
from torch.utils.data import TensorDataset
import math
import numpy as np
from torch.utils.data import DataLoader


# data_dict = pickle.load(open('./data.pickle', 'rb'))
# len_train = math.floor(len(data_dict['data'])*0.8)
# #Extraction des données
# train_data = data_dict['data'][:len_train]
# train_label = np.array(data_dict['labels'][:len_train])

# # Il y a des éléments de taille 84 donc c'est chiant pour transformer en tenseur
# train_data_filtered = [data for data in train_data if len(data) == 42]
# filtered_indices_train = [index for index in range(len(train_data)) if len(train_data[index]) == 42]
# train_label_filtered = train_label[filtered_indices_train].tolist()

# test_data = data_dict['data'][len_train:]
# test_label = np.array(data_dict['labels'][len_train:])

# # Il y a des éléments de taille 84 donc c'est chiant pour transformer en tenseur
# test_data_filtered = [data for data in test_data if len(data) == 42]
# filtered_indices_test = [index for index in range(len(test_data)) if len(test_data[index]) == 42]
# test_label_filtered = test_label[filtered_indices_test].tolist()

# #Création des tenseurs
# train_tensor_x = torch.tensor(train_data_filtered, dtype=torch.float32).unsqueeze(1)
# train_tensor_y = torch.tensor(train_label_filtered, dtype=torch.long).unsqueeze(1)

# test_tensor_x = torch.tensor(test_data_filtered, dtype=torch.float32).unsqueeze(1)
# test_tensor_y = torch.tensor(test_label_filtered, dtype=torch.long).unsqueeze(1)

# #Création dataset
# train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
# test_dataset = TensorDataset(test_tensor_x, test_tensor_y)

# train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# for data, label in train_loader:
#     label = label.squeeze()
#     label_shape = label.size()
#     label_type = label.dtype
#     data_shape = data.size()
# print(f"""Analyse du dataLoader: \n
#       shape: {label_shape},\n
#       type: {label_type}""")
# print(data_shape)

# print(f"len de data_dict: {len(data_dict)}")
# print(f"longueur des data (même que la longueur des labels): {len(data_dict['labels'])}")
# print(f"len d'une data pour une image: {len(data_dict['data'][200])}")
# print(f"analyse de train_label: {train_label[:10]}")
# print(f"analyse type des labels: {data_dict['labels'][:5]}")
# print(f"analyse type des labels: {data_dict['labels'][5]}")
# print(f"analyse type des labels: {data_dict['labels'][-1]}")
# print(f"analyse type des labels: {data_dict['labels'][2000]}")
# print(len(train_data))
# print(len(train_data[0]))
# train_data_filtered = [data for data in train_data if len(data) == 42]
# print(len(train_data_filtered))


def create_dataset(data):
    """
    - Splitting the data (80% for training, 20% for test)
    - Transform it into tensors
    - Mix it to create two dataset so Dataloader will be able to read it 
        : one for the training, one for the test
    
    :param data: original data
    """
    len_train = math.floor(len(data['data'])*0.8)
    #Extraction des données
    train_data = data['data'][:len_train]
    train_label = np.array(data['labels'][:len_train])

    # Il y a des éléments de taille 84 donc c'est chiant pour transformer en tenseur
    train_data_filtered = [data for data in train_data if len(data) == 42]
    filtered_indices_train = [index for index in range(len(train_data)) if len(train_data[index]) == 42]
    train_label_filtered = train_label[filtered_indices_train].tolist()

    test_data = data['data'][len_train:]
    test_label = np.array(data['labels'][len_train:])

    # Il y a des éléments de taille 84 donc c'est chiant pour transformer en tenseur
    test_data_filtered = [data for data in test_data if len(data) == 42]
    filtered_indices_test = [index for index in range(len(test_data)) if len(test_data[index]) == 42]
    test_label_filtered = test_label[filtered_indices_test].tolist()

    #Création des tenseurs
    # features: shape [N, input_dim] (no channel dim)
    train_tensor_x = torch.tensor(train_data_filtered, dtype=torch.float32)
    # labels for CrossEntropyLoss must be LongTensor of shape [N]
    train_tensor_y = torch.tensor(train_label_filtered, dtype=torch.long)

    test_tensor_x = torch.tensor(test_data_filtered, dtype=torch.float32)
    test_tensor_y = torch.tensor(test_label_filtered, dtype=torch.long)

    #Création dataset
    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    test_dataset = TensorDataset(test_tensor_x, test_tensor_y)

    return train_dataset, test_dataset