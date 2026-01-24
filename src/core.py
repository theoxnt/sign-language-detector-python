import os
import cv2
from pathlib import Path
import pickle
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import torch
from src.BestNet import BestNet
from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import time
import language_tool_python
from torch.utils.data import TensorDataset
import math


def collect_images(num_classes, imgs_per_class, folder_name):
    """
    Collect images from webcam and save them into folders named from 0 to num_classes-1
    
    Args:
        num_classes (int): Number of classes (letters) to collect images for
        imgs_per_class (int): Number of images to collect per class
        folder_name (str): Name of the folder to save the images
    
    Returns:
        bool: True if the images were collected successfully, False otherwise
    """
    if type(num_classes) is not int: 
        raise ValueError("num_classes should be an integer")
    
    if num_classes < 1:
        raise ValueError("num_classes should be >= 1")
    
    if type(imgs_per_class) is not int: 
        raise ValueError("imgs_per_class should be an integer")
    
    if imgs_per_class < 1:
        raise ValueError("imgs_per_class should be >= 1") 
    
    if type(folder_name) is not str: 
        raise ValueError("folder_name should be a string")
    
    if folder_name == "":
        raise ValueError("folder_name should not be empty")
    
    DATA_DIR = os.path.join('./src/data', folder_name)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    
    for j in range(num_classes):
        if not os.path.exists(os.path.join(DATA_DIR, str(j))):
            os.makedirs(os.path.join(DATA_DIR, str(j)))

        print(f'Collecting data for class {j}')
        
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Can't receive frame (ret == False)")
            cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break

        counter = 0
        while counter < imgs_per_class:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Can't receive frame (ret == False)")
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(DATA_DIR, str(j), 'f{counter}.jpg'), frame)

            counter += 1

    cap.release()
    cv2.destroyAllWindows()
    return True


def create_dataset(number_of_classes, dataset_name):
    """
    Create a dataset from the collected images and save it as a pickle file
    
    Args:
        number_of_classes (int): Number of classes (letters) in the dataset to be created
        dataset_name (str): Name of the dataset file to be created (without extension)
    
    Returns:
        bool: True if the dataset was created successfully, False otherwise
    """
    if type(number_of_classes) is not int: 
        raise ValueError("number_of_classes should be an integer")
    
    if number_of_classes < 1:
        raise ValueError("number_of_classes should be >= 1")

    if type(dataset_name) is not str:
        raise ValueError("dataset_name should be a string")

    if dataset_name == "":
        raise ValueError("dataset_name should not be empty")
    mp_hands = mp.solutions.hands

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    DATA_DIR = './src/data'
    if not os.path.exists(DATA_DIR):
        raise RuntimeError(f"There isn't a data folder at : {DATA_DIR}")

    data = []
    labels = []
    for actual_label in range(number_of_classes):

        print(f'Creating dataset for label : {actual_label}')

        if len(os.listdir(DATA_DIR)) == 0:
            raise RuntimeError(f'{DATA_DIR} is empty')

        for dir_ in os.listdir(DATA_DIR):
            print(f'Processing folder: {dir_}')
            DATA_DIR_path = os.path.join(DATA_DIR, dir_)

            if len(os.listdir(DATA_DIR_path)) == 0:
                raise RuntimeError(f'{DATA_DIR_path} is empty')

            for img_path in os.listdir(os.path.join(DATA_DIR_path, str(actual_label))):
                data_aux = []

                x_ = []
                y_ = []
                img = cv2.imread(os.path.join(DATA_DIR_path, str(actual_label), img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                results = hands.process(img_rgb)

                if not results:
                    raise RuntimeError("Hands process didn't work")

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                data.append(data_aux)
                labels.append(actual_label)
    DATASET_DIR = './src/data_pickle'
    Path(DATASET_DIR).mkdir(parents=True, exist_ok=True)
    with open(f'{DATASET_DIR}/{dataset_name}.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    return True



def train_classifier(dataset_file, type, num_classes=None):
    """
    Train a classifier (random forest or neural network) on the given dataset and save the trained model
    
    Args:
        dataset_file (str): Name of the dataset file (without extension) to be used for training
        type (str): Type of classifier to train ('f' for random forest, 'n' for neural network)
        num_classes (int, optional): Number of classes in the dataset (required if type is 'n')
    
    Returns:
        bool: True if the model was trained and saved successfully, False otherwise
    """
    BASE_DIR = './src/data_pickle'
    DATA_PATH = f'{BASE_DIR}/{dataset_file}.pickle'
    with open(DATA_PATH, 'rb') as f:
        data_dict = pickle.load(f)
    if type == 'f':
        model = train_forest(data_dict)
    elif type == 'n':
        model = train_neural_network(data_dict, num_classes)
    MODEL_DIR = './src/models'
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    with open(f'{MODEL_DIR}/model_{type}.p', 'wb') as f:
        pickle.dump(model, f)
    return True

def train_forest(data_dict):
    """
    Train a Random Forest classifier on the given dataset
    
    Args:
        data_dict (dict): Dictionary containing 'data' and 'labels' for training
        
    Returns:
        model (RandomForestClassifier): Trained Random Forest model
    """
    if not isinstance(data_dict, dict):
        raise TypeError("data_dict must be a dictionary")
    
    if not data_dict:
        raise ValueError("Data_dict should not be empty")
    
    if 'data' not in data_dict or 'labels' not in data_dict:
        raise KeyError("data_dict must contain 'data' and 'labels'")
    
    if not data_dict['data']:
        raise ValueError("Data in data_dict should not be empty")
    
    if not data_dict['labels']:
        raise ValueError("Labels in data_dict should not be empty")
    
    if len(data_dict['data']) != len(data_dict['labels']):
        raise ValueError("Data and labels do not have the same length")
    
    data_filtered = [
        data_dict['data'][i] 
        for i in range (len(data_dict['data'])) 
        if len(data_dict['data'][i]) == 42
        ]

    if len(data_filtered) == 0:
        raise ValueError("No samples with 42 features after filtering")

    labels_filtered = [
        data_dict['labels'][i] 
        for i in range (len(data_dict['data'])) 
        if len(data_dict['data'][i]) == 42
        ]
    
    data = np.asarray(data_filtered)
    labels = np.asarray(labels_filtered)

    if len(np.unique(labels)) < 2:
        raise ValueError("At least two classes are required for training")

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    model = RandomForestClassifier()

    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    score = accuracy_score(y_predict, y_test)

    print(f'{score*100}% of samples were classified correctly !')

    return model


def train_neural_network(data_dict, num_classes):
    """
    Train a Neural Network classifier on the given dataset
    
    Args:
        data_dict (dict): Dictionary containing 'data' and 'labels' for training
        num_classes (int): Number of classes in the dataset
        
    Returns:
        model (BestNet): Trained Neural Network model
    """

    train_dataset, test_dataset = splitting_dataset(data_dict)

    #42 because it's the number of landmarks on each image
    model = BestNet(42, num_classes) 

    opt = optim.SGD(model.parameters(), lr=0.0016, momentum=0.9)
    loss = []
    for epoch in range(150):
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        total_loss = epoch_trainer(model, opt, train_loader)
        loss.append(total_loss)
        print(f"epoch {epoch + 1} : loss = {total_loss}")    

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
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

    return model

def epoch_trainer(model, opt, data):
    """
    Train the model on the data using the optimizer opt for one epoch

    Args:
        model (nn.Module): The neural network model to be trained
        opt (torch.optim.Optimizer): The optimizer to use for training
        data (DataLoader): The data to train on

    Returns:
        total_loss (float): The total loss for the epoch
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


def splitting_dataset(data):
    """
    Process the data to create training and test datasets.
    - Filter out data with incorrect size
    - Splitting the data (80% for training, 20% for test)
    - Transform it into tensors
    - Mix it to create two dataset so Dataloader will be able to read it 
        : one for the training, one for the test
    
    Args:
        data (dict): Dictionary containing 'data' and 'labels' for processing
    
    Returns:
        train_dataset (TensorDataset): Training dataset
        test_dataset (TensorDataset): Test dataset
    """

    #Filter data with incorrect size
    data_filtered = [data['data'][i] for i in range (len(data['data'])) if len(data['data'][i]) == 42]
    labels_filtered = [data['labels'][i] for i in range (len(data['data'])) if len(data['data'][i]) == 42]
    data['data'] = data_filtered
    data['labels'] = labels_filtered

    #80% for training, 20% for test
    len_train = math.floor(len(data['data'])*0.8)

    #Exctract the data
    train_data = data['data'][:len_train]
    train_label = np.array(data['labels'][:len_train])
    test_data = data['data'][len_train:]
    test_label = np.array(data['labels'][len_train:])

    #Transform to tensor
    train_tensor_x = torch.tensor(train_data, dtype=torch.float32)
    train_tensor_y = torch.tensor(train_label, dtype=torch.long)

    test_tensor_x = torch.tensor(test_data, dtype=torch.float32)
    test_tensor_y = torch.tensor(test_label, dtype=torch.long)

    #Dataset creation
    train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
    test_dataset = TensorDataset(test_tensor_x, test_tensor_y)

    return train_dataset, test_dataset


def inference_classifier(type: str):
    """
    Use the trained model to perform inference on live webcam data.
    
    Args:
        type (str): Type of classifier to use ('f' for random forest, 'n' for neural network)
        
    Returns:
        print the predicted sentence and corrected text
        bool: True if inference was performed successfully, False otherwise
    """
    tool = language_tool_python.LanguageTool('en-US')

    try:
        MODEL_DIR = './src/models'
        model = pickle.load(open(f'{MODEL_DIR}/model_{type}.p', 'rb'))
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        return False

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y', 24: ' '}
    while True:
            ret, frame = cap.read()
            cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break
    sentence_predicted = ""
    last_time = time.time()
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Finished? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
        cv2.putText(frame, sentence_predicted, (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
        if cv2.waitKey(25) == ord('q'):
            break

        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  
                    hand_landmarks,  
                    mp_hands.HAND_CONNECTIONS, 
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            if type == 'f':
                prediction = model.predict([np.asarray(data_aux)])
            elif type == 'n':
                prediction = model(torch.FloatTensor([np.asarray(data_aux)]))
                prediction = torch.argmax(prediction, dim=1)
            else:
                raise ValueError("Invalid model type. Choose 'forest' or 'n'.")
            predicted_character = labels_dict[int(prediction[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            elapsed = time.time() - last_time
            if elapsed >= 3:
                sentence_predicted += predicted_character
                last_time = time.time()

        cv2.imshow('frame', frame)
        cv2.waitKey(1)


    cap.release()
    cv2.destroyAllWindows()
    print("Predicted sentence: ", sentence_predicted)
    corrected_text = tool.correct(sentence_predicted)
    print('corrected text: ', corrected_text) 
    return True

