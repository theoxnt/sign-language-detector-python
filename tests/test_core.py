import sys
from unittest.mock import MagicMock, call, patch

sys.path.append('..')

import mediapipe as mp
from src.core import *
import numpy as np
from pathlib import Path
import pytest


# -------------------------
# -------Fixtures----------
# -------------------------

@pytest.fixture
def data_dict():
    num_classes = 3
    samples_per_class = 10
    num_features = 42

    data = []
    labels = []

    for label in range(num_classes):
        for _ in range(samples_per_class):
            sample = np.random.rand(num_features).tolist()
            data.append(sample)
            labels.append(label)

            sample = np.random.rand(num_features + 1).tolist()
            data.append(sample)
            labels.append(label)

    return {
        'data': data,
        'labels': labels
    }

@pytest.fixture
def train_loader(data_dict):
    train_dataset, _ = splitting_dataset(data_dict)
    return DataLoader(train_dataset, batch_size=64, shuffle=True)


# -------------------------
# -------Tests-------------
# -------------------------


# -------------------------
# Tests for collect_images
# -------------------------
@patch('src.core.cv2.VideoCapture')
@patch('src.core.cv2.imshow')
@patch('src.core.cv2.waitKey')
@patch('src.core.cv2.imwrite')
@patch('src.core.cv2.destroyAllWindows')
@patch('src.core.os.makedirs')
@patch('src.core.os.path.exists')
def test_collect_images_ok(
    mock_exists,
    mock_makedirs,
    mock_destroyAllWindows,
    mock_imwrite,
    mock_waitKey,
    mock_imshow,
    mock_VideoCapture
):

    mock_VideoCapture.return_value.isOpened.return_value = True
    mock_exists.return_value = False

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_VideoCapture.return_value.read.return_value = (True, fake_frame)

    mock_waitKey.return_value = ord('q')

    num_classes = 2
    images_per_class = 3
    folder_name = "test_folder"

    result = collect_images(num_classes, images_per_class, folder_name)

    assert result is True
    assert mock_imwrite.call_count == num_classes * images_per_class
    assert mock_makedirs.call_count == num_classes + 1

    expected_calls = [
        call(f'./src/data/{folder_name}'),
        call(f'./src/data/{folder_name}/0'),
        call(f'./src/data/{folder_name}/1')
    ]
    mock_makedirs.assert_has_calls(expected_calls, any_order=False)

    assert mock_VideoCapture.return_value.isOpened.call_count == 1
    assert mock_VideoCapture.return_value.release.call_count == 1
    assert mock_destroyAllWindows.call_count == 1

@pytest.mark.parametrize(
    "num_classes, img_per_classes, folder_name, expected_exception, match_msg",
    [
        ("", 2, "folder_name", TypeError, "num_classes should be an integer"),
        (0, 2, "folder_name", ValueError, "num_classes should be >= 1"),
        (2, "", "folder_name", TypeError, "imgs_per_class should be an integer"),
        (2, 0, "folder_name", ValueError, "imgs_per_class should be >= 1"),
        (2, 3, 4, TypeError, "folder_name should be a string"),
        (2, 3, "", ValueError, "folder_name should not be empty")

    ]
)
def test_collect_images_parameters_errors(num_classes, img_per_classes, folder_name, expected_exception, match_msg):
    with pytest.raises(expected_exception, match=match_msg):
        collect_images(num_classes, img_per_classes, folder_name)

@patch('src.core.cv2.VideoCapture')
@patch('src.core.os.path.exists')
def test_collect_images_camera_error(mock_exists, mock_videoCapture):
    mock_exists.return_value = True
    mock_videoCapture.return_value.isOpened.return_value = False
    with pytest.raises(RuntimeError, match="Cannot open camera"):
        collect_images(2, 3, "folder_name")

@patch('src.core.cv2.VideoCapture')
@patch('src.core.os.path.exists')
@patch('src.core.os.makedirs')
def test_collect_images_read_error(
    mock_makedirs,
    mock_exists,
    mock_VideoCapture
):

    mock_VideoCapture.return_value.isOpened.return_value = True
    mock_exists.return_value = False

    mock_VideoCapture.return_value.read.return_value = (False, None)

    with pytest.raises(RuntimeError, match="Can't receive frame"):
        collect_images(2, 3, "folder_name")



# -------------------------
# Tests for create_dataset
# -------------------------
@patch("mediapipe.solutions.hands.Hands.process")
@patch("os.listdir")
@patch("os.path.exists")
@patch("cv2.imread")
@patch("cv2.cvtColor")
@patch("pathlib.Path.mkdir")
@patch("builtins.open")
@patch("pickle.dump")
def test_create_dataset_ok(
    mock_pickle,
    mock_open,
    mock_mkdir,
    mock_cvtColor,
    mock_imread,
    mock_path_exists,
    mock_listdir,
    mock_hands_process
):

    mock_path_exists.return_value = True

    def listdir_side_effect(path):
        if path == "./src/data":
            return ["test_folder_1", "test_folder_2"]
        elif "test_folder_1" in path or "test_folder_2" in path:
            return ["0", "1", "2"]
        elif "0" in path or "1" in path:
            return ["0.jpg", "1.jpg", "2.jpg"]
        return []

    mock_listdir.side_effect = listdir_side_effect

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_imread.return_value = fake_frame
    mock_cvtColor.return_value = fake_frame

    fake_hand_landmarks = MagicMock()
    fake_hand_landmarks.landmark = [
        MagicMock(x=0.1 * i, y=0.05 * i) for i in range(21)
    ]
    mock_hands_process.return_value = MagicMock(
        multi_hand_landmarks=[fake_hand_landmarks]
    )

    result = create_dataset(number_of_classes=3, dataset_name="dataset_test")

    assert result is True
    mock_mkdir.assert_called_once()
    mock_open.assert_called_once()
    mock_open.assert_has_calls([
        call(f'./src/data_pickle/dataset_test.pickle', 'wb')
    ])
    mock_pickle.assert_called_once()
    assert mock_listdir.call_count == 18

@pytest.mark.parametrize(
    "number_of_classes, dataset_name, expected_exception, match_msg",
    [
        ("", "dataset_name", TypeError, "number_of_classes should be an integer"),
        (0, "dataset_name", ValueError, "number_of_classes should be >= 1"),
        (2, 3, TypeError, "dataset_name should be a string"),
        (2, "", ValueError, "dataset_name should not be empty")
    ]
)
def test_create_dataset_errors_parameters(
    number_of_classes, 
    dataset_name, 
    expected_exception, 
    match_msg
    ):

    with pytest.raises(expected_exception, match=match_msg):
        create_dataset(number_of_classes, dataset_name)

@patch("src.core.os.path.exists")
def test_create_dataset_data_dir_not_exist(mock_exists):
    mock_exists.return_value = False
    with pytest.raises(RuntimeError, match="There isn't a data folder at"):
        create_dataset(2, "dataset_name")

@patch("src.core.os.path.exists")
@patch("src.core.os.listdir")
def test_create_dataset_folder_empty(mock_listdir, mock_exists):
    mock_exists.return_value = True
    mock_listdir.return_value = []
    with pytest.raises(RuntimeError, match="is empty"):
        create_dataset(2, "dataset_folder")

@patch("src.core.os.path.exists")
@patch("src.core.os.listdir")
def test_create_dataset_folder_empty_2(mock_listdir, mock_exists):
    mock_exists.return_value = True
    def listdir_side_effect(path):
        if path == "./src/data":
            return ["test_folder_1", "test_folder_2"]
        elif "test_folder_1" in path or "test_folder_2" in path:
            return []
    mock_listdir.side_effect = listdir_side_effect
    with pytest.raises(RuntimeError, match="is empty"):
        create_dataset(2, "dataset_folder")

@patch("src.core.os.path.exists")
@patch("src.core.os.listdir")
@patch("src.core.cv2.imread")
@patch("src.core.cv2.cvtColor")
@patch("mediapipe.solutions.hands.Hands.process")
def test_create_dataset_hands_process_error(
    mock_hands_process,
    mock_cvtColor, 
    mock_imread, 
    mock_listdir, 
    mock_exists
    ):

    mock_exists.return_value = True

    def listdir_side_effect(path):
        if path == "./src/data":
            return ["test_folder_1", "test_folder_2"]
        elif "test_folder_1" in path or "test_folder_2" in path:
            return ["0", "1", "2"]
        elif "0" in path or "1" in path:
            return ["0.jpg", "1.jpg", "2.jpg"]
        return []
    mock_listdir.side_effect = listdir_side_effect

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_imread.return_value = fake_frame
    mock_cvtColor.return_value = fake_frame

    mock_hands_process.return_value = None

    with pytest.raises(RuntimeError, match="Hands process didn't work"):
        create_dataset(2, "dataset_folder")

# -------------------------
# Tests for train_classifier
# -------------------------

@patch("builtins.open")
@patch("pickle.dump")
@patch("src.core.train_forest")
@patch("pickle.load")
@patch("src.core.Path.mkdir")
def test_train_classifier_forest_ok(
    mock_path_mkdir,
    mock_pickle_load,
    mock_train_forest,
    mock_pickle_dump,
    mock_open
):

    dataset_file = "dataset_test"
    training_type = 'f'
    num_classes = 2

    result = train_classifier(dataset_file, training_type, num_classes)

    assert result is True
    assert mock_open.call_count == 2 
    expected_calls= [
        call('./src/data_pickle/dataset_test.pickle', 'rb'),
        call('./src/models/model_f.p', 'wb')
    ]
    assert mock_open.has_calls(expected_calls)
    mock_pickle_load.assert_called_once()
    mock_train_forest.assert_called_once()
    mock_pickle_dump.assert_called_once()
    mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True)


@patch("builtins.open")
@patch("pickle.dump")
@patch("src.core.train_neural_network")
@patch("pickle.load")
@patch("src.core.Path.mkdir")
def test_train_classifier_neural_network_ok(
    mock_path_mkdir,
    mock_pickle_load,
    mock_train_neural_network,
    mock_pickle_dump,
    mock_open
):

    dataset_file = "dataset_test"
    training_type = 'n'
    num_classes = 2

    result = train_classifier(dataset_file, training_type, num_classes)

    assert result is True
    assert mock_open.call_count == 2 
    expected_calls= [
        call('./src/data_pickle/dataset_test.pickle', 'rb'),
        call('./src/models/model_n.p', 'wb')
    ]
    assert mock_open.has_calls(expected_calls)
    mock_pickle_load.assert_called_once()
    mock_train_neural_network.assert_called_once()
    mock_pickle_dump.assert_called_once()
    mock_path_mkdir.assert_called_once_with(parents=True, exist_ok=True)

@pytest.mark.parametrize(
        "dataset_file, type_classifier, num_classes, expected_errors, match_msg",
        [
            (1, "f", None, TypeError, "dataset_file should be a string"),
            ("", "f", None ,ValueError, "dataset_file should not be empty"),
            ("dataset_file", 1, None, TypeError, "type_classifier should be a string"),
            ("dataset_file", "dataset_file", None, ValueError, "type_classifier should be equal to 'f' or 'n'"),
            ("dataset_file", 'f', "", TypeError, "num_classes should be an integer"),
            ("dataset_file", 'f', 0, ValueError, "num_classes should be >= 1")
        ]
)
def test_train_classifier_errors_parameters(dataset_file, type_classifier, num_classes, expected_errors, match_msg):
    with pytest.raises(expected_errors, match=match_msg):
        train_classifier(dataset_file, type_classifier, num_classes)




# -------------------------
# Tests for train_forest
# -------------------------
def test_train_forest_ok(data_dict):
    model = train_forest(data_dict)
    assert isinstance(model, RandomForestClassifier)

    predict = model.predict(np.random.rand(1, 42))
    assert len(predict) == 1

@pytest.mark.parametrize(
    "data_dict, expected_exception, match_msg",
    [
        ([], TypeError, "data_dict must be a dictionary"),
        ({}, ValueError, "Data_dict should not be empty"),
        ({'wrong_key1': [], 'wrong_key2': []}, KeyError, "data_dict must contain 'data' and 'labels'"),
        ({'data': [], 'labels': []}, ValueError, "Data in data_dict should not be empty"),
        ({'data': [1], 'labels': []}, ValueError, "Labels in data_dict should not be empty"),
        ({'data': [1, 2], 'labels': [3]}, ValueError, "Data and labels do not have the same length"),
        ({'data': [[1, 2], [3, 4]], 'labels': [0, 1]}, ValueError, "No samples with 42 features after filtering"),
        ({'data': [np.random.rand(42)], 'labels': [4]}, ValueError, "At least two classes are required for training")
    ]
)
def test_train_forest_errors(data_dict, expected_exception, match_msg):
    with pytest.raises(expected_exception, match=match_msg):
        train_forest(data_dict)




# -------------------------
# Tests for train_neural_network
# -------------------------
@patch("src.core.epoch_trainer")
def test_train_neural_network_ok(mock_epoch_trainer, data_dict):
    model = train_neural_network(data_dict, 3)
    assert isinstance(model, BestNet)
    assert mock_epoch_trainer.call_count == 150

@pytest.mark.parametrize(
    "data_dict, num_classes, expected_exception, match_msg",
    [
        ([], 2, TypeError, "data_dict must be a dictionary"),
        ({}, 2, ValueError, "Data_dict should not be empty"),
        ({'wrong_key1': [], 'wrong_key2': []}, 2, KeyError, "data_dict must contain 'data' and 'labels'"),
        ({'data': [], 'labels': []}, 2, ValueError, "Data in data_dict should not be empty"),
        ({'data': [1], 'labels': []}, 2, ValueError, "Labels in data_dict should not be empty"),
        ({'data': [1, 2], 'labels': [3]}, 2, ValueError, "Data and labels do not have the same length"),
    ]
)
def test_train_naural_network_errors(data_dict, num_classes, expected_exception, match_msg):
    with pytest.raises(expected_exception, match=match_msg):
        train_neural_network(data_dict, num_classes)




# -------------------------
# Tests for epoch_trainer
# -------------------------
def test_epoch_trainer_ok(train_loader):
    model = BestNet(42, 3)
    opt = optim.SGD(model.parameters(), lr=0.0016, momentum=0.9)

    model.eval()
    total_loss = epoch_trainer(model, opt, train_loader)

    assert type(total_loss) is float
    assert model.training is True
    assert math.isfinite(total_loss)
    assert total_loss >= 0

@pytest.mark.parametrize(
        "model, opt, data, expected_errors, match_msg",
        [
            ("", "", "", TypeError, "model must be an instance of nn.Module"),
            (BestNet(42, 3), "", "", TypeError, "opt must be an instance of torch.optim.Optimizer"),
            (BestNet(42, 3), optim.SGD(BestNet(42, 3).parameters(), lr=0.0016, momentum=0.9), "", TypeError, "data must be an instance of torch.utils.data.DataLoader"),
        ]
)
def test_epoch_trainer_errors_parameters(model, opt, data, expected_errors, match_msg):
    with pytest.raises(expected_errors, match=match_msg):
        epoch_trainer(model, opt, data)


# -------------------------
# Tests for splitting_dataset
# -------------------------
def test_splitting_dataset(data_dict):
    train_dataset, test_dataset = splitting_dataset(data_dict)

    assert isinstance(train_dataset, TensorDataset)
    assert isinstance(test_dataset, TensorDataset)

    assert len(train_dataset) + len(test_dataset) == len(data_dict['data'])
    assert len(train_dataset) != 0 
    assert len(test_dataset) != 0 
    total = len(data_dict['data'])
    assert len(train_dataset) == math.floor(0.8 * total)
    assert len(test_dataset) == total - len(train_dataset)

    x, y = train_dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (42,)
    assert y.dim() == 0 


@pytest.mark.parametrize(
    "data_dict, expected_exception, match_msg",
    [
        ([], TypeError, "data must be a dictionary"),
        ({}, ValueError, "data should not be empty"),
        ({'wrong_key1': [], 'wrong_key2': []}, KeyError, "data must contain 'data' and 'labels'"),
        ({'data': [], 'labels': []}, ValueError, "Data in data should not be empty"),
        ({'data': [1], 'labels': []}, ValueError, "Labels in data should not be empty"),
        ({'data': [1, 2], 'labels': [3]}, ValueError, "Data and labels do not have the same length"),
        ({'data': [[1, 2], [3, 4]], 'labels': [0, 1]}, ValueError, "No samples with 42 features after filtering"),
        ({'data': [np.random.rand(42)], 'labels': [4]}, ValueError, "At least two classes are required for training")
    ]
)
def test_splitting_dataset_errors_parameters(data_dict, expected_exception, match_msg):
    with pytest.raises(expected_exception, match=match_msg):
        splitting_dataset(data_dict)




# -------------------------
# Tests for inference_classifier
# -------------------------
@patch("src.core.cv2.VideoCapture")
@patch("src.core.pickle.load")
@patch("src.core.open")
@patch("src.core.cv2.waitKey")
@patch("src.core.cv2.cvtColor")
@patch("mediapipe.solutions.hands.Hands.process")
@patch("src.core.cv2.imshow")
@patch("mediapipe.solutions.drawing_utils.draw_landmarks")
@patch("mediapipe.solutions.drawing_styles")
@patch('src.core.cv2.destroyAllWindows')
def test_inference_classifier_ok(
    mock_destroy_allWindows,
    mock_drawing_styles, 
    mock_draw_landmarks,
    mock_imshow,
    mock_hands_process,
    mock_cvtColor,
    mock_waitKey,
    mock_open,
    mock_pickle_load,
    mock_videoCapture
):
    mock_open.return_value.__enter__.return_value = "fake_file"
    fake_model_f = MagicMock()
    mock_pickle_load.return_value = fake_model_f

    mock_videoCapture.return_value.isOpened.return_value = True

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_videoCapture.return_value.read.return_value = (True, fake_frame)

    mock_waitKey.side_effect = [ord('q'), 1, 1, ord('q')]

    mock_cvtColor.return_value = fake_frame

    fake_hand_landmarks = MagicMock()
    fake_hand_landmarks.landmark = [MagicMock(x=0.1*i, y=0.05*i) for i in range(21)]
    mock_hands_process.return_value = MagicMock(multi_hand_landmarks=[fake_hand_landmarks])

    fake_model_f.predict.return_value = [0] 

    prediction = inference_classifier('f')

    assert type(prediction) is str
    mock_destroy_allWindows.assert_called_once()
    mock_videoCapture.return_value.release.assert_called_once()

@pytest.mark.parametrize(
    "type_classifier, expected_errors, match_msg",
    [
        (1, TypeError, "type_classifier must be a string"),
        ("adeda", ValueError, "type_classifier must be equal to 'n' or 'f'")
    ]
)
def test_inference_classifier_errors_parameters(type_classifier, expected_errors, match_msg):
    with pytest.raises(expected_errors, match=match_msg):
        inference_classifier(type_classifier)

@patch("src.core.pickle.load")
@patch("src.core.open")
def test_inference_classifier_error_model_not_find(mock_open, mock_pickle_load):
    mock_open.side_effect = FileNotFoundError
    with pytest.raises(RuntimeError, match="Model file not found. Please train the model first."):
        inference_classifier('f')

@patch("src.core.cv2.VideoCapture")
@patch("src.core.pickle.load")
@patch("src.core.open")
def test_inference_classifier_error_camera_not_open(mock_open, mock_pickle_load, mock_videoCapture):
    mock_videoCapture.return_value.isOpened.return_value = False
    with pytest.raises(RuntimeError, match="Cannot open camera"):
        inference_classifier('f')

@patch("src.core.cv2.VideoCapture")
@patch("src.core.pickle.load")
@patch("src.core.open")
def test_inference_classifier_error_can_not_read_frame(mock_open, mock_pickle_load, mock_videoCapture):
    mock_videoCapture.return_value.isOpened.return_value = True
    mock_videoCapture.return_value.read.return_value = (False, None)
    with pytest.raises(RuntimeError, match="Can't read the frame"):
        inference_classifier('f')

@patch("src.core.cv2.VideoCapture")
@patch("src.core.pickle.load")
@patch("src.core.open")
@patch("src.core.cv2.waitKey")
@patch("src.core.cv2.cvtColor")
@patch("mediapipe.solutions.hands.Hands.process")
@patch("src.core.cv2.imshow")
def test_inference_classifier_error_hands_process_did_not_work(
    mock_imshow,
    mock_hands_process, 
    mock_cvtColor, 
    mock_waitKey, 
    mock_open, 
    mock_pickle_load, 
    mock_videoCapture
    ):

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_videoCapture.return_value.isOpened.return_value = True
    mock_videoCapture.return_value.read.return_value = (True, fake_frame)
    mock_waitKey.side_effect = [ord('q'), 1, ord('q')]
    mock_cvtColor.return_value = fake_frame
    mock_hands_process.return_value = None
    with pytest.raises(RuntimeError, match="Hands process didn't work"):
        inference_classifier('f')

@patch("src.core.cv2.VideoCapture")
@patch("src.core.pickle.load")
@patch("src.core.open")
@patch("src.core.cv2.waitKey")
@patch("src.core.cv2.cvtColor")
@patch("mediapipe.solutions.hands.Hands.process")
@patch("src.core.cv2.imshow")
@patch("mediapipe.solutions.drawing_utils.draw_landmarks")
@patch("mediapipe.solutions.drawing_styles")
def test_inference_classifier_error_bad_prediction_length(
    mock_drawing_styles, 
    mock_draw_landmarks,
    mock_imshow,
    mock_hands_process,
    mock_cvtColor,
    mock_waitKey,
    mock_open,
    mock_pickle_load,
    mock_videoCapture
):
    mock_open.return_value.__enter__.return_value = "fake_file"
    fake_model_f = MagicMock()
    mock_pickle_load.return_value = fake_model_f

    mock_videoCapture.return_value.isOpened.return_value = True

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_videoCapture.return_value.read.return_value = (True, fake_frame)

    mock_waitKey.side_effect = [ord('q'), 1, 1, ord('q')]

    mock_cvtColor.return_value = fake_frame

    fake_hand_landmarks = MagicMock()
    fake_hand_landmarks.landmark = [MagicMock(x=0.1*i, y=0.05*i) for i in range(21)]
    mock_hands_process.return_value = MagicMock(multi_hand_landmarks=[fake_hand_landmarks])

    fake_model_f.predict.return_value = [0, 1] 
    

    with pytest.raises(RuntimeError, match="The model didn't predict a single value"):
        inference_classifier('f')