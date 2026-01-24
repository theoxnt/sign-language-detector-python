import sys
from unittest.mock import MagicMock, call, patch

sys.path.append('..')

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
        ("", 2, "folder_name", ValueError, "num_classes should be an integer"),
        (0, 2, "folder_name", ValueError, "num_classes should be >= 1"),
        (2, "", "folder_name", ValueError, "imgs_per_class should be an integer"),
        (2, 0, "folder_name", ValueError, "imgs_per_class should be >= 1"),
        (2, 3, 4, ValueError, "folder_name should be a string"),
        (2, 3, "", ValueError, "folder_name should not be empty")

    ]
)
def test_collect_images_parameters_errors(num_classes, img_per_classes, folder_name, expected_exception, match_msg):
    with pytest.raises(expected_exception, match=match_msg):
        collect_images(num_classes, img_per_classes, folder_name)

@patch('src.core.cv2.VideoCapture')
def test_collect_images_camera_error(mock_videoCapture):
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
        ("", "dataset_name", ValueError, "number_of_classes should be an integer"),
        (0, "dataset_name", ValueError, "number_of_classes should be >= 1"),
        (2, 3, ValueError, "dataset_name should be a string"),
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
def test_train_classifier_forest(
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
    mock_pickle_load.assert_called_once()
    mock_train_forest.assert_called_once()
    mock_pickle_dump.assert_called_once()


@patch("builtins.open")
@patch("pickle.dump")
@patch("src.core.train_neural_network")
@patch("pickle.load")
def test_train_classifier_neural_network(
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
    mock_pickle_load.assert_called_once()
    mock_train_neural_network.assert_called_once()
    mock_pickle_dump.assert_called_once()




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
def test_train_neural_network():
    assert True




# -------------------------
# Tests for epoch_trainer
# -------------------------
def test_epoch_trainer():
    assert True




# -------------------------
# Tests for splitting_dataset
# -------------------------
def test_splitting_dataset():
    assert True




# -------------------------
# Tests for inference_classifier
# -------------------------
def test_inference_classifier():
    assert True
