import sys
from unittest.mock import MagicMock, call, patch

import pytest

sys.path.append("..")

from src.cli import ask_user, get_parser

from unittest.mock import patch

# -------------------------
# ----Test get_parser------
# -------------------------


def test_get_parser_has_edit_argument():
    parser = get_parser()
    args = parser.parse_args(["--edit"])
    assert args.edit is True


def test_get_parser_edit_default_false():
    parser = get_parser()
    args = parser.parse_args([])
    assert args.edit is False


# -------------------------
# -----Test ask_user-------
# -------------------------


@patch("src.cli.ask")
def test_ask_user_ko(mock_ask):
    mock_ask.return_value = "zieubf"

    with pytest.raises(ValueError, match="Option not recognized."):
        ask_user()


# -------------------------
# Collecting images tests
# -------------------------


@patch("src.cli.ask")
@patch("src.cli.collect_images")
@patch("src.cli.ask_user")
def test_ask_user_collect_images_ok(mock_ask_user, mock_action, mock_ask):
    mock_ask.side_effect = ["1", "10", "5", "data_collected_test", "5"]

    ask_user()

    mock_action.assert_called_once_with(5, 10, "data_collected_test")
    mock_ask_user.assert_called_once()


@patch("src.cli.ask")
@patch("src.cli.collect_images")
def test_ask_user_collect_images_ko(mock_action, mock_ask):
    mock_ask.side_effect = ["1", "10", "5", "data_collected_test"]
    mock_action.return_value = None

    with pytest.raises(RuntimeError, match="Collecting images didn't work"):
        ask_user()

    mock_action.assert_called_once_with(5, 10, "data_collected_test")


# -------------------------
# Creating dataset tests
# -------------------------


@patch("src.cli.ask")
@patch("src.cli.create_dataset")
@patch("src.cli.ask_user")
def test_ask_user_create_dataset_ok(mock_ask_user, mock_action, mock_ask):
    mock_ask.side_effect = ["2", "5", "dataset_pickle_test", "5"]

    ask_user()

    mock_action.assert_called_once_with(5, "dataset_pickle_test")
    mock_ask_user.assert_called_once()


@patch("src.cli.ask")
@patch("src.cli.create_dataset")
def test_ask_user_create_dataset_ko(mock_action, mock_ask):
    mock_ask.side_effect = [
        "2",
        "5",
        "dataset_pickle_test",
    ]
    mock_action.return_value = None

    with pytest.raises(RuntimeError, match="Creating dataset didn't work"):
        ask_user()

    mock_action.assert_called_once_with(5, "dataset_pickle_test")


# -------------------------
# Train classifier tests
# -------------------------


@patch("src.cli.ask")
@patch("src.cli.train_classifier")
@patch("src.cli.ask_user")
@patch("src.cli.os.path.exists")
def test_ask_user_train_classifier_forest_ok(
    mock_exists, mock_ask_user, mock_action, mock_ask
):
    mock_exists.return_value = True
    mock_ask.side_effect = ["3", "model_test", "f", "5"]

    ask_user()

    mock_action.assert_called_once_with("model_test", "f")
    mock_ask_user.assert_called_once()


@patch("src.cli.ask")
@patch("src.cli.train_classifier")
@patch("src.cli.ask_user")
@patch("src.cli.os.path.exists")
def test_ask_user_train_classifier_neural_network_ok(
    mock_exists, mock_ask_user, mock_action, mock_ask
):
    mock_exists.return_value = True
    mock_ask.side_effect = ["3", "model_test", "n", "5", "5"]

    ask_user()

    mock_action.assert_called_once_with("model_test", "n", 5)
    mock_ask_user.assert_called_once


@patch("src.cli.ask")
@patch("src.cli.os.path.exists")
def test_ask_user_train_classifier_ko_wrong_train_classifier_name(
    mock_exists, mock_ask
):
    mock_exists.return_value = True
    mock_ask.side_effect = ["3", "model_test", "czbefz"]

    with pytest.raises(
        ValueError, match="Invalid option selected. Please enter 'f' or 'n'."
    ):
        ask_user()


@patch("src.cli.ask")
@patch("src.cli.ask_user")
@patch("src.cli.os.path.exists")
def test_ask_user_train_classifier_ko_file_does_not_exists(
    mock_exists, mock_ask_user, mock_ask
):
    mock_exists.return_value = False
    mock_ask.side_effect = ["3", "model_test"]

    with pytest.raises(
        RuntimeError,
        match="Dataset file does not exist. Please create the dataset first.",
    ):
        ask_user()


@patch("src.cli.ask")
@patch("src.cli.train_classifier")
@patch("src.cli.os.path.exists")
def test_ask_user_train_classifier_forest_ko_train_classier_did_not_work(
    mock_exists, mock_action, mock_ask
):
    mock_exists.return_value = True
    mock_ask.side_effect = ["3", "model_test", "f"]
    mock_action.return_value = False

    with pytest.raises(RuntimeError, match="Model training failed."):
        ask_user()

    mock_action.assert_called_once_with("model_test", "f")


@patch("src.cli.ask")
@patch("src.cli.train_classifier")
@patch("src.cli.os.path.exists")
def test_ask_user_train_classifier_neural_network_ko_train_classier_did_not_work(
    mock_exists, mock_action, mock_ask
):
    mock_exists.return_value = True
    mock_ask.side_effect = ["3", "model_test", "n", "5"]
    mock_action.return_value = False

    with pytest.raises(RuntimeError, match="Model training failed."):
        ask_user()

    mock_action.assert_called_once_with("model_test", "n", 5)


# -------------------------
# Inference classifier tests
# -------------------------


@patch("src.cli.ask")
@patch("src.cli.inference_classifier")
@patch("src.cli.ask_user")
def test_ask_user_inference_classifier_forest_ok(mock_ask_user, mock_action, mock_ask):
    mock_ask.side_effect = [
        "4",
        "f",
        "5",
    ]

    ask_user()

    mock_action.assert_called_once_with("f")
    mock_ask_user.assert_called_once()


@patch("src.cli.ask")
@patch("src.cli.inference_classifier")
@patch("src.cli.ask_user")
def test_ask_user_inference_classifier_neural_network_ok(
    mock_ask_user, mock_action, mock_ask
):
    mock_ask.side_effect = [
        "4",
        "n",
        "5",
    ]

    ask_user()

    mock_action.assert_called_once_with("n")
    mock_ask_user.assert_called_once()


@patch("src.cli.ask")
def test_ask_user_inference_classifier_ko_wrong_classifier_type(mock_ask):
    mock_ask.side_effect = ["4", "daiae"]

    with pytest.raises(
        ValueError, match="Invalid option selected. Please enter 'f' or 'n'."
    ):
        ask_user()


@patch("src.cli.ask")
@patch("src.cli.inference_classifier")
@patch("src.cli.ask_user")
def test_ask_user_inference_classifier_ko_forest_did_not_work(
    mock_ask_user, mock_action, mock_ask
):
    mock_action.return_value = False
    mock_ask.side_effect = ["4", "f"]

    with pytest.raises(RuntimeError, match="Model inference failed."):
        ask_user()

    mock_action.assert_called_once_with("f")


@patch("src.cli.ask")
@patch("src.cli.inference_classifier")
@patch("src.cli.ask_user")
def test_ask_user_inference_classifier_ko_neural_network_did_not_work(
    mock_ask_user, mock_action, mock_ask
):
    mock_action.return_value = False
    mock_ask.side_effect = ["4", "n"]

    with pytest.raises(RuntimeError, match="Model inference failed."):
        ask_user()

    mock_action.assert_called_once_with("n")


# -------------------------
# Quitting tests
# -------------------------


@patch("src.cli.ask")
@patch("src.cli.print_prompt")
def test_ask_user_quitting_ok(mock_print_prompts, mock_ask):
    mock_ask.return_value = "5"

    ask_user()

    mock_print_prompts.assert_called_once_with("Quitting...")
