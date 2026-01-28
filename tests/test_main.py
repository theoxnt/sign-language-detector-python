import sys
from unittest.mock import patch

sys.path.append("..")
import pytest

from src.main import *


@pytest.mark.parametrize(
    "classifier_type, expected_errors, match_msg",
    [
        (1, TypeError, "classifier_type must be a string"),
        ("zfzef", ValueError, "classifier type must be equal to 'f' or 'n'"),
    ],
)
def test_sign_language_detector_ko_parameters(
    classifier_type, expected_errors, match_msg
):
    with pytest.raises(expected_errors, match=match_msg):
        sign_language_detector(classifier_type)


@patch("src.main.inference_classifier")
def test_sign_language_detector_ok_with_forest_classifier(mock_inference):
    mock_inference.return_value = "test"

    predicted_sentence = sign_language_detector("f")

    mock_inference.assert_called_once_with("f")
    assert predicted_sentence == "test"


@patch("src.main.inference_classifier")
def test_sign_language_detector_ok_with_neural_network_classifier(mock_inference):
    mock_inference.return_value = "test"

    predicted_sentence = sign_language_detector("n")

    mock_inference.assert_called_once_with("n")
    assert predicted_sentence == "test"
