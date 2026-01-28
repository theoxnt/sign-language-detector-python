import sys
from unittest.mock import MagicMock, call, patch

sys.path.append("..")
from src.io_ import *
import pytest


@pytest.mark.parametrize(
    "prompt, expected_errors, match_msg",
    [
        (1, TypeError, "prompt must be a string"),
        ("", ValueError, "prompt must be not empty"),
    ],
)
def test_ask_ko_parameters(prompt, expected_errors, match_msg):
    with pytest.raises(expected_errors, match=match_msg):
        ask(prompt)


@patch("src.io_.input")
def test_ask_ok_with_prompt(mock_input):

    ask("prompt")

    mock_input.assert_called_once_with("prompt")


@patch("src.io_.input")
def test_ask_ok_with_cast_type(mock_input):
    mock_input.return_value = 3

    ask("prompt", int)

    mock_input.assert_called_once_with("prompt")


@patch("src.io_.input")
def test_ask_ko_with_cast_type(mock_input):
    mock_input.return_value = "3"

    ask("prompt", int)

    mock_input.assert_called_once_with("prompt")


@patch("src.io_.input")
def test_ask_ko_with_cast_type(mock_input):
    mock_input.side_effect = ["fe", "3"]

    ask("prompt", int)

    assert mock_input.call_count == 2
    assert mock_input.has_called([call("prompt"), call("prompt")])


@patch("src.io_.input")
def test_ask_ok_with_all_parameters(mock_input):
    mock_input.return_value = "15"
    ask("prompt", int, 1, 100)

    mock_input.assert_called_once_with("prompt")


@patch("src.io_.input")
def test_ask_ko_all_parameters_inf_min(mock_input):
    mock_input.side_effect = ["5", "10"]

    ask("prompt", int, 7)

    assert mock_input.call_count == 2
    assert mock_input.has_called([call("prompt"), call("prompt")])


@patch("src.io_.input")
def test_ask_ko_all_parameters_sup_max(mock_input):
    mock_input.side_effect = ["40", "10"]

    ask("prompt", int, max=30)

    assert mock_input.call_count == 2
    assert mock_input.has_called([call("prompt"), call("prompt")])


@pytest.mark.parametrize(
    "prompt, cast_type, min, max, expected_errors, match_msg",
    [
        (
            "prompt",
            int,
            "string",
            5,
            TypeError,
            "min must be the same type as cast_type",
        ),
        (
            "prompt",
            int,
            5,
            "string",
            TypeError,
            "max must be the same type as cast_type",
        ),
    ],
)
def test_ask_ko_type_with_all_parameters(
    prompt, cast_type, min, max, expected_errors, match_msg
):
    with pytest.raises(expected_errors, match=match_msg):
        ask(prompt, cast_type, min, max)
