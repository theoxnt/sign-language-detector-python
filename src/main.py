from src.cli import ask_user, get_parser
from src.core import inference_classifier
from src.io_ import print_prompt


def sign_language_detector_cli():
    parser = get_parser()
    args = parser.parse_args()

    if args.edit:
        print_prompt(
            "\nWelcome to the Sign Language Detector!\n" "What do you want to do?\n"
        )
        ask_user()
    else:
        inference_classifier("f")


if __name__ == "__main__":
    sign_language_detector_cli()


def sign_language_detector(classifier_type):
    """
    Call the function inference_classifier to use the trained model to perform inference on live webcam data.

    Args:
        type_classifier (str): Type of classifier to use ('f' for random forest, 'n' for neural network)

    Returns:
        corrected_sentence: The predicted sentence, corrected by an other IA
    """
    if type(classifier_type) is not str:
        raise TypeError("classifier_type must be a string")
    if not (classifier_type == "f" or classifier_type == "n"):
        raise ValueError("classifier type must be equal to 'f' or 'n'")

    return inference_classifier(classifier_type)
