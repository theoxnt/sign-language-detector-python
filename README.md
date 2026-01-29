# sign-language-detector-python

This project is a sign language recognition system. Its objective is to construct sentences letter by letter by recognizing the American Sign Language (ASL) alphabet.

Detailed desrcription : <br>
There are two ways to use the project. <br>
First, you can use an already trained model. <br>
Second, you can create and train your own model. <br>
In the second mode, when the project is launched, the user can choose between several actions:
- Data collection : By specifying the number of letters to collect, the program captures photos of the user’s hand signs. These images are later used to train the model.
- Dataset creation : By specifying a data file (generated in the previous step or imported manually), the program analyzes the images and produces a new dataset containing extracted features that represent the photos.
- Model training : By specifying a dataset (created in the previous step or imported), the program trains one of two types of models to recognize the ASL alphabet: a Random Forest model, or a Neural Network model.
- Model usage : By specifying a trained model, the user can use it to write a sentence letter by letter. A secondary AI component then corrects the sentence and displays the corrected version in the terminal when the program is closed.

## Installation

Clone the project : <br>
With HTTPS : 
```bash
https://github.com/theoxnt/sign-language-detector-python.git
```
With SSH : 
```bash
git@github.com:theoxnt/sign-language-detector-python.git
```

## Running the code 

Create a new virtual environment. For example with conda :
```bash
conda create --name sign_language_env
```
Activate it:
```bash
conda activate sign_language_env
```
And install the dependencies:
```bash
pip install .
```
Run the code <br>
If the user want to use the sign language detector :
```bash
python -m src.main
```
If the user want to create his own model : 
```bash
python -m src.main --edit
```

## Use it as a package in your project

You can import the sign_language_detector and run_optuna functions into your project using the following lines: <br>
The sign_language_detector function allows you to use the sign language detector directly in your code. It uses the camera to perform live sign language detection and returns the predicted sentence.
```bash
from SIGN-LANGUAGE-DETECTOR-PYTHON import sign_language_detector
```
The run_optuna function allows you to search for the best hyperparameters if you want to train your own neural network model for sign language detection.
```bash
from SIGN-LANGUAGE-DETECTOR-PYTHON import run_optuna
```
Alternatively, to import both functions at once, you can use:
```bash
from SIGN-LANGUAGE-DETECTOR-PYTHON import *
```



## Project structure 

```text
SIGN-LANGUAGE-DETECTOR-PYTHON
├── src/
│   ├── __init__.py
│   ├── BestNet.py           # the best neural network model
│   ├── cli.py               # command line treatment
│   ├── core.py              # Main function: entry point
│   ├── io_.py               # functions to ask questions to the user in the terminal and print results
│   ├── main.py              # main function to launch the project
│   ├── ModulableNet.py      # Neural network used by optuna to find the best model
│   ├── optuna_trainer.py
│   └── SimpleNet.py         # First neural netwok used
├── tests/
│   ├── test_cli.py
│   ├── test_core.py
│   ├── test_io.py
│   └── test_main.py
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```
## Tests
To run the tests, you need to install pytest : 
```bash
pip install pytest
```
You need to be in the test folder and execute : 
```bash
pytest
```

## License 
This project uses the MIT License


## venv activation
venv_py311\Scripts\activate


