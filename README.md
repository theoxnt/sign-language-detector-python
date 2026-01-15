# sign-language-detector-python

This project is a sign language recognition system. Its objective is to construct sentences letter by letter by recognizing the American Sign Language (ASL) alphabet.

Detailed desrcription : <br>
When the project is launched, the user can choose between several actions: <br>
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

Create a new virtual environment, activate it and install the dependencies. For example with conda :
```bash
conda create --name sign_language_env
```
```bash
conda activate sign_language_env
```
```bash
pip install -r requirements.txt
```
Run the code 
```bash
python -m src.main
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
├── tests/                   # TODO
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

## License 
This project uses the MIT License


