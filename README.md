

# Ensembles: Gradient boosting, random forests, bagging, voting, stacking

This project contains the analysis of the several models, adjustment of the hyperparameters, quality metric, evaluation of the influence of the sample volume on the model quality. The models are implemented in Python and stored the evaluations in Jupyter notebooks (.ipynb files). 

## Project Structure

├── README.md
├── data
│   └── dataset_Malicious_and_Benign_Websites.csv
├── main.py
├── models
│   ├── malicious_benign_classification.py
│   ├── malicious_benign_grid.py
│   ├── random_classification.py
│   └── random_classification_grid.py
├── notebooks
│   ├── best_model_conclusion.ipynb
│   ├── eva_on_quality_metrics.ipynb
│   ├── eval_trainings.ipynb
│   └── volume_evaluations.ipynb
├── requirements.txt
└── utility.py



## Instructions

### Setting Up Environment

1. Clone this repository to your local machine.

```bash
git clone https://github.com/mcndoungerii/hw05.git

```

2. Create a virtual environment.

### python -m venv venv

3. Activate the virtual environment.
    On Windows:
    ```bash
        venv\Scripts\activate
    ```
    On macOS and Linux:
    ```bash
        source venv/bin/activate
    ```
4. Install required packages.
    ```bash
    pip install -r requirements.txt
    ```

### Running Models

1. Navigate to the models directory.
    
    cd models

2. Choose the model you want to run.
3. Run the corresponding Python file in the chosen model directory.
    ```bash
    python model_name.py
    ```
Replace `model_name.py` with the specific file name for the model you want to run.



### Notebooks

The `notebooks` directory contains Jupyter notebooks with textual information about evaluations and metrics for each model. These notebooks provide detailed analysis and insights into the performance of each model.

### Contributors

Allen Mwandunga


This README provides an overview of the project structure, instructions for setting up the environment, running models, and information about the notebooks. Let me know if you need further assistance!

