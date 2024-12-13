# Classification of Meal Images into Major Food Categories and differentiating between Food and non-Food Items

Welcome fellow coders, tech enthusiasts, designers or curious minds to our project. We are thrilled to have you here and excited to show you around in our classification pipeline that includes evaluation with performance metrices, Out-of-Distribution (OOD) detection methods and more!

## Table of Contents
- [Project Overview](#projectoverview)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
Dietary assessment is critical in managing health conditions such as type 2 diabetes and promoting overall well-being. Traditional methods like food diaries and nutritionist-led evaluations are often time-consuming, error-prone, and dependent on subjective input. To address these challenges, this project focuses on developing an automated and reliable food classification system using ResNet50, a state-of-the-art convolutional neural network (CNN). Trained on the Food-11 dataset, consisting of over 16,000 images spanning 11 food categories, the model demonstrates strong capabilities in accurately classifying food items while employing out-of-distribution (OOD) detection methods to reduce misclassification of non-food items.

### Main features:

- **1. Dataset and Preprocessing of Raw Data**:
  - **Dataset**: Food-11 dataset with over 16,000 images across 11 major food categories. Data Manipulation using NumPy and Pandas.
  - **Data Loading**: Framework to systematically load data in batches for training for both ID and OOD data.
  - **Data Augmentation**: Random transformations (rotations, flips, color adjustments) enhance generalization and prevent overfitting of the model.
 
- **2. Model Architecture**:
  - **Model**: ResNet50 convolutional neural networks (CNN), optimization techniques to minimize training loss over several epochs. Model Implementation using PyTorch Library.
    
- **3. Out-of-Distribution (OOD) Detection**:
  The following methods were used to allow and enhance the non-food detection in the dataset.
  - **Maximum Softmax Probability (MSP)**: Thresholds probabilistic softmax scores
  - **Maximum Logit (MaxLog)**: Analyses the maximum logit (model output) values
  - **ODIN**: Detection works with the additional tuning parameters of temperature scaling and adding perturbations

- **4. Evaluation**:
  - **Metrics**: Accuracy, Precision and Recall
  - **Visualization**: Confusion Matrix and plots using Matplotlib Library
    
## Setup

Please use the Anaconda Prompt instead of the terminal/powershell to avoid package installation problems. In the anaconda prompt, Navigate to a directory of your choice, e.g.,
```bash
cd Documents/DM
```
### Step 1: Clone the Repository
Clone the repo to this directory:
```bash
git clone https://github.com/DAck29/Food-Non-Food-Classification.git
```
Navigate to the newly created directory:
```bash
cd Food-Non-Food-Classification
```
### Step 2: Open Anaconda Prompt
To avoid any compatibility issues, we recommend to use the anaconda prompt that comes with your installation of Anaconda.

### Step 3: Create a new conda environment
Create a new conda environment using the provided environment.yml file provided in the repo to ensure that all required dependencies are installed without any hassle:
```bash
conda env create -f environment.yml
```
Alternatively, a manual environment can be created with Python 3.8:
```bash
conda create --name your_env_name python=3.8
```
### Step 4: Activate the conda environment
```bash
conda activate myenv
```
### Step 5: Install dependencies from 'requirements.txt'
Install all required dependencies using pip:
```bash
pip install -r requirements.txt
```
Note: Using the environment.yml file in Step 3 installs all dependencies. The requirements.txt file should assist, should you opt to choose to install more packages later on.

### Step 6: Verify the installation
```bash
python --version
pip list
```

You are now ready to run the python files. Continue with "Usage" to learn how to use the files.

## Usage

The basic functionality of this project is to:
- Load and preprocess datasets (Food-11 and CIFAR-10 for OOD detection)
- Train and evaluate a food classification model using ResNet50
- Implement OOD detection methods: MSP, MaxLog and ODIN
- Visualize results: Data distributions, confusion matrices, OOD performance metrics

### Step 1: Run main.py
This runs the pipeline for training and evaluation of ResNet50 with OOD detection. The outputs will be saved in the Results/ dir
```bash
python main.py
```
### Step 2: Visualization using data_distribution.py
```bash
python data_distribution.py
```
### Step 3: Training: 20 epochs
If UBELIX Slurm script is available:
```bash
sbatch slurm_jobscript.sh
```
Finally the OOD detection is conducted and results stored in Results/ dir.

## Contributing and Future Work
We highly welcome and encourage contributions to the project. Please refer to CONTRIBUTING.md for more details on how to submit pull requests or issues.

Future work may involve to include regional or international food datasets to enhance broader coverage for the model. Adapting the project to work on mobile applications could allow for better usability and an even broader population. Further, incorporating volumetric estimation of food data to improve dietary assessment.

## License 
The project is licensed under "MIT" license. See LICENSE.md file for more details.

## Contact
For any questions, collaborations or support, you can contact us at: 
- Denise Nicole Ackermann: denise.ackermann@students.unibe.ch
- Manuel Jonas Amrein: manuel.amrein@students.unibe.ch
- Noel Roy Palmgrove: noel.palmgrove@students.unibe.ch
