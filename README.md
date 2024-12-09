# Image Gallery and Deep Learning for image editing and analysis

Welcome fellow coders, tech enthusiasts, designers or curious minds to our project. We are thrilled to have you here and excited to show you around in our image gallery editor that does basic editing functions and uses a CNN model for further analysis.

## Table of Contents
- [Project Overview](#projectoverview)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project combines a image gallery and editor with an Convolutional Neuronal Network (CNN) for classification. Image classification is about differentiating image types and  objects in images. Image classification in the medical context can support clinical decision making whether a patient is healthy or not.

The aim of the project is to pre-process images and move them to the desired location. Once the dataset is in the right location the CNN shall be trained, validated, and evaluated. The performance of the model shall be evaluated and visualized by plotting various statistics and metrics.

### Main features:

- **Image Gallery and Editor**:
  - **Upload and Storage**: Multiple image formats including JPEG, PNG.
  - **Organization**: Categorize images by date, event or subject.
  - **Basic Editing**: Cropping, resizing and adjusting brightness/contrast in terminal

  - **Data Preparation**: Preprocess image data effectively for classification tasks.
  - **Model Training**: Train models on the datasets to attain best accuracy.
  - **Performance Evaluation**: Validate the trained model on unseen data and analyze segmentation.

- **Visualization and Performance Metrics**:
  - **Classification**: Visualize output of classification and accuracy.
  - **Performance Metrics**: Learning curves
  - **Metadata Analysis**: Visual output of metadata used.

We will especially focus on the last point, as well as properly preparing the used data so we can also apply the methods learned during the lecture. 

## Setup

Please use the Anaconda Prompt instead of the terminal/powershell to avoid package installation problems. In the anaconda prompt, Navigate to a directory of your choice, e.g.,
```bash
cd Documents/DataScience
```
### Step 1: Clone the Repository
First, navigate to a directory of your choice, e.g.,
```bash
cd Documents/DataScience
```
Then, clone the repo to this directory:
```bash
git clone https://github.com/PythonDataScience24/Image-Gallery-and-Deep-Learning-for-image-editing-and-analysis.git
```
Navigate to the newly created directory:
```bash
cd Image-Gallery-and-Deep-Learning-for-image-editing-and-analysis/Homework10
```
### Step 2: Open Anaconda Prompt
To avoid any compatibility issues, we recommend to use the anaconda prompt that comes with your installation of Anaconda.

### Step 3: Create a new conda environment
Create a new conda environment with python 3.8. "myenv" is the name of your environment which you are free to adjust.
```bash
conda create --name myenv python=3.8
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
Navigate: 
```bash
cd Homework10
```
Note: We have included a conda command to install. It is commented out but if you wish to do so, you can use that command and not do it by pip. Here you are expected to be able to use conda installation. We recommend however to do so as instructed above using pip for easier installation and access.

You are now ready to run the python files. Continue with "Usage" to learn how to use the files.

## Usage

The basic functionality of this project is to:
- Upload and manage images
- Perform basic editing
- Run models
- Interpret outputs

Important: The zip files (building_images.zip, x_ray_images.zip) are extracted by running our code and further creates folders (building_images, Forests, resized_Img, xray_dataset). These created folders can be deleted after running our app as these will be created next time you run the app. Do not delete anything else!

**Start here:** While being in the anaconda prompt from before, type: 
```bash
python main.py
```
From here on you will instructed via user interface to navigate. Start with Image Gallery to create all necessary files for running CNN later.

Img_Gallery_Examples.py - this will unpack the images and give a small demonstration of the ImgEditorGallery class. Ensure you are in the file directory when running the file. This file relies on Img_Gallery.py. After having run this, the dataset should be ready and you can move on to the next step.

Run CNN.py - this file relies on CNN_utils.py. For memory reasons we set n_epochs to only 2. Feel free to adjust. If you want use UBELIX GPU there is a SLURM script ready to use.


### Expected Output

Datasets are prepared in dedicated folders.

Image editings like size, name, and location.

Visualization of various example images results from the trained CNN network.


## Contributing 
We highly welcome and encourage contributions to the project. Please refer to CONTRIBUTING.md for more details on how to submit pull requests or issues.

## License 
The project is licensed under "MIT" license. See LICENSE.md file for more details.

## Contact
For any questions, collaborations or support, you can contact us at: 
- Kirchhofer Fabricio: fabricio.kirchhofer@students.unibe.ch
- Nikolic Filip: filip.nikolic@students.unibe.ch
- Osman Ibrahim: osman.ibrahim@students.unibe.ch
- Palmgrove Noel Roy: noel.palmgrove@students.unibe.ch
