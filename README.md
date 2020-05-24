# IMapBook Text Classification repository

Welcome to the IMapBook text classification repository, made as an assignment
at the __Natural Language Processing__ course at the Faculty of Computer and
Information Science, University of Ljubljana. 

The project was developed by Žiga Babnik, Miha Štravs and Kristian Šenk.

### Report location

The report presenting our approach and findings is located in the main folder
under the name report.pdf

### Set up 

The repository was developed using [Conda](https://docs.conda.io/en/latest/miniconda.html)
 and Python 3.7. To set up working code perform the following steps:
  - create a new Conda Python 3.7 environment 
  - install all the required packages contained in the requirements.txt file
  by running 
    - **pip install -r requirements.txt** 
  - dowload the [word vector embeddings](https://drive.google.com/drive/folders/1nc0FovOn5pqEVjjmW_Swikc66Nel1i6j?usp=sharing)
  - place the word vector embeddings into the **./embeddings** folder


### Running the code
 
 Running the main.py script will perform classification and output the results.
 The script is controlled by two variables:
 
 - variable predicting (line 411) holds which target variable will be used for prediction
 - variable classifier (line 412) holds which classification model will be used
 
 All possible values for both variables are listed in the comments 

