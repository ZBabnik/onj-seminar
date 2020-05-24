# onj-seminar Text classification

## Report location

Report preview for the first submission is in the main folder 
with the name report_preview.pdf

## Set up 

file requirements.txt holds all packages.

pip install -r requirements.txt

if you are having problems with torch installation use
 [conda environment.](https://docs.conda.io/en/latest/miniconda.html)
 
 Download [word vector embeddings](https://drive.google.com/drive/folders/1nc0FovOn5pqEVjjmW_Swikc66Nel1i6j?usp=sharing)
 and put them into the embeddings folder.
 
 ### Testing
 
 To run predictor testing there are two parameters in a form of a string that need to be set up.
On line 417 predicting holds the string which holds the data that the model will predict and on line 418 classifier
gives an option between logistic regression and neural network.
