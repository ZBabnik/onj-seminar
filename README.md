# onj-seminar Text classification

## Set up 

file requirements.txt holds all packages.

pip install -r requirements.txt

if you are having problems with torch installation use
 [conda environment.](https://docs.conda.io/en/latest/miniconda.html)
 
 Download [word vector embeddings](https://drive.google.com/drive/folders/1nc0FovOn5pqEVjjmW_Swikc66Nel1i6j?usp=sharing)
 and put them into the embeddings folder.
 
 ## How to use
 
 #### pickling
 After the first run all steps of preprocessing should have been saved
 in pickle files in pickle folder. If you want a fresh preprocessing 
 change use_pickled_data = True to False in main.py or delete pickle folder.