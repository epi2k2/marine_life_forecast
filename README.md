# Marine Life Forecasting
## Introduction
This repository contains the final project for KSchool’s Data Science Master’s degree.

It involves the creation of an application that applies a multi-label classifying model that predicts the marine life that can be found in a given scuba diving area based on location, depth, time and  day of year.

## Requirements
There are two files that contain the necessary libraries depending on whether conda or pip are used.
- conda: environment.yml
- pip: requirements.txt

## Structure
- /: the application and notebooks are found here, numbered in running order
- /docs: holds the project report
- /malif: contains the Multi Target Encoder built for the project
- /models: in this folder you can find the final model and the knn imputer used in the application
- /tests: holds the previous versions for the notebooks and the different test runs that have been done during the development of the project

## Execution
To run the application, you only need to execute the command: streamlit run app.py.

If the process must be run from the beginning, each notebook must be executed in the given order:
- 01_analysis_and_cleaning.ipynb
- 02_data_enrichment_and_insights.ipynb
- 03_feature_engineering_and_variable_importance.ipynb
- 04_knnimputer.ipynb
- 05_model_selection.ipynb
- app.py (run with the command described above)
