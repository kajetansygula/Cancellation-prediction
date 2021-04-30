## Cancellations of hotel reservations

This repository contains solution of classification problem of hotel reservations cancellations.
The main goal of the project is to find out the most accurate model (with at least 85% average 
cross validation accuracy score) with a high simplicity.

Dataset used to train a model is [Hotel booking demand.](https://www.kaggle.com/jessemostipak/hotel-booking-demand)
For training a model, there is used data only of City Hotel to make it possible to check, 
how the model will act for the data from different type of the property. 

Project workflow:
1. Generate data (src/data/generate_data.py)
2. Explore dataset (notebooks/Exploratory data analysis.ipynb)
3. Inspect features (notebooks/Feature engineering.ipynb)
4. Compare different classifiers (notebooks/Model selection.ipynb)

All functions used across the notebooks are defined in src package. 

Structure of the project was inspired by [CookieCutter.](https://drivendata.github.io/cookiecutter-data-science/)

 

