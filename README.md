# CS-433 Machine Learning Project 1

Predicting the risk for Cardiovascular Disease using Machine Learning methods - more precisely Logistic Regression. 

## Dependencies

This project mainyl requires numpy. During development, scikit was also used to analyze the data. However, the resulting submission can be generated fully without it.

## Structure

The repository itself is structured into the `code` and `report` directory. `code` contains the actual implementation of the model whereas `report` contains descriptions or results to use in the report.

Report then contains the following files:
- `config.py`: Configuration for the model and the preprocessing to use
- `cross_validation.py`: Runs cross validation on the model, can also be configured in `config.py`
- `evaluation_metrics.py`: Metrics used for evaluation of the model
- `helpers.py`: Predefined helpers for loading the dataset and creating the submission
- `implementations.py`: Implementations on machine learning functions and algorithms
- `labels.py`: The labels of the features as well as specs describing how these values should be interpreted
- `preprocessing.py`: Functionality to preprocess the data with
- `run.py`: Trains the model and generates predictions for the test set, creating a submission in the process
- `vis.ipynb`: Testing notebook using scikit methods to analyze the data beforehand

## Running

To run either the cross validaiton or generate predictions for the test set, first place the csv's of the dataset into the `code` directory and then run the corresponding file with the python runner of your coice.