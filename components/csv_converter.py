import pandas as pd

from algorithm_picker import Algorithm
from components.model_training import ModelTraining


class CSVConverter:
    @staticmethod
    def getFromCSVFiles():
        print("Fetching train data...")
        trainData = pd.read_csv('tables/train.csv', parse_dates=['Dates'])
        print("Fetched train data!")
        print("Fetching test data...")
        testData = pd.read_csv('tables/test.csv', parse_dates=['Dates'])
        print("Fetched test data!")
        return trainData, testData

    @staticmethod
    def TrainAndTestToCSVFiles(trainData, testData):
        print('Converting trainData to CSV...')
        trainData.to_csv('export_tables/trainModified.csv')
        print('Converted trainData to CSV!')
        print('Converting testData to CSV...')
        testData.to_csv('export_tables/testModified.csv')
        print('Converted testData to CSV!')

    @staticmethod
    def toCSVFile(data):
        print('Converting Data to CSV...')
        if ModelTraining.currentAlg == Algorithm.RandomForestClassifier:
            data.to_csv('export_tables/solutionRFC.csv', index_label='Id')
        elif ModelTraining.currentAlg == Algorithm.DecisionTreeClassifier:
            data.to_csv('export_tables/solutionDTC.csv', index_label='Id')
        elif ModelTraining.currentAlg == Algorithm.KNN:
            data.to_csv('export_tables/solutionKNN.csv', index_label='Id')
        print('Converted Data to CSV!')
