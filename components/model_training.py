from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from algorithm_picker import Algorithm
from components.data_encoder import DataEncoder


class ModelTraining:
    currentAlg = Algorithm.KNN

    @staticmethod
    def decideTrainingPath(trainData, trainCategory, testData):
        if ModelTraining.currentAlg == Algorithm.DecisionTreeClassifier:
            return ModelTraining.trainWithDecisionTreeClassifier(trainData, trainCategory, testData)
        elif ModelTraining.currentAlg == Algorithm.RandomForestClassifier:
            return ModelTraining.trainingWithRandomForestClassifier(trainData, trainCategory, testData)
        elif ModelTraining.currentAlg == Algorithm.KNN:
            return ModelTraining.trainingWithKNNAlgorithm(trainData, trainCategory, testData)

    @staticmethod
    def trainWithDecisionTreeClassifier(trainData, trainCategory, testData):  # Score: 26.94747
        print('Started training model using DecisionTreeClassifier...')
        dtc_model = DecisionTreeClassifier(criterion='entropy')
        dtc_model.fit(trainData, trainCategory)
        print('Finished with model training!')
        print('Prediction has started...')
        dataFrame = pd.DataFrame(data=dtc_model.predict(testData), columns=DataEncoder.categoryNamesArray)
        print('Finished prediction!')
        return dataFrame

    @staticmethod
    def trainingWithRandomForestClassifier(trainData, trainCategory, testData):  # Score[40]: 4.31084
        print('Started training model using RandomForestClassifier...')
        rfc = RandomForestClassifier(n_estimators=40)
        rfc.fit(trainData, trainCategory)
        print('Finished with model training!')
        print('Prediction has started...')
        dataFrame = pd.DataFrame(data=rfc.predict(testData), columns=DataEncoder.categoryNamesArray)
        print('Finished prediction!')
        return dataFrame

    @staticmethod
    def trainingWithKNNAlgorithm(trainData, trainCategory, testData):
        print('Started training model using KNN...')
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(trainData, trainCategory)
        print('Finished with model training!')
        print('Prediction has started...')
        dataFrame = pd.DataFrame(data=knn.predict(testData), columns=DataEncoder.categoryNamesArray)
        print('Finished prediction!')
        return dataFrame

