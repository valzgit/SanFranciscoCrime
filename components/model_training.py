from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from algorithm_picker import Algorithm
from catboost import CatBoostClassifier


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
        elif ModelTraining.currentAlg == Algorithm.CatBoost:
            return ModelTraining.trainingWithCatBoostAlgorithm(trainData, trainCategory, testData)

    @staticmethod
    def trainWithDecisionTreeClassifier(trainData, trainCategory, testData):  # Score: 24.82348
        print('Started training model using DecisionTreeClassifier...')
        dtc_model = DecisionTreeClassifier(criterion='entropy')
        trainCategory = np.ravel(trainCategory)
        dtc_model.fit(trainData, trainCategory)
        print('Finished with model training!')
        print('Prediction has started...')
        dataFrame = pd.DataFrame(data=dtc_model.predict(testData), columns=['Category'])
        print('Finished prediction!')
        return dataFrame

    @staticmethod
    def trainingWithRandomForestClassifier(trainData, trainCategory,
                                           testData):  # Score[40]: 4.31084 //MUCH WORSE WITH LABELENCODING
        print('Started training model using RandomForestClassifier...')
        rfc = RandomForestClassifier(n_estimators=40, verbose=1, n_jobs=4)
        trainCategory = np.ravel(trainCategory)
        rfc.fit(trainData, trainCategory)
        print('Finished with model training!')
        print('Prediction has started...')
        dataFrame = pd.DataFrame(data=rfc.predict(testData), columns=['Category'])
        print('Finished prediction!')
        return dataFrame

    @staticmethod
    def trainingWithKNNAlgorithm(trainData, trainCategory, testData):  # 26.32801
        print('Started training model using KNN...')
        knn = KNeighborsClassifier(n_neighbors=1000, n_jobs=3)
        trainCategory = np.ravel(trainCategory)
        knn.fit(trainData, trainCategory)
        print('Finished with model training!')
        print('Prediction has started...')
        dataFrame = pd.DataFrame(data=knn.predict(testData), columns=['Category'])
        print('Finished prediction!')
        return dataFrame

    @staticmethod
    def trainingWithCatBoostAlgorithm(trainData, trainCategory, testData):
        print('Started training model using CatBoost...')
        cat = CatBoostClassifier()
        trainCategory = np.ravel(trainCategory)
        cat.fit(trainData, trainCategory)
        print('Finished with model training!')
        print('Prediction has started...')
        dataFrame = pd.DataFrame(data=cat.predict(testData), columns=['Category'])
        print('Finished prediction!')
        return dataFrame
