import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class DataEncoder:
    categoryNamesArray = ['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT',
                          'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT', 'EXTORTION',
                          'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING', 'KIDNAPPING',
                          'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON', 'NON-CRIMINAL',
                          'OTHER OFFENSES', 'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION', 'RECOVERED VEHICLE', 'ROBBERY',
                          'RUNAWAY', 'SECONDARY CODES', 'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE',
                          'STOLEN PROPERTY', 'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM',
                          'VEHICLE THEFT', 'WARRANTS', 'WEAPON LAWS']

    @staticmethod
    def convertToHotEncodedCategories(data):
        print("Converting encoded categories...")
        ohe = OneHotEncoder(dtype=int, sparse=False)
        i = 0
        while i < len(DataEncoder.categoryNamesArray):
            data = data.append({'Category': DataEncoder.categoryNamesArray[i]}, ignore_index=True)
            i += 1
        category = ohe.fit_transform(data.Category.to_numpy().reshape(-1, 1))
        data.drop(columns=['Category'], inplace=True)
        dataFrame = pd.DataFrame(data=category, columns=DataEncoder.categoryNamesArray)
        dataFrame.dropna(inplace=True)
        dataFrame.drop(dataFrame.tail(39).index, inplace=True)
        print("Converted encoded categories!")
        return dataFrame

    @staticmethod
    def fetchCategoriesAndClassifyThem(data):
        dataFrame = data['Category'].copy()
        data.drop(columns=['Category'], inplace=True)
        return dataFrame

    @staticmethod
    def labelEncodeDistrict(dataTrain, dadaTest):
        print("Hot encoding districts...")
        le = LabelEncoder()
        dataTrain['PdDistrict'] = le.fit_transform(dataTrain['PdDistrict'])
        dadaTest['PdDistrict'] = le.transform(dadaTest['PdDistrict'])
        print("Hot encoded districts!")
        return dataTrain, dadaTest

    @staticmethod
    def hotEncodeDistrict(data):
        print("Hot encoding districts...")
        ohe = OneHotEncoder(dtype=int, sparse=False)
        district = ohe.fit_transform(data.PdDistrict.to_numpy().reshape(-1, 1))
        data.drop(columns=['PdDistrict'], inplace=True)
        data = data.join(pd.DataFrame(data=district, columns=ohe.get_feature_names_out(['PdDistrict'])))
        data.dropna(inplace=True)
        print("Hot encoded districts!")
        return data

    le = LabelEncoder()

    @staticmethod
    def labelEncodeCategories(data):
        print("Label encoding categories...")
        encodedData = DataEncoder.le.fit_transform(data)
        print("Label encoded categories!")
        return pd.DataFrame(data=encodedData, columns=['Category'])

    @staticmethod
    def labelDecodeCategories(data):
        print("Label decoding categories...")
        data = np.ravel(data.to_numpy())
        print(data)
        inverseData = DataEncoder.le.inverse_transform(data)
        print("Label decoded categories!")
        return pd.DataFrame(data=inverseData, columns=['Category'])
