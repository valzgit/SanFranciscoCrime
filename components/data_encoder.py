import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class DataEncoder:
    categoryNamesArray = []

    @staticmethod
    def getEncodedCategories(data):
        print("Fetching encoded categories...")
        ohe = OneHotEncoder(dtype=int, sparse=False)
        category = ohe.fit_transform(data.Category.to_numpy().reshape(-1, 1))
        data.drop(columns=['Category'], inplace=True)
        DataEncoder.categoryNamesArray = ohe.get_feature_names_out(['Category'])
        i = 0
        while i < len(DataEncoder.categoryNamesArray):
            DataEncoder.categoryNamesArray[i] = DataEncoder.categoryNamesArray[i].replace('Category_', '')
            i += 1
        dataFrame = pd.DataFrame(data=category, columns=DataEncoder.categoryNamesArray)
        dataFrame.dropna(inplace=True)
        print("Fetched encoded categories!")
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
        # trainCategory.drop(trainCategory.tail(875726 - 873412).index, inplace=True)
