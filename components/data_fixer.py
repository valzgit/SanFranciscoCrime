from sklearn.impute import SimpleImputer
import numpy as np


class DataFixer:
    @staticmethod
    def removeDuplicates(dataFrame):
        print('Dropping duplicates...')
        dataFrame.drop_duplicates(inplace=True)
        print('Dropped duplicates!')

    @staticmethod
    def adjustXYToSanFranciscoIfElsewhere(dataFrame):
        print('Adjusting X and Y parameters to be in San Francisco...')
        dataFrame.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)
        simpleImputer = SimpleImputer()
        for district in dataFrame['PdDistrict'].unique():
            dataFrame.loc[
                dataFrame['PdDistrict'] == district, ['X', 'Y']] = simpleImputer.fit_transform(
                dataFrame.loc[dataFrame['PdDistrict'] == district, ['X', 'Y']])
        print('Adjusted X and Y parameters to be in San Francisco!')
