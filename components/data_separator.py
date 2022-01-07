import numpy
from numpy.lib import math


class DataSeparator:
    @staticmethod
    def separateDate(data):
        print('Separating dates...')
        data['Year'] = data['Dates'].dt.year
        data['Day'] = data['Dates'].dt.day
        data['Month'] = data['Dates'].dt.month
        data['DayOfWeek'] = data['Dates'].dt.weekday
        data['Hour'] = data['Dates'].dt.hour
        data['Minute'] = data['Dates'].dt.minute
        data.drop(columns=['Dates'], inplace=True)
        print('Separated dates!')

    @staticmethod
    def extractAddressInfo(data):
        print('Extracting address info...')
        data['Block'] = data['Address'].str.contains('block', case=False)
        data.drop(columns=['Address'], inplace=True)
        print('Extracted address info!')

    @staticmethod
    def dropUselessColumns(data):
        print('Dropping useless columns...')
        data.drop(columns=['Descript', 'Resolution'], inplace=True)
        print('Dropped useless columns!')
