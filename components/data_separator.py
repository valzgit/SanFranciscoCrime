import numpy
from numpy.lib import math


class DataSeparator:
    @staticmethod
    def separateDate(data):
        print('Separating dates...')
        data['Year'] = data['Dates'].dt.year
        data['Day'] = data['Dates'].dt.day
        data['MonthX'] = numpy.sin(data['Dates'].dt.month * 2.0 * math.pi / 11.0)
        data['MonthY'] = numpy.cos(data['Dates'].dt.month * 2.0 * math.pi / 11.0)
        data['DayOfWeekX'] = numpy.sin(data['Dates'].dt.weekday * 2.0 * math.pi / 6.0)
        data['DayOfWeekY'] = numpy.cos(data['Dates'].dt.weekday * 2.0 * math.pi / 6.0)
        data['HourX'] = numpy.sin(data['Dates'].dt.hour * 2.0 * math.pi / 23.0)
        data['HourY'] = numpy.cos(data['Dates'].dt.hour * 2.0 * math.pi / 23.0)
        data['MinuteX'] = numpy.sin(data['Dates'].dt.minute * 2.0 * math.pi / 59.0)
        data['MinuteY'] = numpy.cos(data['Dates'].dt.minute * 2.0 * math.pi / 59.0)
        data.drop(columns=['Dates', 'DayOfWeek'], inplace=True)
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
