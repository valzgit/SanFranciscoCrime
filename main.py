from components.csv_converter import CSVConverter
from components.data_encoder import DataEncoder
from components.data_fixer import DataFixer
from components.data_separator import DataSeparator

from components.model_training import ModelTraining

trainData, testData = CSVConverter.getFromCSVFiles()

DataFixer.removeDuplicates(trainData)
DataFixer.adjustXYToSanFranciscoIfElsewhere(trainData)
DataFixer.adjustXYToSanFranciscoIfElsewhere(testData)

DataSeparator.separateDate(trainData)
DataSeparator.extractAddressInfo(trainData)
DataSeparator.dropUselessColumns(trainData)
DataSeparator.separateDate(testData)
DataSeparator.extractAddressInfo(testData)

trainCategory = DataEncoder.labelEncodeCategories(DataEncoder.fetchCategoriesAndClassifyThem(trainData))
trainData, testData = DataEncoder.labelEncodeDistrict(trainData, testData)
# trainCategory.drop(trainCategory.tail(875726 - 873412).index, inplace=True)

# testData = DataEncoder.labelEncodeDistrict(testData)

print('Dropping Id columns from Test and Result tables...')
testData.drop(columns=['Id'], inplace=True)
print('Success!')

# CSVConverter.TrainAndTestToCSVFiles(trainData, testData)

solution = ModelTraining.decideTrainingPath(trainData, trainCategory, testData)
CSVConverter.toCSVFile(solution)
