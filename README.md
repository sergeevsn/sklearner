# SKLEARNER v. 0.3

```git clone https://github.com/sergeevsn/sklearner.git```
```pip install -r requirements.txt```
```python sklearner.py <mode> <parameters>```

There is a test data along with parameters file in folder ```test_data```, you can try like this:
```python sklearner.py learn test_data/params.json```   and then 
```python sklearner.py predict test_data/params.json```.
Results will be in ```test_data``` folder too.

Learning and prediction based on various algorithms from the Scikit-Learn library
Program parameters are set in a json file, the name of which is passed as an argument
The program works in 2 modes:

1) Mode ```learn``` - training and evaluation of the effectiveness of the model based on a sample of data in the csv/excel table.


2) Mode ```predict``` - prediction based on a trained model. From the parameters, the name of the file with the trained
model, a folder with data in SEG-Y format. A table with the data on which the training was performed,
opens in order to initialize normalization (if it is specified that it was applied when
learning). The output is SEGY with the prediction result

## Parameter file Keys:

1)```LearnDataFile``` is the path to the file with the training data in comma delimiter text table file
(.csv) or Excel(.xlsx) table.
The following rules apply for columns: 1st column - an indication of belonging to
training or test sample ("Train" or "Test"), the last column is the target variable
that the machine learning model will predict. The remaining columns contain signs for
training.

2)```FeatureColumns``` stands for table column names where attributes that would be used as features in 
machine learning

3)```WellColumn``` which column is for well identifier

4)```TargetColumn``` which columnn is for target attribute in machine learning

5)```TrainTestSplitMode``` can be ```BlindWells```, then there will be separate slpit for every well, where 
one goes to test data and other to train data, and ```NoSplit``` to learn and evaluate on all data

6) ```Regressor``` is the name of the machine learning algorithm. So far, the following algorithms from Scikit-Learn are supported:
ensemble methods
- ```RF``` (RandomForestRegressor)
- ```ET``` (ExtraTreesRegressor)
- ```GB``` (GradientBoostingRegressor)
other methods:
- ```MLP``` (MLPRegressor)
- ```KNN``` (KNNRegressor)
- ```SVR``` (SVR)

7) ```RegressorParams``` - parameters passed to the MO algorithm during initialization.
The possible parameters for each algorithm can be found in the scikit-learn documentation.
Empty parameters mean default initialization.

8) ```ScaleFeatures```, true or false - whether to scale the signs

9) ```ScaleTarget```, true or false - whether to scale the target variable

10) ```Scaler``` - scaling/normalization algorithm. The following are available:
- ```Standard``` (StandardScaler)
- ```MinMax``` (MinMaxScaler)
- ```MaxAbs``` (MaxAbsScaler)
- ```Robust``` (RobustScaler)

11) ```TestResultFile``` - a file with the prediction results on the test sample. It is written as a
csv or excel table with columns: Truth, Prediction, R2, RMSE - true value, prediction, R2
and RMSE metrics

12) ```ModelFile``` - a file with a trained model. In the Learn mode, the learning result is recorded there, and in
the Predict mode, the trained model is read from there for prediction

11) ```FeatureFolder``` - a folder with files containing signs by which a prediction will be made
in the Predict mode. There is a separate file for each attribute

12) ```DataFormat``` - the format of the feature files and the prediction result file for the Predict mode. So far, only "SEGY" is supported
