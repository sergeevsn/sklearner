# SKLEARNER v. 0.1
Learning and prediction based on various algorithms from the Scikit-Learn library
Program parameters are set in a json file, the name of which is passed as an argument
The program works in 2 modes:

1) ```Learn``` - training and evaluation of the effectiveness of the model based on a sample of data in the csv table.
Training takes place on data marked as ```TRAIN``` (1st column of the table), and verification -
on data marked as ```TEST```. If there are no ```TEST``` marks in the table, then the check is performed
based on the same data on which the training was conducted. The output is metrics and a trained model.
The parameters also specify the method of data normalization, whether it is necessary to normalize the signs
and the target variable.

2) ```Predict``` - prediction based on a trained model. From the parameters, the name of the file with the trained
model, a folder with data in SEG-Y format. A table with the data on which the training was performed,
opens in order to initialize normalization (if it is specified that it was applied when
learning). The output is SEGY with the prediction result

## Parameter file Keys:

1) ```Mode``` - the mode of operation of the program, can be ```Learn``` or ```Predict```

2)```LearnDataFile``` is the path to the file with the training data in csv format (comma delimiter).
The following rules apply for columns: 1st column - an indication of belonging to
training or test sample ("Train" or "Test"), the last column is the target variable
that the machine learning model will predict. The remaining columns contain signs for
training.

3) ```Regressor``` is the name of the machine learning algorithm. So far, the following algorithms from Scikit-Learn are supported:
ensemble methods
- ```RF``` (RandomForestRegressor)
- ```ET``` (ExtraTreesRegressor)
- ```GB``` (GradientBoostingRegressor)
other methods:
- ```MLP``` (MLPRegressor)
- ```KNN``` (KNNRegressor)
- ```SVR``` (SVR)

4) ```RegressorParams``` - parameters passed to the MO algorithm during initialization.
The possible parameters for each algorithm can be found in the scikit-learn documentation.
Empty parameters mean default initialization.

5) ```ScaleFeatures```, true or false - whether to scale the signs

6) ```ScaleTarget```, true or false - whether to scale the target variable

7) ```Scaler``` - scaling/normalization algorithm. The following are available:
- ```Standard``` (StandardScaler)
- ```MinMax``` (MinMaxScaler)
- ```MaxAbs``` (MaxAbsScaler)
- ```Robust``` (RobustScaler)

8) ```TestResultFile``` - a file with the prediction results on the test sample. It is written as a
csv table with columns: Truth, Prediction, R2, RMSE - true value, prediction, R2
and RMSE metrics

9) ```ModelFile``` - a file with a trained model. In the Learn mode, the learning result is recorded there, and in
the Predict mode, the trained model is read from there for prediction

10) ```FeatureFolder``` - a folder with files containing signs by which a prediction will be made
in the Predict mode. There is a separate file for each attribute

11) ```DataFormat``` - the format of the feature files and the prediction result file for the Predict mode. So far, only "SEGY" is supported
