import os
import sys
import shutil
import json
import pickle
import pandas as pd
import numpy as np
import segyio
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
import warnings
warnings.filterwarnings("ignore")


LEARN_KEYS = {'LearnDataFile', 
              'FeatureColumns', 
              'WellColumn',
              'TargetColumn', 
              'TrainTestSplitMode',              
              'Regressor', 
              'RegressorParams', 
              'ScaleFeatures', 
              'ScaleTarget', 
              'Scaler', 
              'TestResultsFile', 
              'ModelFile'}

PREDICT_KEYS = {'LearnDataFile', 
                'FeatureColumns', 
                'WellColumn',
                'TargetColumn', 
                'ScaleFeatures', 
                'ScaleTarget', 
                'Scaler', 
                'ModelFile', 
                'FeaturesFolder',                  
                'PredictResultFile'}

REGRESSORS = {"RF": RandomForestRegressor,
              "GB": GradientBoostingRegressor,
              "ET": ExtraTreesRegressor,
              "MLP": MLPRegressor,
              "SVR": SVR,
              "KNN": KNeighborsRegressor
              }

SCALERS = {"Standard": StandardScaler,
           "MinMax": MinMaxScaler,
           "MaxAbs": MaxAbsScaler,
           "Robust": RobustScaler
          }               

SPLITMODES = {"BlindWells",
              "NoSplit",            
             }                 


def error_msg(msg):
    print(f'ERROR: {msg}')

class SklearnerBase:    

    def __init__(self) -> None:  
           
        self.params = None       
        self.table = None    
        self.data_file_name = ""
        self.feature_columns = []
        self.target_column = ""
        self.well_column = ""
        self.scaler_x = None
        self.scaler_y = None
        self.X_data = None
        self.y_data = None        
        self.split_mode = "NoSplit"
        self.wells = []
        self.regressor = None
        

    def read_params(self, params_file_name, mode):    
        try:
            f = open(params_file_name)
            params = json.load(f)
            f.close()
        except:
            error_msg(f'Cannot read parameters file {params_file_name}!')    
            sys.exit()
        
        if mode == 'learn':
            check_keys = LEARN_KEYS        
        else:
            check_keys = PREDICT_KEYS

        if not set(check_keys).issubset(set(params.keys())):
            error_msg(f'Check parameter keys in file {self.params_file_name}!')
            print(f'Keys in file: {set(params.keys())}')
            print(f'Must be keys: {check_keys}')
            print(f'Difference: {set(params.keys())-check_keys}')
        self.params = params

    def read_learning_table(self, data_file_name):        

        if data_file_name.endswith('.xlsx'):
            try:
                learn_df = pd.read_excel(data_file_name).dropna()
            except:
                error_msg(f"Cannot read Excel file {data_file_name}!")    
                return False
        else:    
            try:
                learn_df = pd.read_csv(data_file_name).dropna()
            except:
                error_msg(f"Cannot read CSV file {data_file_name}!")    
                return False            
            
        self.data_file_name = data_file_name    
                         
        if len(learn_df.columns) < 3:
            error_msg('At least 3 columns must be present in the table!')
            return False
        if len(learn_df.columns) < 2:
            error_msg('At least 2 rows must be present in the data!')
            return False        
        
        self.table = learn_df
        return True

class SklearnerLearn(SklearnerBase):

    def __init__(self) -> None:
        super().__init__()
       #super().read_params(mode="learn")
        self.scaled_X_data = None
        self.scaled_y_data = None        
        self.predicted_data = None
        self.metrics = None

    def scaleXYdata(self):

        if type(self.params['ScaleFeatures'])!=bool or type(self.params['ScaleTarget'])!=bool:
            error_msg('Parameters ScaleFeatures and ScaleTarget can be either "true" or "false"!')
            sys.exit() 

        if not (self.params['Scaler'] in SCALERS.keys()):
            error_msg(f'The Scaler parameter must be one of these: {SCALERS.keys()}!')
            sys.exit()    
        
        self.scaled_X_data = self.X_data.copy()
        self.scaled_y_data = self.y_data.copy()

        if self.params['ScaleFeatures']:            
            self.scaler_x = SCALERS[self.params['Scaler']]()  
            print(f"Scaling data with scaler {self.scaler_x}")
            scaled_values = self.scaler_x.fit_transform(self.X_data.iloc[:, 1:].values)
            for i, col in enumerate(self.X_data.columns[1:]):
                self.scaled_X_data[col] = scaled_values[:, i]
                
        if self.params['ScaleTarget']:
            self.scaler_y = SCALERS[self.params['Scaler']]()            
            scaled_values = self.scaler_y.fit_transform(self.y_data.iloc[:, 1].values.reshape(-1,1)).flatten()
            self.scaled_y_data[self.y_data.columns[1]] = scaled_values
        
    def unscale_target(self, target):
        if not self.scaler_y:
            return target 
        else:
            return self.scaler_y.inverse_transform(target.reshape(-1,1)).flatten()    

    def init_regressor(self):
        if not (self.params['Regressor'] in REGRESSORS.keys()):
            error_msg(f"Parameter Regressor must be one of these: {REGRESSORS.keys()}")
            sys.exit()
        try:
            self.regressor = REGRESSORS[self.params['Regressor']](**self.params['RegressorParams']) 
        except:
            error_msg(f"Cannot initialize regressor {REGRESSORS[self.params['Regressor']]} with parameters {self.params['RegressorParams']}")      
            sys.exit()

    def nosplit_learn(self):
        self.init_regressor()       
        #print(self.scaled_y_data[:,1])
        #print()
        print(f"Performing NoSplit learn with regression model {self.regressor}") 
        self.regressor.fit(self.scaled_X_data.iloc[:, 1:], self.scaled_y_data.iloc[:, 1])            
        pred = []
        for well in self.wells:
            X = self.scaled_X_data[self.scaled_X_data.iloc[:,0]==well].iloc[:,1:]                     
            pred.extend(self.regressor.predict(X))            
        self.predicted_data = self.y_data.copy()
        self.predicted_data['prediction'] = self.unscale_target(np.array(pred))

    def wellsplit_learn(self, init_rgr_once=False):
        if init_rgr_once:
            self.init_regressor()
        print(f"Performing BlindWells learn with regression model {self.regressor}")    
        pred = []    
        for well in self.wells:    
            print(f'Learning with test well {well}...')
            X_train = self.scaled_X_data[self.scaled_X_data.iloc[:, 0]!=well].iloc[:,1:]
            X_test = self.scaled_X_data[self.scaled_X_data.iloc[:, 0]==well].iloc[:,1:]             
            y_train = self.scaled_y_data[self.scaled_y_data.iloc[:, 0]!=well].iloc[:,1]
            y_test = self.scaled_y_data[self.scaled_y_data.iloc[:, 0]==well].iloc[:,1]
            if not init_rgr_once:
                self.init_regressor()
            self.regressor.fit(X_train, y_train)               
            prediction = self.regressor.predict(X_test)            
            print('R2 score is ', r2_score(y_test, prediction))
            pred.extend(prediction)
        self.predicted_data = self.y_data.copy()
        self.predicted_data['prediction'] = self.unscale_target(np.array(pred))    

    def calc_metrics(self, wells_separate=True):
        if wells_separate:
            r2 = []
            rmse = []
            mae = []
            for well in self.wells:
                temp_df = self.predicted_data[self.predicted_data.iloc[:, 0]==well].reset_index(drop=True)
                r2.extend(np.ones(len(temp_df))*r2_score(temp_df.iloc[:, 1], temp_df.iloc[:,2]))
                rmse.extend(np.ones(len(temp_df))*np.sqrt(mean_squared_error(temp_df.iloc[:, 1], temp_df.iloc[:,2])))
                mae.extend(np.ones(len(temp_df))*mean_absolute_error(temp_df.iloc[:, 1], temp_df.iloc[:,2]))
        else:
            r2 = np.ones(len(self.predicted_data))*r2_score(self.predicted_data.iloc[:, 1], self.predicted_data.iloc[:,2])   
            rmse = np.ones(len(self.predicted_data))*np.sqrt(mean_squared_error(self.predicted_data.iloc[:, 1], self.predicted_data.iloc[:,2]))
            mae = np.ones(len(self.predicted_data))*mean_absolute_error(self.predicted_data.iloc[:, 1], self.predicted_data.iloc[:,2])     

        print('Calculated mean prediction metrics:')
        print(f'R2 = {np.mean(r2)}, RMSE = {np.mean(rmse)}, MAE = {np.mean(mae)}')
    
        self.metrics = pd.DataFrame({'r2': r2, 'rsme': rmse, 'mae': mae})  


    def save_results(self):
        fname = self.params['TestResultsFile']
        if fname.endswith('.xlsx'):
            try:        
                pd.concat([self.predicted_data, self.metrics], axis=1).to_excel(fname)                
            except:    
                error_msg(f"Cannot save results to file {fname}") 
                sys.exit()
        else:   
            try: 
                pd.concat([self.predicted_data, self.metrics], axis=1).to_csv(fname, index=None)                
            except:
                error_msg(f"Cannot save results to file {fname}")    
                sys.exit()

    def save_model(self):
        try:
            pickle.dump(self.regressor, open(self.params['ModelFile'], 'wb'))
        except:
            error_msg(f"Cannot save model to file {self.params['ModelFile']}") 
            sys.exit()

    def execute(self):
        self.read_learning_table()
        print(f"Well list: {self.wells}")        
        self.scaleXYdata()
        if self.params['TrainTestSplitMode'] == "NoSplit":
            self.nosplit_learn()
        elif self.params['TrainTestSplitMode'] == "BlindWells":
            self.wellsplit_learn()
        else:
            error_msg('Unknown TrainTestSplitMode!')   
            sys.exit()       
        print('Learning complete.')    
        self.calc_metrics()    
        self.save_results()
        self.save_model()     

class SklearnerPredict(SklearnerBase):

    def __init__(self, fname) -> None:
        super().__init__(fname)            
        super().read_params(mode="predict")
        self.feature_names = {}
        self.data = None
        self.cube_shape = np.zeros(3)

    def read_model(self):
        try:
            self.regressor = pickle.load(open(self.params['ModelFile'], 'rb'))
        except:
            error_msg(f"Cannot load model from file {self.params['ModelFile']}!")    
            sys.exit()
        print(f'Model loaded: {self.regressor}')    

    def read_data(self):
        try:
            fnames = [fname for fname in os.listdir(self.params['FeaturesFolder']) if fname.endswith('.sgy') or fname.endswith('.segy')]
        except:
            error_msg(f"Cannot read from folder {self.params['FeaturesFolder']}")    
            sys.exit()
        if not fnames:
            error_msg(f"There are no SEG-Y files in folder {self.params['FeaturesFolder']}!")
            sys.exit(os.EX_DATAERR)    
        
        fnames_noext = [os.path.splitext(fname)[0] for fname in fnames]

        self.feature_names = self.regressor.feature_names_in_

        if not (set(self.feature_names).issubset(fnames) or set(self.feature_names).issubset(fnames_noext)):
            error_msg('Feature names in loaded model are inconsistent with given feature files for prediction!')
            print(f'Feature names in model: {set(self.feature_names)}')
            print(f'File names in specified folder: {set(fnames)}')
            print(set(self.feature_names).issubset(fnames))

        try:
            shutil.copy(os.path.join(self.params['FeaturesFolder'], fnames[0]), self.params['PredictResultFile'])
        except:
            error_msg(f"Cannot write file {self.params['PredictResultFile']}!")    
            sys.exit()

        self.data = []        
        for feature_name in self.feature_names:
            fname = feature_name + '.sgy' if os.path.splitext(feature_name)[1] == "" else feature_name
            try:
                self.data.append(segyio.cube(os.path.join(self.params['FeaturesFolder'], fname)))
            except:
                error_msg(f"Cannot read file {os.path.join(self.params['FeaturesFolder'], fname)}")    
                sys.exit()
            print(f"File {fname} loaded")    

        self.cube_shape = self.data[0].shape
        self.data = np.array(self.data).reshape(len(self.data), self.cube_shape[0]*self.cube_shape[1]*self.cube_shape[2]).T       

    def scale_data(self):
        if type(self.params['ScaleFeatures'])!=bool or  type(self.params['ScaleTarget'])!=bool:
            error_msg('Parameters ScaleFeatures and ScaleTarget can be either "true" or "false"!')
            sys.exit() 

        if not (self.params['Scaler'] in SCALERS.keys()):
            error_msg(f'The Scaler parameter must be one of these: {SCALERS.keys()}!')
            sys.exit()         
        
        if self.params['ScaleFeatures']:            
            self.scaler_x = SCALERS[self.params['Scaler']]()  
            print(f"Scaling data with scaler {self.scaler_x}")
            self.scaler_x.fit(self.X_data.iloc[:, 1:].values)
            self.data = self.scaler_x.transform(self.data)

        if self.params['ScaleTarget']:
            if self.params['ScaleTarget']:
                self.scaler_y = SCALERS[self.params['Scaler']]()            
                self.scaler_y.fit_transform(self.y_data.iloc[:, 1].values.reshape(-1,1)).flatten()
            
    def predict(self):
        print('Prediction...')
        prediction = self.regressor.predict(self.data)
        if self.scaler_y:
            prediction = self.scaler_y.inverse_transform(prediction.reshape(-1,1)).flatten()
        prediction = prediction.reshape(self.cube_shape[0]*self.cube_shape[1], self.cube_shape[2])          
        print(f"Saving result to file {self.params['PredictResultFile']}...")
        with segyio.open(self.params['PredictResultFile'], "r+") as f:
            for i in range(f.tracecount):
                f.trace[i] = prediction[i]    

    def execute(self):
        self.read_model()
        self.read_data()
        self.read_learning_table()
        self.scale_data()
        self.predict()  








