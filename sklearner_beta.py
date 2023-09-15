# Обучение и предсказание на основе различных алгоритмов из библиотеки Scikit-Learn
# Параметры программы задаются в файле json, имя которого передается как аргумент 
# Программа работает в 2-х режимах: 
#
# 1) LEARN - обучение и оценка эффективности модели на основе выборки данных в таблице csv.
# Обучение происходит на данных, помеченных как TRAIN (1й столбец таблицы), а проверка - 
# на данных, помеченных как TEST. Если пометок TEST в таблице нет, то проверка выполняется
# на тех же данных, на которых происводилось обучение. На выходе - метрики и обученная модель.
# Так же в параметрах указывается способ нормализации данных, нужно ли нормализовать признаки
# и целевую переменную.
#
# 2) PREDICT - предсказание на основе обученной модели. Из параметров берется имя файла с обученной
# моделью, папка с данными в формате SEG-Y или NPY. Таблица с данными, на которых производилось обучение,
# открывается для того, чтобы инициализировать нормализацию (если указано, что она применялась при 
# обучении). На выходе - SEGY с результатом предсказания

import pandas as pd
import numpy as np
import sys
import json
import pickle
import os
import shutil

import segyio
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

LEARN_KEYS = {'Mode', 'LearnDataFile', 'Regressor', 'RegressorParams', 'ScaleFeatures', 'ScaleTarget', 'Scaler', 'TestResultsFile', 'ModelFile'}

PREDICT_KEYS = {'Mode', 'LearnDataFile', 'ScaleFeatures', 'ScaleTarget', 'Scaler', 'ModelFile', 'FeaturesFolder', 'DataFormat', 'PredictResultFile'}

MODES = {"Learn", "Predict"}

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

DATA_FORMATS = {'SGY', 'NPY'}          

def read_params(params_file_name):
#-------------------------------------------------------------------------------------------------------
# Чтение параметров в формате JSON
#------------------------------------------------------------------------------------------------------- 
    try:
        f = open(params_file_name)
        params = json.load(f)
        f.close()
    except:
        print(f'Ошибка чтения файла параметров! Проверье файл f{params_file_name}')    
        os._exit(os.EX_USAGE)

    # проверяем, что есть параметр "Mode"
    if not 'Mode' in params.keys() or not params['Mode'] in MODES:
        print(f'Файл параметров должен обязательно содержать ключ "Mode" одним из следующих значений: {MODES}!')
        os._exit(os.EX_DATAERR)
    
    if type(params['ScaleFeatures'])!=bool or type(params['ScaleTarget'])!=bool:
        print('Параметры ScaleFeatures и ScaleTarget могут принимать только значения "true" или "false"!')
        os._exit(os.EX_DATAERR) 

    if not (params['Scaler'] in SCALERS.keys()):
        print(f'Ключ Scaler может принимать одно из следующих значений: {SCALERS.keys()}!')
        os._exit(os.EX_DATAERR)   

    # Проверка наличия ключей, соответсвующих режиму
    if params['Mode'] == 'Learn':
        if not LEARN_KEYS.issubset(params.keys()): 
            print(f'Файл параметров должен содержать следующие ключи: {LEARN_KEYS}!')
            os._exit(os.EX_DATAERR)
    else:
        if not PREDICT_KEYS.issubset(params.keys()): 
            print(f'Файл параметров должен содержать следующие ключи: {PREDICT_KEYS}!')
           
            os._exit(os.EX_DATAERR)
    return params

def read_table_for_learning(params):
    try:
        data = pd.read_csv(params['LearnDataFile']).dropna()
    except:
        print(f"Ошибка чтения файла {params['LearnDataFile']}!")    
        os._exit(os.EX_DATAERR)
    if len(data.columns) < 3:
        print('Слишком мало столбцов!')
        os._exit(os.EX_DATAERR)
    if len(data) < 2:
        print('Не достаточно данных!')
        os._exit(os.EX_DATAERR)
    train_test_flag_column = data.columns[0]
    if not pd.unique(data[train_test_flag_column]).any() in ['TRAIN', 'train', 'Train', 'TEST', 'test', 'Test']:
        print('Первый столбец должен определять, относится ли данная строка к Train или к Test!')
        os._exit(os.EX_DATAERR)    
    target_name = data.columns[-1]
    features_names = [name for name in data.columns if (name!=target_name) & (name!=train_test_flag_column)]             
    return data, train_test_flag_column, features_names, target_name    



def read_features_for_prediction(params):
### Дальше надо подумать, возможно следует реализовать более экономящий память способ считывания данных потрассно.
### Пока считываем сразу всё
    try:
        fnames = os.listdir(params['FeaturesFolder'])
    except:
        print(f"Проблема чтения папки {params['FeaturesFolder']}")    
        os._exit(os.EX_DATAERR)
    if not fnames:
        print(f"Папка {params['FeaturesFolder']} пуста!")
        os._exit(os.EX_DATAERR)

    if params['DataFormat'] == 'SGY':
        fnames_sgy = [fname for fname in fnames if (fname.endswith('sgy') or fname.endswith('segy'))]    
        if not fnames_sgy:
            print(f"В папке {params['FeaturesFolder']} нет файлов с расширением .sgy или .segy ")
            os._exit(os.EX_IOERR)   

        print('Чтение SEG-Y файлов...')
        try:
            data = np.array([segyio.cube(os.path.join(params['FeaturesFolder'], fname)) for fname in fnames_sgy])
        except:
            print(f"Ошибка чтения SEG-Y файлов из папки {params['FeaturesFolder']}!")    
            os._exit(os.EX_IOERR)
    elif params['DataFormat'] == 'NPY':
        fnames_npy = [fname for fname in fnames if fname.endswith('npy')]    
        if not fnames_npy:
            print(f"В папке {params['FeaturesFolder']} нет файлов с расширением .npy!")
            os._exit(os.EX_IOERR)   

        print('Чтение файлов .npy ...')
        try:
            data = np.array([np.load(os.path.join(params['FeaturesFolder'], fname)) for fname in fnames_npy])
        except:
            print(f"Ошибка чтения NPY файлов из папки {params['FeaturesFolder']}!")    
            os._exit(os.EX_IOERR)    

        
    return data

def handle_learn_params(params):
    if not (params['Regressor'] in REGRESSORS.keys()):
        print(f"Ключ Regressor может принимать одно из следующих значений: {REGRESSORS.keys()}")
        os._exit(os.EX_DATAERR)
    try:
        rgr = REGRESSORS[params['Regressor']](**params['RegressorParams']) 
    except:
        print(f"Не удалось инициализировать экземпляр класса {REGRESSORS[params['Regressor']]} с параметрами {params['RegressorParams']}")      
        os._exit(os.EX_DATAERR)     
    
    data, train_test_flag_column, features_names, target_name = read_table_for_learning(params)  
    train_data = data[data[train_test_flag_column].isin(['TRAIN', 'Train', 'train'])]
    test_data = data[data[train_test_flag_column].isin(['TEST', 'Test', 'test'])]
    X_train = train_data[features_names].values
    X_test = test_data[features_names].values
    y_train = train_data[target_name].values
    y_test = test_data[target_name].values    
    scaler_x = None
    scaler_y = None
    if params['ScaleFeatures']:
        scaler_x = SCALERS[params['Scaler']]()            
        scaler_x.fit(np.vstack([X_train, X_test]))
    if params['ScaleTarget']:
        scaler_y = SCALERS[params['Scaler']]()   
        scaler_y.fit(np.hstack([y_train, y_test]).reshape(-1,1)) 
    return data, rgr, X_train, X_test, y_train, y_test, scaler_x, scaler_y

def handle_predict_params(params):
    try:
        model = pickle.load(open(params['ModelFile'], 'rb'))
    except:
        print(f"Ошибка загрузки модели {params['ModelFile']}!")    
        os._exit(os.EX_DATAERR)   

    scaler_x = None
    scaler_y = None 

    if params['DataFormat'] not in DATA_FORMATS:
        print(f'Параметр DataFormat может принимать только следующие значения: {DATA_FORMATS}')
        os._exit(1)

    if params['ScaleTarget'] or params['ScaleFeatures']:
        learn_data, train_test_flag_column, features_names, target_name = read_table_for_learning(params)
        
        if params['ScaleFeatures']:
            scaler_x = SCALERS[params['Scaler']]()    
            scaler_x.fit(learn_data[features_names].values)
        if params['ScaleTarget']:
            scaler_y = SCALERS[params['Scaler']]()
            scaler_y.fit(learn_data[target_name].values.reshape(-1,1))            

    features = read_features_for_prediction(params)   

    if params['DataFormat'] == 'SGY':
        # Скопировать первый файл, будет заготовкой для результата    
        try:
            shutil.copy(os.path.join(params['FeaturesFolder'], os.listdir(params['FeaturesFolder'])[0]), params['PredictResultFile'])
        except:
            print(f"Ошибка открытия файла {params['PredictResultFile']}. Проверьте правильности пути!")    
            os._exit(os.EX_IOERR)
        
    return features, model, scaler_x, scaler_y

#--------------------------------------------------------------------------------------------------------------
# ОСНОВНОЕ ТЕЛО ПРОГРАММЫ
#--------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":    
    print('Sklearner version 0.2') 
    if len(sys.argv) < 2:
        print('Укажите файл с параметрами!')
        os._exit(os.EX_USAGE)    

    params_file_name = sys.argv[1]
    params = read_params(params_file_name)
    if params['Mode'] == 'Learn':
        print('Режим обучения')
        data, rgr, X_train, X_test, y_train, y_test, scaler_x, scaler_y = handle_learn_params(params)       
        if scaler_x:
            X_train = scaler_x.transform(X_train)
            if len(X_test)>0:
                X_test = scaler_x.transform(X_test)
        if scaler_y:
            y_train = scaler_y.transform(y_train.reshape(-1,1)).flatten()
            if len(y_test)>0:
                y_test = scaler_y.transform(y_test.reshape(-1,1)).flatten()

        rgr.fit(X_train, y_train)
        if len(y_test)>0:
            y_pred = rgr.predict(X_test)
            if params['ScaleTarget']:
                y_test = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()
                y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).flatten()            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)   
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))             
        else:    
            print('Тестовых данных не обнаружено. Предсказание на тренировочных данных.')
            y_pred = rgr.predict(X_train) 
            if params['ScaleTarget']:
                y_train = scaler_y.inverse_transform(y_train.reshape(-1,1)).flatten()
                y_pred = scaler_y.inverse_transform(y_pred.reshape(-1,1)).flatten()      
            r2 = r2_score(y_train, y_pred)    
            mae = mean_absolute_error(y_train, y_pred)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))                
            

        print(f'Обучение успешно завершено. R2={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}')       

        # Запись истинного значения целевой величины, предсказания и метрик
        results = pd.DataFrame(columns=['Truth', 'Prediction', 'R2'])
        results['Truth'] = y_test
        results['Prediction'] = y_pred
        results['R2'] = r2
        results['RMSE'] = rmse
        results['MAE'] = mae

        results.to_csv(params['TestResultsFile'], index=None)

        try:
            pickle.dump(rgr, open(params['ModelFile'], 'wb'))
        except:
            print(f"Ошибка сохранения модели {params['ModelFile']}") 
            os._exit(os.EX_IOERR)    

    else:
        print('Режим предсказания') 
        features, model, scaler_x, scaler_y = handle_predict_params(params)
        cube_shape = features.shape[1:]
        features = features.reshape(features.shape[0], features.shape[1]*features.shape[2]*features.shape[3]).T
        if scaler_x:
            features = scaler_x.transform(features)
        prediction = model.predict(features)                
        if scaler_y:
            prediction = scaler_y.inverse_transform(prediction.reshape(-1,1)).flatten()
        prediction = prediction.reshape(cube_shape[0]*cube_shape[1], cube_shape[2]).astype('float32')    

       
        if params['DataFormat'] == 'SGY':
            with segyio.open(params['PredictResultFile'], 'r+') as f:
                for i in range(f.tracecount):
                    f.trace[i] = prediction[i]
            
        elif params['DataFormat'] == 'NPY':
            try:
                np.save(params['PredictResultFile'], prediction.reshape(cube_shape))
            except:
                print(f"Ошибка записи файла {params['PredictResultFile']}!")    
                os._exit(os.EX_IOERR)

        print(f"Предсказание завершено. Файл {params['PredictResultFile']} записан.")     

