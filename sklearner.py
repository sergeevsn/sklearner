import sys

# import classes
from classlib import *

# Some constants
MODES = {"learn", "predict"}

GREETING_MSG = "SkLearner version 0.3. Learning on seismic data and prediction well data with scikit-learn algorythms"


# Main
if __name__ == "__main__":    
    print(GREETING_MSG) 
    
    if len(sys.argv) < 3:
        error_msg('You must specify mode (learn or predict) and parameters file!')
        sys.exit()    

    mode = sys.argv[1]
    if not mode in MODES:
        error_msg('Mode must be either learn or predict!')
        sys.exit()

    params_file_name = sys.argv[2]
    if not os.path.exists(params_file_name) or not os.path.isfile(params_file_name):
        error_msg(f'Cannot find {params_file_name}!')
        sys.exit()

    if mode == "learn":
        print('Learning mode start')
        learn = SklearnerLearn(params_file_name)            
        print(f"Feature: {learn.params['FeatureColumns']}")
        print(f"Target: {learn.params['TargetColumn']}")
        learn.execute()
        print('Learning and evaluating successfully finished. Results saved')        
    else:
        print('Prediction mode start')
        predict = SklearnerPredict(params_file_name)    
        predict.execute()
        print('Prediction successfully finished. SEG-Y file saved')




    




        
        

