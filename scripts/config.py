\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
   
import os

                   
                                                    
                                                               
                                                                
DATA_ROOT = os.environ.get(
                      ,
                                                               ,
)
SCRIPT_DIR = os.environ.get(
                    ,
                                                                         ,
)
RESULTS_DIR = os.environ.get(
                     ,
                                                             ,
)
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

                           
FILTER_LOW = 0.1                           
FILTER_HIGH = 30.0                        
FILTER_DESIGN = 'firwin'

BASELINE = (-0.2, 0.0)               
REJECT_THRESHOLD = 100e-6                                          

     
DO_ICA = True
ICA_N_COMPONENTS = 0.99
ICA_HIGHPASS_FOR_FIT = 1.0                                            
ICA_METHOD = 'fastica'
ICA_RANDOM_STATE = 42

                      
TMIN_EPOCH = -0.2
TMAX_EPOCH = 0.8

                          
EARLY_WINDOW = (0.0, 0.150)                          
P300_WINDOW = (0.300, 0.600)                   

                                            
EARLY_WINDOW_200 = (0.0, 0.200)
EARLY_WINDOW_250 = (0.0, 0.250)

                                        
BASELINE_CONTROL_WINDOW = (-0.150, 0.0)

                      
EARLY_CHANNEL = 'Fz'
P300_CHANNEL = 'Pz'
EARLY_FALLBACK = ['FCz', 'F1', 'F2', 'Cz']
P300_FALLBACK = ['P1', 'P2', 'CPz', 'Cz']

                                                     
                                                                               
                                                                           
                                                                              
                                                                              
                                                                              
                                                                                
                                                                                
TARGET_CODES = [11, 22, 33, 44, 55]
STANDARD_CODES = [12, 13, 14, 15,
                  21, 23, 24, 25,
                  31, 32, 34, 35,
                  41, 42, 43, 45,
                  51, 52, 53, 54]

                    
MIN_TRIALS_REQUIRED = 10

                   
ALPHA = 0.05
RANDOM_SEED = 42

                                    
                                                                      
                                                                 
                        
PSEUDOTRIAL_MATCH_REAL_N = True

                                                                     
                                                                
                                                   
PSEUDOTRIAL_MIN_GAP_FROM_REAL = 1.0           

                                                
PSEUDOTRIAL_SEED = 12345

                              
                                                            
FILES = {
                    :          'trial_features_canonical.csv',
                             : 'trial_features_per_electrode.csv',
                 :             'lmm_summary_canonical.csv',
                           :   'per_electrode_canonical.csv',
                         :     'pseudotrial_lmm_summary.csv',
                 :             'model_diagnostics.csv',
                :              os.path.join('logs', 'subject_exclusions.csv'),
                       :       os.path.join('logs', 'preprocessing_log.csv'),
                  :            'figure1_grand_average.npz',
                                                                   
                        :      os.path.join('logs', 'interelectrode_all.csv'),
                         :     os.path.join('logs', 'interelectrode_val2_same_channel.csv'),
                          :    os.path.join('logs', 'interelectrode_val1_cross_to_Pz.csv'),
                          :    os.path.join('logs', 'interelectrode_val3_shape_to_Pz.csv'),
                           :   os.path.join('logs', 'heterogeneity_primary_trials.csv'),
                            :  os.path.join('logs', 'heterogeneity_ds006018_checkpoint.csv'),
}

def get_channel_index(ch_names, primary, fallbacks):
                                                                                  
    if primary in ch_names:
        return primary, ch_names.index(primary)
    for fb in fallbacks:
        if fb in ch_names:
            return fb, ch_names.index(fb)
    return None, None

def banner(msg):
                                                            
    print("=" * 70)
    print(msg)
    print("=" * 70)
