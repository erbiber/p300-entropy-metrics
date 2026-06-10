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
   

                                                                            
             
                                                                            
DATASET_ID = "ds006018"
CACHE_DIR = "./data_ds006018"
TASK_FILTER = "visualoddball"                                                  

                                                                            
                                                                           
                                                                            
SFREQ_EXPECTED = 500.0                                              
TMIN_EPOCH = -0.2
TMAX_EPOCH = 0.8
BASELINE = (-0.2, 0.0)
EARLY_WINDOW = (0.0, 0.150)                       
EARLY_WINDOW_200 = (0.0, 0.200)
EARLY_WINDOW_250 = (0.0, 0.250)
P300_WINDOW = (0.300, 0.600)                
BASELINE_CONTROL_WINDOW = (-0.150, 0.0)

                                       
                                                                 
                           
                           

                                                                            
                                                                   
                                                                            
EARLY_CHANNEL = 'Fz'
P300_CHANNEL = 'Pz'
                                                                 
EARLY_FALLBACK = ['FC1', 'FC2', 'F3', 'F4']                                    
P300_FALLBACK = ['CP1', 'CP2', 'P3', 'P4']                                      

                                                                                  
EOG_CHANNELS = ['HEL', 'HER', 'VER']                                     
MASTOID_CHANNELS = ['LM', 'RM']                                                
                                           
                                                                                         

                                                                            
                                                                           
                                                                            
                                           
                                                                      
                                                                            
                                          
 
                                                                            
                                                                        
                                                                             
                                                                         
                                                     
 
                                                             
                                                    
                                                                          
                               

                                                                              
TARGET_CODES = [10001, 10007, 10013, 10019, 10025]                        
STANDARD_CODES = []                                              
TRIAL_CODES = TARGET_CODES

                                                                              
                                                    
                                                                           
                                                                           
                                               
                                             

RESPONSE_CODES = {'correct': 10026, 'error': 10027}                
BOUNDARY_CODE = 10028

                                                                            
                                                  
                                                                            
FILTER_LOW = 0.1
FILTER_HIGH = 30.0
FILTER_DESIGN = 'firwin'

DO_ICA = True
ICA_N_COMPONENTS = 0.99
ICA_HIGHPASS_FOR_FIT = 1.0
ICA_METHOD = 'fastica'
ICA_RANDOM_STATE = 42

REJECT_THRESH_DEFAULT = 100e-6
REJECT_THRESH_RELAXED = 150e-6

                                                                            
                                       
                                                                            
MIN_TRIALS_REQUIRED = 10
PSEUDOTRIAL_SEED = 12345

                                     
PSEUDOTRIAL_CONFIGS = [
    ('config1', 1.0, 100e-6),
    ('config2', 0.5, 100e-6),
    ('config3', 1.0, 150e-6),
    ('config4', 0.5, 150e-6),
]

                                                           
PERM_ENTROPY_ORDER = 3
PERM_ENTROPY_DELAY = 1

                                                                            
        
                                                                            
RESULTS_DIR_DS = "./results_ds006018"

                                                                            
                                    
                                                                            
def get_channel_index(ch_names, primary, fallbacks):
                                                                                 
    lower = [c.lower() for c in ch_names]
    if primary.lower() in lower:
        idx = lower.index(primary.lower())
        return ch_names[idx], idx
    for fb in fallbacks:
        if fb.lower() in lower:
            idx = lower.index(fb.lower())
            return ch_names[idx], idx
    return None, None

def banner(msg):
    line = "=" * 70
    print(f"\n{line}\n{msg}\n{line}")
