import os 
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pdll
