import pandas as pd
import matplotlib.pyplot as plt 
from datetime import datetime
from statsmodels.tsa.statespace.sarimax import SARIMAX

path = ''
dataf = pd.read_csv(path)
dataf.head