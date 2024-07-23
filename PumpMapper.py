import pandas as pd  
from matplotlib import pyplot as plt   
import datetime

pumpData = pd.read_csv(r'c:\Projects\OXY\OXY_PumpSlurry_UTC.csv')
pumpData['Time'] = pd.to_datetime(pumpData['Time'], format='%Y%m%d%H%M%S')


plt.plot(pumpData['Time'],pumpData['Slurry Rate'])
plt.show()
