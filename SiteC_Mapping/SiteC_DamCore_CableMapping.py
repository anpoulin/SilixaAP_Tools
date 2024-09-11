import pandas as pd
import numpy as np
from DTStools import ReadXT



class MapCable (object):
    
  def __init__(self):
        pass
  
  def __enter__(self):
      return self
  
  def readCalibrationPoints(self, caliCSV):
    cable = pd.read_csv(caliCSV)
    cableX = np.array(cable['Easting'])
    cableY = np.array(cable['Northing'])
    cableZ = np.array(cable['Elevation'])
    cableMM = np.array(cable['MM'])
    cableFD = np.array(cable['FD'])
    return (cableX,cableY,cableZ,cableMM,cableFD)
  
  def readSurvey(self, surveyCSV):
      survey = pd.read_csv(surveyCSV)
      X = np.array(survey['Easting'])
      Y = np.array(survey['Northing'])
      Z = np.array(survey['Elevation'])
      return (X,Y,Z)
  
  def readDTS(self, DTSxml)
      
    # Create ReadXT object for this file.
    DTS = ReadXT(one_file)

    # Read measurement time from this XML file.
    time, _ = DTS.get_date_time()
        
        # Parse the time string into a datetime object.
    timeObject = datetime.datetime.strptime(time, '%Y-%m-%dT%H:%M:%S.%fZ')
    # Reformat the datetime object into a string format to be used in output file naming.
    timeString = timeObject.strftime('%Y%m%d_%H%M%S.%fZ')

    # Read data from XML file
    data = DTS.get_data()

    # Read individual columns from data array.
    Dist    = data[:,0]
    SF      = data[:,1]
    ASF     = data[:,2]
    SR      = data[:,3]
    ASR     = data[:,4]
    Temp    = data[:,5]
    return (Dist,Temp)
  
if __name__ == '__main__':
   calipoints = MapCable.readDTS(r'C:\Users\andrew.poulin\OneDrive - Luna Innovations, Inc\LLC - E&I Projects LLC - LLC\External\AFDE\INC24-029-AFD_SiteC_Commissioning\4_Technical\4_Data\DTS_data\XT22050\temperature\20240909_FO-DAM-A_all\channel 1\channel 1_UTC_20221025_173250.160.xml')
