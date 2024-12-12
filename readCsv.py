import pandas as pd
import numpy as np

def readSensorConnectCsv(fileName : str):
    data = pd.read_csv( fileName, skiprows= [i for i in range(0,20)] )
    return data.to_numpy()