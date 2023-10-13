import sys
sys.path.append("/Users/eduardo/Desktop/GUI_LSTM_FFT/Pakages/DataSetgenPacks")

from PyQt5.QtCore import *

###   Model creation
from tensorflow.keras.layers import Dense, Activation 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers.core import Activation

from datetime import datetime, date
import time

### Check if model is created
import os.path as path
import os 


##DataSet Creator

# import all classes
from Retriver_and_Processor_Dataset import DatasetGenerator



class DL_DataSet(QThread):
    Update_Progress = pyqtSignal(int)
    Update_Progress_String = pyqtSignal(str)
    Update_DataSetCreationStatus = pyqtSignal(bool)
    


    def __init__(self):
        super().__init__()
        self.Date_Time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        self.Path_DataSet = ""
        self.Seed_DataSet_id_FRGN =0
        self.DataSet_id_Just_Created=0
        self.dataSet_Gen = DatasetGenerator()
        
    
    def Set_SeedParam(self,val1):
        self.SeedDataSetList=list(val1)
        self.CurrentSeedDataRow=self.SeedDataSetList[0]
            
            
    def Set_Last_DataSet_Crated(self,val):
        self.Last_DataSet_Crated=val
        
    def Set_TypeProcessToDo(self,val):
        self.TypeProcessToDo=val
        
    def Get_NewDataSet_Data(self):
        return self.Date_Time,self.Path_DataSet,self.Seed_DataSet_id_FRGN
    
    def Get_DataSet_id_Just_Created(self):
        return self.DataSet_id_Just_Created
    
    def GetModelCreationStatus(self):
        pass
    
    def run(self):
        
        self.Update_DataSetCreationStatus.emit(False)
        self.Date_Time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        
        self.Seed_DataSet_id_FRGN =self.CurrentSeedDataRow
        self.DataSet_id_Just_Created=self.Last_DataSet_Crated+1
        
        StartDay="2001-01-02"
        EndDate="2001-01-15"
        #EndDate=date.today().strftime("%Y-%m-%d")
        ObjectiveFilePath=self.ToCreateOrUpdateDataSet(self.DataSet_id_Just_Created,self.SeedDataSetList,StartDay,EndDate,self.TypeProcessToDo)
        self.Path_DataSet = ObjectiveFilePath
        
        #Verify if objective DataSet was created
        if ObjectiveFilePath:
            self.Update_DataSetCreationStatus.emit(True)
            self.Update_Progress_String.emit("DataSet Succesfully created")
            self.Update_Progress.emit(100)
        else:
            self.Update_DataSetCreationStatus.emit(False)
            self.Update_Progress_String.emit("DataSet was not succesfully created")
            self.Update_Progress.emit(95)
        
        time.sleep(5)
        self.Update_Progress_String.emit("Ready to create another model")
        self.Update_Progress.emit(0)
    
    
    def ToCreateOrUpdateDataSet(self,DataSetId,SeedDataSetlist,dateStart,dateEnd,ProcessToDo):
        print("this is the one: "+str(SeedDataSetlist[1]))
        
        itemName=SeedDataSetlist[1]
        BackDays=SeedDataSetlist[2]
        Open_C=SeedDataSetlist[3]
        High_C=SeedDataSetlist[4]
        Low_C=SeedDataSetlist[5]
        Close_C=SeedDataSetlist[6]
        Volume_C=SeedDataSetlist[7]
        Open_FFT_C=SeedDataSetlist[8]
        High_FFT_C=SeedDataSetlist[9]
        Low_FFT_C=SeedDataSetlist[10]
        Close_FFT_C=SeedDataSetlist[11]
        Volum_FFT_C=SeedDataSetlist[12]
        Day_Wk_N_C=SeedDataSetlist[13]
        Day_MonthNDay_C=SeedDataSetlist[14]
        Year_C=SeedDataSetlist[15]
        FFT_Frec=SeedDataSetlist[16]
         
        ParentPath= "APP/DataSets/"
        BasePath="{}/Id{}".format(itemName,DataSetId)
        pathTocreated=path.join(ParentPath,BasePath)
        
        Original_Path_Retiving=pathTocreated+"/CRUDE_OIL_Data.csv"
        Onlyonecolumn=pathTocreated+"/CRUDE_OIL_Data_onlyClose.csv"
        LastOnetwice=pathTocreated+"/CRUDE_OIL_Data_LastOneTwice.csv"
        DirectionPrice=pathTocreated+"/CRUDE_OIL_Data_DirePrice.csv"
        DayNumAddedPath=pathTocreated+"/CRUDE_OIL_Dataand_DayNum.csv"
        MonthAddedPath=pathTocreated+"/CRUDE_OIL_Data_And_month.csv"
        yearAddedPath=pathTocreated+"/CRUDE_OIL_Data_And_year.csv"
        FFTAddedPath=pathTocreated+"/CRUDE_OIL_CloseFFT_2400_5Backdys.csv"
        LastPopcolum=pathTocreated+"/CRUDE_OIL_Close_lastPopcolum.csv"
        
        #Check is directory exist, otherwise To Create A Directory With Subdirectories
        
        if path.exists(pathTocreated):
            pass
        else:
            os.makedirs(pathTocreated)

        if ProcessToDo=="1":
            addToOld=False
            self.dataSet_Gen.RetivingDataPrices_Yahoo(itemName,dateStart, dateEnd,Original_Path_Retiving,Original_Path_Retiving,addToOld)
        elif ProcessToDo=="0":
            self.dataSet_Gen.UpdateToday(itemName,Original_Path_Retiving)

        
        #columns to pop up # Solution applying a mask 
        Colums_Selection=[Open_C,High_C,Low_C,Close_C,Volume_C]
        columns=['Open','High','Low','Close','Volume']
        ColumnsToPop=[]
        
        for i in range(0,len(Colums_Selection)-1):
            if Colums_Selection[i]==0:
                ColumnsToPop.append(columns[i])
        self.dataSet_Gen.PopListdf(ColumnsToPop,Original_Path_Retiving,Onlyonecolumn)

        
        #To add the day of the week 0=monday, 1=tuesday .... 4= friday
        if Day_Wk_N_C==1:
            self.dataSet_Gen.AddColumnWeekDay(Onlyonecolumn, DayNumAddedPath,False)
        else:
            DayNumAddedPath=Onlyonecolumn
            
        #To add the number*100 Month of the year: january = 100, dicember=1200
        # and Day of the month 1...30 or 1...28 or 1...31
        if Day_MonthNDay_C==1:
            self.dataSet_Gen.AddColumnMothandDay(DayNumAddedPath, MonthAddedPath,False)
        else:
            MonthAddedPath=DayNumAddedPath

        #To add the year
        if Year_C==1:
            self.dataSet_Gen.AddColumnYear(MonthAddedPath,yearAddedPath)
        else:
            yearAddedPath=MonthAddedPath
    

        """
        #Generate new FFT columns
        Colums_Selection=[Open_FFT_C,High_FFT_C,Low_FFT_C,Close_FFT_C,Volum_FFT_C]
        columns=['Open','High','Low','Close','Volume']
        ColumnsToFFT=[]
        for i in range(0,len(Colums_Selection)-1):
            if Colums_Selection[i]==0:
                ColumnsToFFT.append(columns[i])

        backdaysToconsider=6
        inicialPath=yearAddedPath
        FFTNew_FileData=FFTAddedPath
        frec=int(FFT_Frec)

        self.dataSet_Gen.getTheLastFFTValue(backdaysToconsider,frec,ColumnsToFFT,inicialPath, FFTNew_FileData)   


        columns=['High','Low']

        self.dataSet_Gen.PopListdf(columns,FFTNew_FileData,LastPopcolum)"""
        
        #Return the resulting DataSet
        return Original_Path_Retiving
        
        