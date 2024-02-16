
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

from datetime import datetime
import time

### Check if model is created
import os.path as path



class DL_Model(QThread):
    Update_Progress = pyqtSignal(int)
    Update_Progress_String = pyqtSignal(str)
    Update_ModelCreationStatus = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        keras.backend.clear_session()  # Reseteo sencillo
        ### Model Data ###
        self.date_Time=""
        self.Path_Model=""
        self.N_epochs_Done=0
        self.DataSet_id_FRGN=0
        self.Forcasting_Result_id_FRGN=0
        self.Colm_T_Predict=0
        self.ModelCreationStatus=False
        self.n_future=1
        
        
    def Set_SeedParam(self,val_0,val_1,val_2,val_3,val_4,val_5,val_6,val_7,val_8):#---------Layes are created
        ### Data Seed ####
        self.Seed_Data_id_FRGN=val_0
        self.LSTM1_Units=val_1
        self.LSTM2_Units=val_2
        self.LryDcoeff=val_3
        self.Lyr_Dns=val_4
        self.Lyr_Dn_Rgzr=val_5
        self.OptAdam_Co=val_6
        self.Colums=val_7
        self.BackDays=val_8
        
        
        
    def Get_NewModelData(self):
        return self.date_Time,self.Path_Model,self.N_epochs_Done,self.Seed_Data_id_FRGN,self.Colm_T_Predict
    
    def GetModelCreationStatus(self):
        return self.ModelCreationStatus
    
    def Set_Last_model_Crated(self,val):
        self.last_model_Created_N=val
        
    def Get_Last_model_Create(self):
        return self.last_model_Created_N
        
        
    def run(self):
        self.Update_Progress_String.emit("Model Creation Process Starting")
        self.Update_Progress.emit(0)
        
        self.Update_ModelCreationStatus.emit(False)
        self.today = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        last_model_Created_N_Plus1=int(self.last_model_Created_N)+1
        #self.Update_Progress.emit(self.progess)
        
        ### Model Data ###
        print("Today's date:", self.today)
        self.date_Time=str(self.today)
        self.Path_Model="APP/Models/{}.keras".format(last_model_Created_N_Plus1)
        self.N_epochs_Done=0
        
        self.Update_Progress_String.emit("Model Data Assigned")
        self.Update_Progress.emit(30)
        #SeedData is set at Set_SeedParam
                
        n_past=int(self.BackDays) #**
        modelPath=self.Path_Model
        Columns_N=int(self.Colums)#**
        LSTM1_Units=int(self.LSTM1_Units)
        LSTM2_Units=int(self.LSTM2_Units)
        LryDcoeff=float(self.LryDcoeff)
        Lyr_Dns=int(self.Lyr_Dns)
        Lyr_Dn_Rgzr=float(self.Lyr_Dn_Rgzr)
        OptAdam_Co=float(self.OptAdam_Co)
        Rrrnt_dpout=0.25
        


        inputs=keras.Input(shape=(n_past,Columns_N))

        #LSTM_Layer1=keras.layers.LSTM(n_past, input_shape=(n_past,Columns_N), return_sequences=True,activation='PReLU')(inputs)
        LSTM_Layer1=keras.layers.LSTM(LSTM1_Units, input_shape=(n_past,Columns_N), return_sequences=True,activation='PReLU', recurrent_dropout=Rrrnt_dpout)(inputs)

        #Dropout_layer2=keras.layers.Dropout(0.5)(LSTM_Layer1)# modify
        #x=Dropout_layer1=keras.layers.Dropout(0.2)(x)
        LSTM_Layer2=keras.layers.LSTM(LSTM2_Units, return_sequences=False,activation='PReLU',recurrent_dropout=Rrrnt_dpout)(LSTM_Layer1)

        Dropout_layer3=keras.layers.Dropout(LryDcoeff)(LSTM_Layer2)# modify

        #---------------------------Outputs
        #dense=keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00001, l2=0.00001))(Dropout_layer3)# L1 + L2 penalties
        dense=keras.layers.Dense(1)(Dropout_layer3)
        
        #the one 12 feb 2024
        #dense=keras.layers.Dense(Lyr_Dns,kernel_regularizer=tf.keras.regularizers.L2(Lyr_Dn_Rgzr),activation='sigmoid')(LSTM_Layer2)

        #-------Layers outputs are linked
        outputs=dense

        #-----The model it's created
        outputArray=[outputs]

        model=keras.Model(inputs=inputs, outputs=outputArray, name='Prices_Forcasting_LSTM_FFT')
        #model=keras.Model(inputs=[inputs,None], outputs=[outputs,outputs2,outputs3,outputs4,outputs5,outputs6], name='Prices_Prediction')
        #keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)


        #------------------- Loss and optimizer ----------------------------------------
        #got to ensure MeanAbsoluteError it's the good one for our data
        loss = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")

        #optim=keras.optimizers.Adam(1e-3)
        optim=keras.optimizers.Adam(OptAdam_Co)
        Metrics=["mean_squared_error"]

        losses={
            "dense": loss
        }

        #model.compile(loss=losses, optimizer=optim, metrics=Metrics)
        model.compile(loss=losses, optimizer=optim,metrics=Metrics)
        print(model.summary())

        #tf.keras.utils.plot_model(model, "FFT_added_LSTM/ModelGen/Model/Model_LSTM_31_FFT.png", show_shapes=True)


        model.save(modelPath,save_format="keras")
        
        if path.exists(modelPath):
            self.Update_ModelCreationStatus.emit(True)
            self.Update_Progress_String.emit("Model Succesfully created")
            self.Update_Progress.emit(100)
        else:
            self.Update_ModelCreationStatus.emit(False)
            self.Update_Progress_String.emit("Model was not succesfully created")
            self.Update_Progress.emit(95)
        
        time.sleep(5)
        self.Update_Progress_String.emit("Ready to create another model")
        self.Update_Progress.emit(0)