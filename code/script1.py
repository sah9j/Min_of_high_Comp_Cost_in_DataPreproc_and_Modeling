#!/usr/bin/env python
from mpi4py import MPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import process_time    #checking cpu time
from sklearn.preprocessing import MinMaxScaler 
import time

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size
name = MPI.Get_processor_name()
if rank == 0:
    #start = process_time()
    def data_process(data, scaler, cs ="yes"):
        #reverse index
        tdata=data.reindex(index=data.index[::-1])
        #get the infected 
        I=tdata['TOTAL_CONFIRMED']
        #get the recovered
        R =tdata['TOTAL_INACTIVE_RECOVERED']
        #get the length of the data
        nn =len(I)
        # show whether want to scaling
        if cs =="yes": ##indicate yes
            tt=np.linspace(0,nn, nn)
            y1 =np.array(I[:nn]).reshape((-1,1))
            y2 =np.array(R[:nn]).reshape((-1,1))
            #scaling
            II =scaler.fit_transform(y1)
            RR =scaler.fit_transform(y2)
            #plot
            plt.plot(tt, II, '--r')
            plt.plot(tt, RR, '--b')
            plt.legend(['Infected', 'Recovered'])
            plt.title('Scaled Data') 
            plt.xlabel('Time (Days)')
            plt.ylabel('Data (# of People)')
            plt.savefig('scaled_mpi.png')
            plt.show()
        else:  ##indicate no
            tt=np.linspace(0,nn, nn)
            II =np.array(I[:nn]).reshape((-1,1))
            RR =np.array(R[:nn]).reshape((-1,1))
            #plot
            plt.plot(tt, II, '--r')
            plt.plot(tt, RR, '--b')
            plt.legend(['Infected', 'Recovered'])
            plt.title('Unscaled Data') 
            plt.xlabel('Time (Days)')
            plt.ylabel('Data (# of People)')
            plt.savefig('scaled_mpi_2.png')
            plt.show()    
        return tt, II, RR  #scaled
    print("rank0 finished")
    data =pd.read_csv("tndata.csv")
    scaler =MinMaxScaler()
    tt, II, RR=data_process(data, scaler, "no")  #scaled
    
    
#get the start time

