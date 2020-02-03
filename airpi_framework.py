import subprocess, time
import pdb
import numpy as np
#test the air pi here
import threading
import logging
import concurrent.futures

from multiprocessing.pool import ThreadPool

import subprocess
import os
import signal
import pisensors
import m2mcomms





#things we care about
bitmap_mode = True
bitmode = 10
loop_time = 15*60 #15 minutes or 15*60 seconds per loop

temp_dir = '/home/matsuo/Documents/SENIOR/PROJECTS/pistuff/test_data/'#directory of where to temporarily store data

def sensor_process(event, sensors):
    """
        This process wraps the class for taking in the sensor data 
        event : shutdown "signal" to tell this process to stop taking in data 
        sensors : i2c sensor class from the sensors file 
    """

    # data_dict = sensors.get_sensordata(event)

    #for now try this for testing
    data_dict = {'fuck' : [213,12312,23],'fuck2':[213,123,123]}

    #store data in netcdf with time stamps 
    #returns dictionary of data 
    return data_dict

    
def camera_process(cam,image_num = 10):
    """
    This process wraps the class for taking images 
    Inputs
    cam : camera sensor class from the sensors file 
    image_num : int number of images to be taken


    Returns
    ??? i don't know

    """

    #do i wanna write immediately to netcdf or carry it around in?

    image_times = np.zeros(image_num)
    images = [] #initialize some way of storing images 

    for i in range(image_num):
        images.append(cam.take_photo())
        image_times[i] = time.time()


    #return dictionary for images too
    image_dict = {'image_times': image_times, 'images': images}


    #maybe comm this down separately from sensor data 

    #return images as string array of temporary files
    return image 


def comms_process(data):
    """
    This process handles the m2m comms passing a file name
    """




    #delete temp netcdf? or not lol storage is cheap 

    return 1 

def operational_check_process(e):
    """
    this process will set the event under unoperational conditions 
    """
    return 1

def combine_data(temp_dir, sensor_data_dict,image_dict):
    """
    This function will take in the sensor data and images, and combine them into a netcdf 
    """

    temp_file = os.path.join(temp_dir,'combined_pack.nc')
    f = netCDF4.Dataset(temp_file, 'w') #initiate netcdf write


    #set group name to first image time 
    first_time = image_dict['IMAGE_TIMES'][0]

    #put the sensor data
    data_group = f.createGroup('{}'.format(first_time))
    for k,v in sensor_data_dict.items():
        setattr(data_group, k, v)

    #read images and put in netcdf
    

    #put images in netcdf

    return combined_data




#main script 
try:
    #init stuff

    #instantiate the sensor classes 
    comms = pisensors.landlinecomms() #for now set this to now until we can get this developed 
    sensors = pisensors.i2c_sensors(useIMU = True, useATMO = True, useGPS = False, useANEM = False)
    camera = pisensors.EO3222(bitmode = 10, bitmap_mode = True, return_Thermal = False)

    #test if well connected (or incorporate tests in the init )
    #this should throw an error if connections are faulty
    

    #start the main stuff
    loop_num = 0
    operational_event = threading.Event()
    operational = True


    while operational: #continue loop while operational 

        time_loop_start = time.time() #get start time for loop


        e = threading.Event()#make an event that will end the data collection 
        if loop_num == 0:
            sensor_proc = threading.Thread(target = sensor_process, args = (e,sensors,), name = 'Sensors', daemon = True)
        image_proc = threading.Thread(target = camera_process, args = (camera,),name = 'Camera', daemon = True)
        
        #start the processes 
        if loop_num == 0: #if the first loop, start sensor process. Otherwise it should carry over from the previous loop
            sensor_proc.start()
        image_proc.start() #start image taking 

        #get the images (since this process should finish first)
        image_proc.join()
        images = image_proc.results() #get the images (these are bitmap files?)

        time_in_cycle = time.time() - time_loop_start #get current time
        time.sleep(loop_time/2.0 - time_in_cycle) #sleep until the mid cycle in the cycle

        #stop the sensor process with the event 
        e.set()
        sensor_proc.join()
        sensor_data = sensor_proc.results()

        #combine this data into a format that comms wants
        loop_data_file = combinedata(sensor_data,images)

        e = threading.Event()#make an event that will end the data collection 
        #start comms and restart sensor_proc 
        sensor_proc = threading.Thread(target = sensor_process, args = (e,sensors,), name = 'Sensors', daemon = True)
        comms_proc = threading.Thread(target = comms_process, args = (loop_data_file,),name = 'Comms', daemon = True)

        #start processes again
        sensor_proc.start() #this process should continue to the next cycle 
        comms_proc.start()

        #join the comms
        comms_proc.join()

        time_in_cycle = time.time() - time_loop_start #get current time
        time.sleep(loop_time - time_in_cycle) #sleep until the mid cycle in the cycle

        #keep track of loop numbers
        loop_num += 1 


except KeyboardInterrupt:
    #this code runs before the program exits 
    print("target reached:{}".format(counter))

    #in the case of the air pi, this will be a command from the ground pi 
    print("recieved signal to shutdown")


finally:
    #this will always run regardless of input 
    # tie up loose ends 
    

    #do the shutdown for the processes 

