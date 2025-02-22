import sys, getopt
import pdb
sys.path.append('.')
import RTIMU
import os.path
import time
import math
import spidev

import bme280

class i2c_sensors(object):
    """
    This organizes the sensor connections to the respective pis 
    """
    def __init__(self, useIMU = True, useATMO = True, useGPS = False, useANEM = False):


        #setup sensor
        self.useIMU = useIMU
        self.useATMO = useATMO
        self.useGPS = useGPS

        #preallocate dictionary of data to keep sensors tidy
        data_dict = {}

        if useIMU:
            self.imu = self.i2cinit_imu()
            data_dict['IMU'] = []
        else:
            self.imu = None

        if useATMO:
            self.atmo.bus, self.atmo.address, self.atmo.cal_params = self.i2cinit_atmo()
            data_dict['ATMO'] = []
        else:
            self.atmo = None

        #to do once we get them 
        if useGPS:
            self.gps = self.i2cinit_gps()
        else:
            self.gps = None 

        if useANEM:
            self.anem = self.i2cinit_anem()
        else: 
            self.anem = None 

        #starting empty data dictionary
        self.data_dict = data_dict
    def i2cinit_imu(self):
        """
        This function prepares the MPU sensor for i2c initiation with the Raspberry Pi
        """
        SETTINGS_FILE = "RTIMULib"

        print("Using settings file " + SETTINGS_FILE + ".ini")
        if not os.path.exists(SETTINGS_FILE + ".ini"):
          print("Settings file does not exist, will be created")

        s = RTIMU.Settings(SETTINGS_FILE)
        imu = RTIMU.RTIMU(s)

        print("IMU Name: " + imu.IMUName())

        if (not imu.IMUInit()):
            print("IMU Init Failed")
            # sys.exit(1)
            return None 
        else:
            print("IMU Init Succeeded")

        # this is a good time to set any fusion parameters
        imu.setSlerpPower(0.02)
        imu.setGyroEnable(True)
        imu.setAccelEnable(True)
        imu.setCompassEnable(True)

        poll_interval = imu.IMUGetPollInterval()
        # print("Recommended Poll Interval: %dmS\n" % poll_interval)

        return imu

    def i2cinit_atmo(self, port = 1):

        port = port
        address = 0x77
        bus = smbus2.SMBus(port)
        cal_params = bme280.load_calibration_params(bus,address)

        return bus, address, cal_params

    def i2cinit_gps(self, asdf):

        return asdf 

    def i2cinit_anem(self,asdf):
        spi_ch = 0

        # Enable SPI
        spi = spidev.SpiDev(0, spi_ch)
        spi.max_speed_hz = 1200000

        def read_adc(adc_ch, vref=3.3):

            # Make sure ADC channel is 0 or 1
            if adc_ch != 0:
                adc_ch = 1

            # Construct SPI message
            #  First bit (Start): Logic high (1)
            #  Second bit (SGL/DIFF): 1 to select single mode
            #  Third bit (ODD/SIGN): Select channel (0 or 1)
            #  Fourth bit (MSFB): 0 for LSB first
            #  Next 12 bits: 0 (don't care)
            msg = 0b11
            msg = ((msg << 1) + adc_ch) << 5
            msg = [msg, 0b00000000]
            reply = spi.xfer2(msg)

            # Construct single integer out of the reply (2 bytes)
            adc = 0
            for n in reply:
                adc = (adc << 8) + n

            # Last bit (0) is not part of ADC value, shift to remove it
            adc = adc >> 1

            # Calculate voltage form ADC value
            voltage = (vref * adc) / 1024

            return voltage

        # Report the channel 0 and channel 1 voltages to the terminal
        try:
            adc_0 = read_adc(0)
            adc_1 = read_adc(1)
            print("Ch 0:", round(adc_0, 2), "V Ch 1:", round(adc_1, 2), "V")
            time.sleep(0.2)

        # clean up GPIO
        GPIO.cleanup()
        # return voltages to main script
        return[adc_0, adc_1]


    def get_sensordata(self, event):

        #get a new empty data dictionary
        data_dict = self.data_dict

        while not event.isSet():
            data_time = time.time()
            if self.imu is not None:
                #get the IMU data
                # data_imu = self.imu.getIMUData()
                data_imu = 5
                data_dict['IMU'].append(data_imu)


            if self.atmo is not None:
                # data_atmo = bme280.sample(self.atmo.bus,self.atmo.address,self.atmo.cal_params)
                data_atmo = 5
                data_dict['ATMO'].append(data_atmo)
            # if self.gps is not None:
            #     data_gps = self.gps.getGPSData()
            time.sleep(0.1)

        return data_dict

    def test_sensors(self):

        #do some light testing with the sensors
        #return error if not good
        return 1 


class EO3222(object):
    """
        this class is just a wrapper for the pyeye interface to talk to the camera since pyeye has horrible function names
    """

    def __init__(self, bitmode = 10, bitmap_mode = True, return_thermal = False ):
        #do the initialization
        from pyueye import ueye


        # #could also use simple-pyueye https://pypi.org/project/simple-pyueye/
        # from simple_pyueye import CameraObj as Camera
        # camera=Camera(CamID)
        # camera.open()
        # camera.capture_still(save=True,filename='img.png')
        # camera.close()


        self.bitmode = bitmode
        self.bitmap_mode = bitmap_mode
        self.return_thermal = return_thermal 

  
        h_cam = ueye.HIDS(0)
        ret = ueye.is_InitCamera(h_cam, None)

        #check if it did well
        # if ret != ueye.IS_SUCCESS:
        #     pass
        # else:
        #     print("EO3222 Init Failed")

        #insert lewis code here
        ueye.is_SetDisplayMode()
        #get the code from the notebook


    def take_photo(self):
        """
        Use the software triggers for the camera to capture an image 
        """
        bitmap = 1

        #call the freeze video function
        bitmap = ueye.is_FreezeVideo() #mmediately triggers the capture of an image and then transfers the image to the PC

        #or use the image capture
        bitmap = ueye.is_CaptureVideo() # triggering of image capture and the transfer of images are performed continuously.
        return bitmap 

    def test_image():

    def shutdown_camera(self):
        ueye.is_ExitCamera()

class landlinecomms(object):

    def __init__(self ):


            """
            Sam will eventually fill this out
            """