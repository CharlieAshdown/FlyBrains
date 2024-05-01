from picamera2.encoders import H264Encoder, Quality
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import yaml
from time import sleep
import time
start = time.time()
try:
    with open('/media/charlie/FLYBRAINS5/config.yaml', 'r') as file:
        configs = yaml.safe_load(file)
    root = "/media/charlie/FLYBRAINS5/samples/test_"+str(configs["test_number"])+"/"
except:
    with open('config.yaml', 'r') as file:
        configs = yaml.safe_load(file)
    root = "samples/test_"+str(configs["test_number"])
    
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(23,GPIO.OUT)
IR_led = GPIO.PWM(23,configs["IR_LED"]["frequency"])
IR_led.start(0)
IR_led.ChangeDutyCycle(configs["IR_LED"]["duty"])

picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration())
exposure = configs["camera"]["collection"]["exposure"]
gain = configs["camera"]["collection"]["gain"]
picam2.set_controls({"ExposureTime": exposure, "AnalogueGain": gain})
encoder = H264Encoder()
VIDEO_ON = time.time()
picam2.start_recording(encoder, root +'.h264', quality=Quality.VERY_HIGH)
sleep(30)
picam2.stop_recording()
VIDEO_OFF = time.time()
IR_led.ChangeDutyCycle(0)
print("Done")
f = open(root+"_video_timings.txt", "w")
f.write(f"Cideo ON: {VIDEO_ON-start} \tVideo OFF: {VIDEO_OFF-start}")
f.close()
