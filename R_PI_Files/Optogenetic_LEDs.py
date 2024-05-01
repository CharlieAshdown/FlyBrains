import RPi.GPIO as GPIO
import yaml
import time

from time import sleep

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

GPIO.setup(24,GPIO.OUT)


Optogenetics_led = GPIO.PWM(24,configs["Optogenetic_LEDs"]["frequency"])
Optogenetics_led.start(0)
sleep(configs["Optogenetic_LEDs"]["initial_delay"])
LED_ON = time.time()
Optogenetics_led.ChangeDutyCycle(configs["Optogenetic_LEDs"]["duty"])
sleep(configs["Optogenetic_LEDs"]["flash_length"])
LED_OFF = time.time()
Optogenetics_led.ChangeDutyCycle(0)
GPIO.cleanup()

f = open(root+"_LED_timings.txt", "w")
f.write(f"LEDs ON: {LED_ON-start} \t LEDs OFF: {LED_OFF-start}")
f.close()
