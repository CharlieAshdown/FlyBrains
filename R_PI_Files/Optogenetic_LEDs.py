import RPi.GPIO as GPIO
import yaml
import time
import os
import sys

from time import sleep

start = time.time()
try:
    with open(sys.argv[1], 'r') as file:
        configs = yaml.safe_load(file)
    root = f"{os.path.split(sys.argv[1])[0]}/test_{str(sys.argv[2]).zfill(3)}"
except:
    with open('config.yaml', 'r') as file:
        configs = yaml.safe_load(file)
    root = "samples/test_" + str(configs["test_number"])

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

file_name = os.path.split(os.path.splitext(root)[0])[-1]
with open(f"{root}/{file_name}_LED_timings.txt", "w") as f:
    f.write(f"Video ON: {LED_ON - start} \tVideo OFF: {LED_OFF - start}")

