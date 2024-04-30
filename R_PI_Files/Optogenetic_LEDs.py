import RPi.GPIO as GPIO
import yaml
from time import time as t

from time import sleep

try:
    with open('/media/charlie/FLYBRAINS5/config.yaml', 'r') as file:
        configs = yaml.safe_load(file)
    root = "/media/charlie/FLYBRAINS5/samples/"
except:
    with open('config.yaml', 'r') as file:
        configs = yaml.safe_load(file)
    root = "samples"
    
GPIO.setwarnings(False)

GPIO.setmode(GPIO.BCM)

GPIO.setup(24,GPIO.OUT)
test_number = configs["test_number"]
test_file = f"{root}/{test_number}"

Optogenetics_led = GPIO.PWM(24,configs["Optogenetic_LEDs"]["frequency"])
Optogenetics_led.start(0)
sleep(configs["Optogenetic_LEDs"]["initial_delay"])
leds_on = t()
Optogenetics_led.ChangeDutyCycle(configs["Optogenetic_LEDs"]["duty"])
sleep(configs["Optogenetic_LEDs"]["flash_length"])
Optogenetics_led.ChangeDutyCycle(0)
leds_off = t()
GPIO.cleanup()

with open(f"{test_file}/test_{test_number}_LED_timings", "w") as file:
    file.write(f"LEDs ON: {leds_on}\tLEDs OFF: {leds_off}")
