import RPi.GPIO as GPIO
import yaml
import os
from time import time as t

from time import sleep
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder


def run(configs, root):
    test_number = configs["test_number"]
    test_file = f"{root}/{test_number}"
    os.mkdir(test_file)

    picam2 = Picamera2()
    picam2.options["quality"] = configs["camera"]["setup"]["quality"]
    picam2.options["compress_level"] = configs["camera"]["setup"]["compress_level"]
    video_config = picam2.create_video_configuration()
    picam2.configure(video_config)
    GPIO.setwarnings(False)

    GPIO.setmode(GPIO.BCM)

    GPIO.setup(23,GPIO.OUT)

    IR_led = GPIO.PWM(23,configs["IR_LED"]["frequency"])
    IR_led.start(0)
    picam2.start()

    IR_led.ChangeDutyCycle(configs["IR_LED"]["duty"])

    encoder = H264Encoder(bitrate=10000000)
    output = f"{test_file}/test_{test_number}.h264"
    vid_start = t()
    picam2.start_recording(encoder, output)
    sleep(configs["camera"]["collection"]["video_delay"])
    picam2.stop_recording()
    vid_end = t()
    IR_led.ChangeDutyCycle(0)

    GPIO.cleanup()

    with open(f"{test_file}/test_{test_number}_video_timings", "w") as file:
        file.write(f"Video ON: {vid_start}\tVideo OFF: {vid_end}")


if __name__ == "__main__":
    try:
        with open('/media/charlie/FLYBRAINS5/config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        root = "/media/charlie/FLYBRAINS5/samples"
    except:
        with open('config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        root = "samples"
    run(configs, root)
