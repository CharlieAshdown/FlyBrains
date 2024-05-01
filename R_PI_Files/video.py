import os.path
import sys

from picamera2.encoders import H264Encoder, Quality
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import yaml
from time import sleep
import time


def record_video(configs, root):
    """
    Record the video
    :param configs: The configuration settings
    :param root: Root to save the video
    :return:
    """
    start = time.time()

    if not os.path.exists(root):
        os.mkdir(root)

    file_name = os.path.split(os.path.splitext(root)[0])[-1]

    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(23,GPIO.OUT)
    IR_led = GPIO.PWM(23,configs["IR_LED"]["frequency"])
    IR_led.start(0)
    IR_led.ChangeDutyCycle(configs["IR_LED"]["duty"])

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration())
    exposure = configs["camera"]["exposure"]
    gain = configs["camera"]["gain"]
    picam2.set_controls({"ExposureTime": exposure, "AnalogueGain": gain})
    encoder = H264Encoder()
    sleep(configs["camera"]["delay"])
    VIDEO_ON = time.time()
    picam2.start_recording(encoder, f"{root}/ {file_name}.h264", quality=Quality.VERY_HIGH)
    sleep(configs["camera"]["record_time"])
    picam2.stop_recording()
    VIDEO_OFF = time.time()
    IR_led.ChangeDutyCycle(0)
    print("Done")
    with open(f"{root}/{file_name}_video_timings.txt", "w") as f:
        f.write(f"Video ON: {VIDEO_ON-start} \tVideo OFF: {VIDEO_OFF-start}")


if __name__ == "__main__":
    try:
        with open(sys.argv[1], 'r') as file:
            configs = yaml.safe_load(file)

        root = f"{os.path.split(sys.argv[1])[0]}/test_{str(sys.argv[2]).zfill(3)}"
    except:
        with open('config.yaml', 'r') as file:
            configs = yaml.safe_load(file)
        root = "samples/test_" + str(configs["test_number"])

    record_video(configs, root)