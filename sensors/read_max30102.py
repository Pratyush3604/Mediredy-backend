# sensors/max30102_reader.py
# Install dependencies: pip install max30102 smbus2

from max30102 import MAX30102
import time

def read_max30102():
    """
    Reads heart rate and SpO2 from MAX30102 sensor.
    Returns:
        heart_rate (int) - beats per minute
        spo2 (int) - oxygen saturation percentage
    """
    try:
        m = MAX30102()
        time.sleep(1)  # let sensor initialize

        hr, spo2 = m.read_sequential()  # returns tuple (heart_rate, SpO2)
        if hr is None:
            hr = 0
        if spo2 is None:
            spo2 = 0

        # Sometimes values are float, convert to int
        heart_rate = int(hr)
        spo2_percent = int(spo2)

        return heart_rate, spo2_percent

    except Exception as e:
        print(f"[ERROR] MAX30102 read failed: {e}")
        return None, None
