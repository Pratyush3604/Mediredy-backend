# sensors/mlx90614_reader.py
# Install dependencies: pip install smbus2

import smbus2
import time

# Default I2C address of MLX90614
MLX90614_ADDR = 0x5A

def read_mlx90614():
    """
    Reads object temperature (°C) from MLX90614 sensor.
    Returns:
        temperature_c (float)
    """
    try:
        bus = smbus2.SMBus(1)  # 1 for Raspberry Pi
        temp_raw = bus.read_word_data(MLX90614_ADDR, 0x07)
        # Swap bytes
        temp_raw_swapped = ((temp_raw & 0xFF) << 8) | (temp_raw >> 8)
        # Convert to Celsius
        temperature_c = (temp_raw_swapped * 0.02) - 273.15
        bus.close()
        return round(temperature_c, 1)
    except Exception as e:
        print(f"[ERROR] MLX90614 read failed: {e}")
        return None
