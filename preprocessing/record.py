import asyncio
from bleak import BleakClient
import struct
from datetime import datetime, timedelta
import threading
import sys
import signal

import matplotlib.pyplot as plt
import matplotlib.animation as animation

# BLE configuration
BLE_ADDRESS = "25:F0:98:AD:57:C7"
CHARACTERISTIC_UUID = "20a4a273-c214-4c18-b433-329f30ef7275"

# Plotting and data configuration
dataList = []
max_datapoints_to_display = 700
min_buffer_uV = 150
inamp_gain = 50
sample_rate = 256
enable_filters = True
write_to_file = True
autoscale = False

# File setup for recording
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"OpenEarableEEG_BLE_{current_time}.csv"
recording_file = open(filename, 'w') if write_to_file else None

# CSV header
if recording_file:
    recording_file.write("time,raw_data\n")

# Event to signal exit
exit_event = threading.Event()
last_valid_timestamp = None


# Function to handle incoming BLE notifications
def notification_handler(sender, data):
    global dataList
    global enable_filters
    global sample_rate
    global last_valid_timestamp

    readings = struct.unpack('<5f', data)
    timestamp = datetime.now()

    if last_valid_timestamp is None:
        last_valid_timestamp = timestamp - timedelta(seconds=5 * 1 / sample_rate)

    for i, float_value in enumerate(readings):
        # Calculate the correct timestamp for each reading
        time_diff = (timestamp - last_valid_timestamp) / 5
        timestamp_for_float_value = last_valid_timestamp + (i + 1) * time_diff

        raw_data = (float_value / inamp_gain) * 1e6  # Convert to microvolts

        dataList.append(raw_data)

        # Write data to file
        if write_to_file and recording_file:
            recording_file.write(f"{timestamp_for_float_value.strftime('%H:%M:%S.%f')},{raw_data}\n")

    last_valid_timestamp = timestamp  # Update last valid timestamp


# Async function to run the BLE client
async def run_ble_client():
    async with BleakClient(BLE_ADDRESS) as client:
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        print("Connected and receiving data...")

        # Keep the client running until exit event is set
        while not exit_event.is_set():
            await asyncio.sleep(1)


# Start the async loop in a separate thread
def start_async_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_ble_client())


# Cleanup function for handling program exit
def cleanup(*args):
    exit_event.set()
    if write_to_file and recording_file:
        recording_file.close()
    print("Cleanup complete. Exiting...")


# Handler for closing the program window
def handle_close(evt):
    cleanup()
    sys.exit(0)


# Main entry point
if __name__ == "__main__":
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    threading.Thread(target=start_async_loop, daemon=True).start()