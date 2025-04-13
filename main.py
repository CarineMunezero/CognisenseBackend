from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
import time
from statistics import median
import random
import numpy as np
from scipy.signal import welch

# Import pylsl for both outlet and inlet functionality.
from pylsl import StreamInfo, StreamOutlet, StreamInlet, resolve_streams

# --------------------
# Part 1: Simulated EEG Stream Outlet
# --------------------
def simulated_eeg_stream():
    """
    Simulate an EEG stream using LSL by continuously pushing random data.
    Each sample is a list of 5 floating-point numbers.
    """
    # Create a new StreamInfo object.
    info = StreamInfo(name="Simulated EEG", type="EEG", channel_count=5,
                      nominal_srate=500, channel_format='float32', source_id="sim_eeg_01")
    outlet = StreamOutlet(info)
   
    print("Simulated EEG stream outlet created and streaming data...")
    while True:
        # Generate 5 random values. Adjust the range if needed.
        sample = [random.uniform(-150, 350) for _ in range(5)]
        outlet.push_sample(sample)
        # Wait to simulate a 500 Hz sampling rate.
        time.sleep(1/500.0)

# Start the simulated EEG stream in a background thread.
sim_thread = threading.Thread(target=simulated_eeg_stream, daemon=True)
sim_thread.start()

# Allow a moment for the simulated stream to start.
time.sleep(1)

# --------------------
# Part 2: LSL Stream Inlet and FastAPI Server
# --------------------
print("Resolving all LSL streams...")
streams = resolve_streams()  # Look for all available LSL streams

# Loop through available streams and pick the one with type "EEG"
inlet = None
for s in streams:
    if s.type() == "EEG":
        inlet = StreamInlet(s)
        print("Connected to EEG stream:", s.name())
        break

if inlet is None:
    print("No EEG stream of type 'EEG' found. Make sure the simulated EEG stream is running.")

# --------------------
# FastAPI Setup
# --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to restrict origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Global State for Data and Computation
# --------------------
running = False            # Whether we are currently reading data from LSL.
data_buffer = []           # Storage for recent EEG samples.
cog_load = 0.0             # Last computed cognitive load.
SAMPLE_RATE = 500          # Approximate sampling rate (Hz).

cog_load_history = []      # Stores computed cognitive load values over the session.
start_time = None          # When EEG reading started.
stop_time = None           # When EEG reading stopped.

# --------------------
# Reading Thread: Pulls data from the LSL inlet.
# --------------------
def read_eeg_thread():
    global running, data_buffer, cog_load, cog_load_history, inlet
    while running and inlet:
        # Non-blocking sample pull (with a very short timeout)
        sample, timestamp = inlet.pull_sample(timeout=0.0)
        if sample:
            data_buffer.append(sample)
            # Once we have at least one second of samples, compute cognitive load.
            if len(data_buffer) >= SAMPLE_RATE:
                window_data = np.array(data_buffer[-SAMPLE_RATE:])
                cog_load = compute_cognitive_load(window_data)
                cog_load_history.append(cog_load)
        time.sleep(0.001)  # Small sleep to prevent busy-waiting.

# --------------------
# Cognitive Load Calculation Function
# --------------------
def compute_cognitive_load(eeg_data):
    """
    Compute the ratio of Beta band power (13-30 Hz) to Alpha band power (8-12 Hz)
    using channel 0.
    eeg_data is expected to have shape (samples, channels).
    """
    channel0 = eeg_data[:, 0]  # Use the first channel for demonstration.
    f, psd = welch(channel0, fs=SAMPLE_RATE, nperseg=256)
    # Identify indices for Alpha and Beta frequency bands.
    alpha_idx = np.where((f >= 8) & (f <= 12))
    beta_idx  = np.where((f >= 13) & (f <= 30))
    alpha_power = np.sum(psd[alpha_idx])
    beta_power  = np.sum(psd[beta_idx])
    if alpha_power == 0:
        return 0.0
    return float(beta_power / alpha_power)

# --------------------
# FastAPI Endpoints
# --------------------
@app.get("/")
def root():
    return {"message": "EEG server is running."}

@app.get("/start-eeg")
def start_eeg():
    """
    Start reading EEG data from the LSL stream.
    Records the session start time and resets the data buffer.
    """
    global running, start_time, cog_load_history, data_buffer
    if not running:
        running = True
        start_time = time.strftime("%H:%M:%S")
        # Reset session data.
        cog_load_history = []
        data_buffer = []
        t = threading.Thread(target=read_eeg_thread, daemon=True)
        t.start()
    return {"status": "EEG started", "start_time": start_time}

@app.get("/stop-eeg")
def stop_eeg():
    """
    Stop reading EEG data, record the stop time, and provide session feedback
    based on the median cognitive load.
    """
    global running, stop_time, cog_load_history
    running = False
    stop_time = time.strftime("%H:%M:%S")
   
    if cog_load_history:
        med_cog = median(cog_load_history)
    else:
        med_cog = 0.0

    # Define a threshold for cognitive load feedback.
    threshold = 1.0
    if med_cog > threshold:
        feedback = f"High cognitive load detected (median = {med_cog:.2f})."
    else:
        feedback = f"Low cognitive load detected (median = {med_cog:.2f})."
   
    feedback_message = {
        "feedback": feedback,
        "start_time": start_time,
        "stop_time": stop_time,
        "median_cognitive_load": med_cog
    }
   
    return {"status": "EEG stopped", "feedback": feedback_message}

@app.get("/realtime")
def realtime():
    """
    Return the most recent EEG data (last 20 samples) and the current cognitive load.
    """
    global data_buffer, cog_load
    last_samples = data_buffer[-20:] if len(data_buffer) > 20 else data_buffer
    return {
        "eeg": last_samples,
        "cognitive_load": cog_load
    }

# --------------------
# Run the FastAPI server via uvicorn if executed directly.
# --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


