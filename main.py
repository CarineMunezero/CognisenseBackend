from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import threading
import time
import asyncio
from statistics import median
import numpy as np
from scipy.signal import welch
from pylsl import StreamInlet, resolve_streams
from collections import deque

# --------------------
# Configuration
# --------------------
SAMPLE_RATE = 500  # # samples/second
NUM_CHANNELS = 8    # e.g., Fp1, Fp2, etc.

# --------------------
# Resolve LSL Streams
# --------------------
print("Resolving available LSL streams...")
streams = resolve_streams()  # Look for all available LSL streams

inlet = None
for s in streams:
    if s.type() == "EEG":
        inlet = StreamInlet(s)
        print("Connected to EEG stream:", s.name())
        break

if inlet is None:
    print("No EEG stream of type 'EEG' found. Make sure the CGX device is streaming via LSL.")

# --------------------
# FastAPI App
# --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust this to your frontend address if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Global State
# --------------------
running = False

# We keep a separate buffer for each channel to compute Beta/Alpha ratio
channel_buffers = [deque(maxlen=5 * SAMPLE_RATE) for _ in range(NUM_CHANNELS)]
data_buffer = deque(maxlen=1000)  # Stores the "average ratio" for plotting
buffer_lock = threading.Lock()

cog_load_history = []
start_time = None
stop_time = None
current_ratio = 0.0  # Current average ratio

# --------------------
# Cognitive Load (Ratio) Calculation
# --------------------
def compute_cognitive_load(channel_data):
    """
    Compute Beta/Alpha ratio for a single channel_data array
    using Welch's method. Return float ratio = BetaPower/AlphaPower.
    """
    # f, psd from welch
    f, psd = welch(channel_data, fs=SAMPLE_RATE, nperseg=128, noverlap=64)
    alpha_idx = np.where((f >= 8) & (f <= 12))
    beta_idx  = np.where((f >= 13) & (f <= 30))
    alpha_power = np.sum(psd[alpha_idx])
    beta_power  = np.sum(psd[beta_idx])
    if alpha_power == 0:
        return 0.0
    return float(beta_power / alpha_power)

# --------------------
# Reading Thread: Pull EEG & Compute Ratio
# --------------------
def read_eeg_thread():
    global running, channel_buffers, buffer_lock, current_ratio, cog_load_history, data_buffer

    while running and inlet:
        sample, timestamp = inlet.pull_sample(timeout=0.0)
        if sample:
            # We expect at least 8 channels. 
            # Convert from microvolts to millivolts if needed.
            scaled_sample = [ch_val / 1000.0 for ch_val in sample[:NUM_CHANNELS]]

            # Add each channel's sample to the buffer
            with buffer_lock:
                for i in range(NUM_CHANNELS):
                    channel_buffers[i].append(scaled_sample[i])

                # Once we have at least 1 second of data for *each* channel, compute ratio
                if all(len(cb) >= SAMPLE_RATE for cb in channel_buffers):
                    ratio_sum = 0.0
                    for i in range(NUM_CHANNELS):
                        # Last second of data for channel i
                        ch_data = np.array(list(channel_buffers[i])[-SAMPLE_RATE:])
                        ratio_sum += compute_cognitive_load(ch_data)
                    
                    # Average ratio across all channels
                    avg_ratio = ratio_sum / NUM_CHANNELS
                    current_ratio = avg_ratio
                    data_buffer.append(avg_ratio)
                    cog_load_history.append(avg_ratio)

        time.sleep(0.001)

# --------------------
# Start/Stop Endpoints
# --------------------
@app.get("/")
def root():
    return {"message": "EEG server for Beta/Alpha ratio is running."}

@app.get("/start-eeg")
def start_eeg():
    global running, start_time, cog_load_history, data_buffer, channel_buffers
    if not running:
        running = True
        start_time = time.strftime("%H:%M:%S")
        # Clear old data
        cog_load_history = []
        with buffer_lock:
            data_buffer.clear()
            for cb in channel_buffers:
                cb.clear()

        t = threading.Thread(target=read_eeg_thread, daemon=True)
        t.start()
        return {"status": "EEG started", "start_time": start_time}
    else:
        return {"status": "EEG already running", "start_time": start_time}

@app.get("/stop-eeg")
def stop_eeg():
    global running, stop_time, cog_load_history
    running = False
    stop_time = time.strftime("%H:%M:%S")

    if cog_load_history:
        med_cog = median(cog_load_history)
    else:
        med_cog = 0.0

    # Example threshold
    threshold = 2.0
    if med_cog > threshold:
        feedback = f"High cognitive load detected (median = {med_cog:.2f})."
    else:
        feedback = f"Low/normal cognitive load detected (median = {med_cog:.2f})."

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
    Returns the last 20 ratio points plus the current ratio.
    """
    with buffer_lock:
        ratio_points = list(data_buffer)[-20:] if len(data_buffer) > 20 else list(data_buffer)
        current_val = current_ratio
    return {"ratio": ratio_points, "current_ratio": current_val}

# --------------------
# WebSocket for Real-Time Updates
# --------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with buffer_lock:
                ratio_points = list(data_buffer)[-20:] if len(data_buffer) > 20 else list(data_buffer)
                current_val = current_ratio
            payload = {
                "ratio": ratio_points,
                "current_ratio": current_val
            }
            await websocket.send_json(payload)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        print("WebSocket disconnected")

# --------------------
# Run via Uvicorn
# --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
