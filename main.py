from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
import time
from statistics import median
import random
import numpy as np
from scipy.signal import welch

# --------------------
# FastAPI Setup
# --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to your frontend domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Global State
# --------------------
running = False
data_buffer = []
cog_load = 0.0
SAMPLE_RATE = 500

cog_load_history = []
start_time = None
stop_time = None

# --------------------
# Fake EEG Generator
# --------------------
def generate_fake_sample():
    return [random.uniform(-150, 350) for _ in range(5)]  # 5 channels

def read_eeg_thread():
    global running, data_buffer, cog_load, cog_load_history
    while running:
        sample = generate_fake_sample()
        data_buffer.append(sample)

        if len(data_buffer) >= SAMPLE_RATE:
            window_data = np.array(data_buffer[-SAMPLE_RATE:])
            cog_load = compute_cognitive_load(window_data)
            cog_load_history.append(cog_load)

        time.sleep(1 / SAMPLE_RATE)

# --------------------
# Cognitive Load Calculation
# --------------------
def compute_cognitive_load(eeg_data):
    """
    Compute Beta/Alpha power ratio using Welch’s method.
    eeg_data: np.array of shape (samples, channels)
    """
    channel0 = eeg_data[:, 0]
    f, psd = welch(channel0, fs=SAMPLE_RATE, nperseg=256)
    alpha_idx = np.where((f >= 8) & (f <= 12))
    beta_idx  = np.where((f >= 13) & (f <= 30))
    alpha_power = np.sum(psd[alpha_idx])
    beta_power  = np.sum(psd[beta_idx])
    return float(beta_power / alpha_power) if alpha_power != 0 else 0.0

# --------------------
# API Endpoints
# --------------------
@app.get("/")
def root():
    return {"message": "Cognisense backend is live."}

@app.get("/start-eeg")
def start_eeg():
    global running, start_time, cog_load_history, data_buffer
    if not running:
        running = True
        start_time = time.strftime("%H:%M:%S")
        cog_load_history = []
        data_buffer = []
        threading.Thread(target=read_eeg_thread, daemon=True).start()
    return {"status": "EEG started", "start_time": start_time}

@app.get("/stop-eeg")
def stop_eeg():
    global running, stop_time, cog_load_history
    running = False
    stop_time = time.strftime("%H:%M:%S")

    med_cog = median(cog_load_history) if cog_load_history else 0.0
    feedback = (
        f"High cognitive load detected (median = {med_cog:.2f})."
        if med_cog > 1.0 else
        f"Low cognitive load detected (median = {med_cog:.2f})."
    )

    return {
        "status": "EEG stopped",
        "feedback": {
            "feedback": feedback,
            "start_time": start_time,
            "stop_time": stop_time,
            "median_cognitive_load": med_cog,
        }
    }

@app.get("/realtime")
def realtime():
    global data_buffer, cog_load
    last_samples = data_buffer[-20:] if len(data_buffer) > 20 else data_buffer
    return {
        "eeg": last_samples,
        "cognitive_load": cog_load
    }

# --------------------
# Uvicorn Entry Point (for local dev only)
# --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
