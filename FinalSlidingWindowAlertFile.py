import serial
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew
from collections import deque
from datetime import datetime
import joblib
import platform
import numpy as np

# ==== SETTINGS ====
PORT = "COM12"  # Change for your system
BAUD_RATE = 115200
WINDOW_SIZE = 100  # number of samples in window (~2 sec at 50Hz)
DEBOUNCE_COUNT = 3  # consecutive "Fall" predictions before alert
TEST_FILE = "Resources/Test.csv"  # reference format
MODEL_FILE = "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/Fall_detector_rf_model.pkl"
SCALER_FILE = "C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/Fall_detector_scaler.pkl"

# ==== Activity label mapping ====
LABEL_MAP = {
    "BSC": "Bending Sideways Chair",
    "CSI": "Chair Sit",
    "CSO": "Chair Sit to Stand (Opposite)",
    "FKL": "Fake Fall",
    "FOL": "Forward Fall",
    "JOG": "Jogging",
    "JUM": "Jumping",
    "SCH": "Sitting on Chair",
    "SDL": "Sideways Fall Left",
    "STD": "Standing",
    "STN": "Stumble Near",
    "STU": "Stumble",
    "WAL": "Walking"
}

# ==== Load model & scaler ====
model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)
print("Model & scaler loaded.")

# ==== Load target format ====
test_df = pd.read_csv(TEST_FILE)
target_columns = [c for c in test_df.columns if c.lower() not in ["label", "unnamed: 0"]]

# ==== Helpers ====
def low_pass_filter(data, alpha=0.8):
    filtered = np.zeros_like(data)
    filtered[0] = data[0]
    for i in range(1, len(data)):
        filtered[i] = alpha * filtered[i - 1] + (1 - alpha) * data[i]
    return filtered

def convert_to_features(raw_df):
    raw_df["acc"] = np.sqrt(raw_df["ax"]**2 + raw_df["ay"]**2 + raw_df["az"]**2)
    raw_df["gyro"] = np.sqrt(raw_df["gx"]**2 + raw_df["gy"]**2 + raw_df["gz"]**2)
    raw_df["lin"] = np.abs(raw_df["acc"] - 9.81)
    raw_df["post_gyro"] = low_pass_filter(raw_df["gyro"].values)
    raw_df["post_lin"] = low_pass_filter(raw_df["lin"].values)

    feature_map = {}
    for col in target_columns:
        if "_" in col:
            base, stat = col.rsplit("_", 1)
            feature_map.setdefault(base, set()).add(stat)

    features = {}
    for base, stats_needed in feature_map.items():
        if base in raw_df.columns:
            series = raw_df[base]
            if "max" in stats_needed:
                features[f"{base}_max"] = series.max()
            if "kurtosis" in stats_needed:
                features[f"{base}_kurtosis"] = kurtosis(series, fisher=True, bias=False)
            if "skewness" in stats_needed:
                features[f"{base}_skewness"] = skew(series, bias=False)
        else:
            for stat in stats_needed:
                features[f"{base}_{stat}"] = np.nan

    return pd.DataFrame([[features.get(c, np.nan) for c in target_columns]], columns=target_columns)

def beep_alert():
    if platform.system() == "Windows":
        import sounddevice as sd
        def play_modulated_tone(base_freq, mod_freq, duration, volume=4, sample_rate=44100):
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            modulation = (np.sin(2 * np.pi * mod_freq * t) + 1) / 2  # 0 to 1
            carrier = np.sin(2 * np.pi * base_freq * t)
            wave = volume * modulation * carrier  # volume control
            sd.play(wave, samplerate=sample_rate)
            sd.wait()

        play_modulated_tone(1016, 1000, duration=0.5)  # 16HZ for 0.5 sec
    else:
        print("\a")  # terminal beep

# ==== Serial connection ====
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
ser.flushInput()
print(f"Connected to {PORT} at {BAUD_RATE} baud.")

# Rolling buffer
buffer = deque(maxlen=WINDOW_SIZE)
fall_streak = 0

while True:
    try:
        line = ser.readline().decode(errors="ignore").strip()

        # Ignore empty or header lines
        if not line or line.startswith("ax"):
            continue

        parts = line.split(",")
        if len(parts) != 6:
            continue

        try:
            ax, ay, az, gx, gy, gz = map(float, parts)
        except ValueError:
            continue

        buffer.append({"ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz})

        if len(buffer) == WINDOW_SIZE:
            raw_df = pd.DataFrame(buffer)
            features_df = convert_to_features(raw_df)

            # === Ensure scaler feature match ===
            for col in scaler.feature_names_in_:
                if col not in features_df.columns:
                    features_df[col] = 0
            features_df = features_df[scaler.feature_names_in_]

            scaled_features = scaler.transform(features_df)
            prediction_code = model.predict(scaled_features)[0]
            prediction_full = LABEL_MAP.get(prediction_code, prediction_code)

            if prediction_code.lower() == "fall" or prediction_code in ["FOL", "SDL", "STN", "STU", "FKL"]:
                fall_streak += 1
            else:
                fall_streak = 0

            print(f"[{datetime.now()}] Prediction: {prediction_code} - {prediction_full} (streak={fall_streak})")

            if fall_streak >= DEBOUNCE_COUNT:
                print("🚨 FALL DETECTED! 🚨")
                beep_alert()
                fall_streak = 0

    except KeyboardInterrupt:
        print("Stopping...")
        break
