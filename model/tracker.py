import json
import os
from datetime import datetime

log_file = "dataset/plant_history.json"

def save_observation(plant_id, predicted_class, confidence):
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            history = json.load(f)
    else:
        history = {}

    if plant_id not in history:
        history[plant_id] = []

    observation = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "prediction": predicted_class,
        "confidence": round(confidence, 2)
    }

    history[plant_id].append(observation)

    with open(log_file, "w") as f:
        json.dump(history, f, indent=4)

    print("Observation saved for", plant_id)

def analyze_trend(plant_id):
    with open(log_file, "r") as f:
        history = json.load(f)

    if plant_id not in history or len(history[plant_id]) < 2:
        return "Not enough observations yet to determine a trend."

    observations = history[plant_id]
    latest = observations[-1]
    previous = observations[-2]

    latest_status = "healthy" if "healthy" in latest["prediction"].lower() else "diseased"
    previous_status = "healthy" if "healthy" in previous["prediction"].lower() else "diseased"

    if previous_status == "diseased" and latest_status == "healthy":
        return "Improving — plant appears to have recovered."
    elif previous_status == "healthy" and latest_status == "diseased":
        return "Worsening — new symptoms detected since last check."
    elif previous_status == "diseased" and latest_status == "diseased":
        if latest["prediction"] != previous["prediction"]:
            return "Changed — a different disease is now detected. Needs attention."
        elif latest["confidence"] > previous["confidence"]:
            return "Worsening — same disease, symptoms appear more pronounced."
        else:
            return "Stable or slightly improving — same disease, confidence has decreased."
    else:
        return "Stable — plant remains healthy."
   
    