"""Field Diagnosis History and Multi-Visit Progress Tracking Module.

Provides persistent longitudinal crop monitoring:
- Stores photographic diagnosis records, dates, growth stages, and farmer field observations.
- Compares repeated visits over time (e.g., Day 0 Baseline vs Day 7 vs Day 14 Follow-up).
- Tracks whether a disease or pest infestation is spreading, stabilizing, or recovering.

DISCLAIMER:
Repeated field visits across multiple dates are essential to evaluate disease
trajectory. A single photographic observation cannot determine recovery.
"""

from pathlib import Path
from datetime import datetime
import json
import base64
import io
from typing import List, Dict, Any, Optional
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent
HISTORY_FILE = ROOT_DIR / "data" / "diagnosis_history.json"


def _image_to_base64_thumbnail(image: Image.Image, max_size=(140, 140)) -> str:
    """Generates a compact base64 thumbnail string for persistent history storage."""
    thumb = image.copy()
    thumb.thumbnail(max_size)
    buffered = io.BytesIO()
    thumb.save(buffered, format="JPEG", quality=75)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def load_diagnosis_history() -> List[Dict[str, Any]]:
    """Loads all saved historical field diagnoses."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_diagnosis_record(
    field_name: str,
    crop: str,
    crop_stage: str,
    condition_raw: str,
    condition_display: str,
    severity: str,
    confidence: float,
    farmer_notes: str = "",
    image: Optional[Image.Image] = None,
) -> Dict[str, Any]:
    """Appends a new field diagnosis record to the persistent history."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = load_diagnosis_history()

    now = datetime.now()
    record_id = f"REC-{now.strftime('%Y%m%d-%H%M%S')}"

    thumb_b64 = _image_to_base64_thumbnail(image) if image is not None else ""

    record = {
        "id": record_id,
        "date_str": now.strftime("%Y-%m-%d %H:%M"),
        "timestamp": now.isoformat(),
        "field_name": field_name.strip() or "General Field Plot",
        "crop": crop,
        "crop_stage": crop_stage,
        "condition_raw": condition_raw,
        "condition_display": condition_display,
        "severity": severity,
        "confidence": round(confidence, 1),
        "farmer_notes": farmer_notes.strip(),
        "thumbnail_b64": thumb_b64,
    }

    history.insert(0, record)  # latest first
    with HISTORY_FILE.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return record


def analyze_field_progression(field_name: str) -> Dict[str, Any]:
    """Analyzes repeated visits for a specific field plot to evaluate disease trajectory."""
    history = load_diagnosis_history()
    records = [r for r in history if r["field_name"].lower() == field_name.lower().strip()]
    records.sort(key=lambda r: r["timestamp"])  # chronological order

    if len(records) < 2:
        return {
            "has_progression": False,
            "status": "Insufficient Multi-Visit Data",
            "message": "Only 1 diagnosis record found for this plot. Record a follow-up diagnosis (after 5–10 days) to track symptom progression.",
            "records": records,
        }

    first = records[0]
    latest = records[-1]

    # Severity scoring: Healthy=0, Medium=1, High=2, Critical=3
    sev_map = {"Healthy": 0, "Medium": 1, "High": 2, "Critical": 3}
    s_first = sev_map.get(first["severity"], 1)
    s_latest = sev_map.get(latest["severity"], 1)

    t_first = datetime.fromisoformat(first["timestamp"])
    t_latest = datetime.fromisoformat(latest["timestamp"])
    days_elapsed = max(1, (t_latest - t_first).days)

    if s_latest < s_first or (s_latest == 0 and s_first > 0):
        trajectory = "Improving / Recovering"
        badge_color = "🟢"
        analysis = (
            f"Foliage condition has improved over {days_elapsed} days (from {first['condition_display']} "
            f"to {latest['condition_display']}). Applied interventions appear to be successfully controlling the pathogen."
        )
    elif s_latest > s_first:
        trajectory = "Deteriorating / Spreading"
        badge_color = "🔴"
        analysis = (
            f"Condition severity has escalated over {days_elapsed} days (from {first['condition_display']} "
            f"to {latest['condition_display']}). Previous treatments were insufficient; consider secondary biocontrol or targeted chemical intervention."
        )
    else:
        trajectory = "Stabilized / Active"
        badge_color = "🟠"
        analysis = (
            f"Condition remains active but stable over {days_elapsed} days ({latest['condition_display']}). "
            "Continue recommended hygiene and preventive measures."
        )

    return {
        "has_progression": True,
        "trajectory": trajectory,
        "badge_color": badge_color,
        "days_elapsed": days_elapsed,
        "analysis": analysis,
        "first_visit": first,
        "latest_visit": latest,
        "all_records": records,
    }
