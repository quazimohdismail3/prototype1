"""
NTNU PROTOTYPE — Gemini Mapper v4 (Fresh)
==========================================
Author: Quazi Mohd Ismail, MBBS MD
Model:  gemini-2.5-flash (confirmed available)
SDK:    google-genai (unified, not legacy)
Method: Pydantic schema + controlled JSON generation

Scientific Basis:
  Tempo    → Bernardi 2006, Jeong 2024
  Freq     → Calamassi & Pomponi 2019 (432Hz HRV study)
  Harmonics → Thayer & Lane 2000
  Rhythm   → HRV coherence 0.1Hz research 2025
  Binaural → McConnell et al. 2014
"""

import os
import json
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

load_dotenv()

# ── Model ───────────────────────────────────────────────
MODEL = "models/gemini-2.5-flash"

# ── Output Schema ───────────────────────────────────────
class MusicParameters(BaseModel):
    tempo_bpm: int = Field(
        ge=40, le=120,
        description="40-65=parasympathetic, 65-90=mixed, 90-120=sympathetic"
    )
    frequency_hz: int = Field(
        ge=110, le=440,
        description="110=A2 deep calm, 220=A3 balanced, 440=A4 alert"
    )
    harmonic_complexity: float = Field(
        ge=0.0, le=1.0,
        description="0.0=pure sine, 0.5=natural harmonics, 1.0=dense overtones"
    )
    rhythmic_density: float = Field(
        ge=0.0, le=1.0,
        description="0.0=sustained tones, 0.5=moderate pulse, 1.0=dense rhythm"
    )
    dynamics: float = Field(
        ge=0.0, le=1.0,
        description="0.0=very soft ppp, 0.5=moderate mf, 1.0=very loud fff"
    )
    binaural_offset_hz: float = Field(
        ge=0.0, le=14.0,
        description="0=none, 4-8=theta parasympathetic, 8-14=alpha balanced"
    )
    mapping_rationale: str = Field(
        description="One sentence: ANS state to music parameter logic"
    )


# ── System instruction ──────────────────────────────────
SYSTEM = """You are a physiological-to-music mapping function.

Map ANS state variables to music parameters using psychoacoustics.

RULES:
- autonomic_score is your primary guide (0.0=sympathetic, 1.0=parasympathetic)
- Be internally consistent — never mix sympathetic and parasympathetic parameters
- Base all decisions on the HRV markers and state space provided

MAPPING:
Parasympathetic (score >= 0.55): tempo 40-65, freq 110-220, complexity 0.0-0.3, rhythm 0.0-0.3, dynamics 0.0-0.3, binaural 4-8
Mixed (score 0.35-0.55):        tempo 65-90, freq 220-330, complexity 0.3-0.5, rhythm 0.3-0.5, dynamics 0.3-0.5, binaural 6-10
Sympathetic (score <= 0.35):    tempo 90-120, freq 330-440, complexity 0.5-0.8, rhythm 0.5-0.8, dynamics 0.5-0.8, binaural 8-14"""


def map_to_music(features: dict, state_space: dict, ans_state: dict) -> MusicParameters:
    """
    Map ANS state to music parameters via Gemini controlled generation.
    Returns validated MusicParameters Pydantic object.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # Build structured input
    payload = {
        "hrv_markers": {
            "rmssd_ms":    features["rmssd"],
            "sdnn_ms":     features["sdnn"],
            "pnn50_pct":   features["pnn50"],
            "sd2_sd1":     features["sd2_sd1"]
        },
        "state_space": {
            "vagal_tone":      state_space["vagal_tone"],
            "para_activity":   state_space["para_activity"],
            "overall_hrv":     state_space["overall_hrv"],
            "sym_load":        state_space["sym_load"],
            "autonomic_score": state_space["autonomic_score"]
        },
        "ans_state": {
            "label":      ans_state["state"],
            "polyvagal":  ans_state["polyvagal"],
            "physiology": ans_state["physiology"]
        }
    }

    response = client.models.generate_content(
        model=MODEL,
        contents=json.dumps(payload, indent=2),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json",
            response_schema=MusicParameters,
            temperature=0.0,
            max_output_tokens=1024,
        )
    )

    # Parse and validate via Pydantic
    music = MusicParameters.model_validate_json(response.text)
    return music


def validate_music_params(music: MusicParameters) -> bool:
    """Secondary validation layer — confirms Pydantic bounds held."""
    checks = {
        "tempo_bpm":           (40,  120,  music.tempo_bpm),
        "frequency_hz":        (110, 440,  music.frequency_hz),
        "harmonic_complexity": (0.0, 1.0,  music.harmonic_complexity),
        "rhythmic_density":    (0.0, 1.0,  music.rhythmic_density),
        "dynamics":            (0.0, 1.0,  music.dynamics),
        "binaural_offset_hz":  (0.0, 14.0, music.binaural_offset_hz),
    }
    for param, (lo, hi, val) in checks.items():
        if not (lo <= float(val) <= hi):
            raise ValueError(f"{param}={val} outside [{lo},{hi}]")
    return True


# ── Standalone test ─────────────────────────────────────
if __name__ == "__main__":

    print("═" * 55)
    print("  NTNU PROTOTYPE — Gemini Mapper v4")
    print(f"  Model: {MODEL}")
    print("  Method: Pydantic schema + controlled JSON")
    print("═" * 55)

    tests = [
        {
            "label": "PARASYMPATHETIC DOMINANT",
            "features":    {"rmssd": 68.0, "sdnn": 78.0, "pnn50": 35.0, "sd2_sd1": 1.3},
            "state_space": {"vagal_tone": 0.85, "para_activity": 0.82,
                            "overall_hrv": 0.79, "sym_load": 0.82, "autonomic_score": 0.83},
            "ans_state":   {"state": "PARASYMPATHETIC DOMINANT",
                            "polyvagal": "Ventral vagal — safe and social",
                            "physiology": "High vagal tone. Recovery mode."}
        },
        {
            "label": "MIXED AUTONOMIC STATE",
            "features":    {"rmssd": 42.0, "sdnn": 50.0, "pnn50": 19.0, "sd2_sd1": 2.0},
            "state_space": {"vagal_tone": 0.50, "para_activity": 0.47,
                            "overall_hrv": 0.50, "sym_load": 0.50, "autonomic_score": 0.49},
            "ans_state":   {"state": "MIXED AUTONOMIC STATE",
                            "polyvagal": "Transitional",
                            "physiology": "Neither system dominant."}
        },
        {
            "label": "SYMPATHETIC DOMINANT",
            "features":    {"rmssd": 18.0, "sdnn": 22.0, "pnn50": 2.0, "sd2_sd1": 3.8},
            "state_space": {"vagal_tone": 0.05, "para_activity": 0.00,
                            "overall_hrv": 0.10, "sym_load": 0.14, "autonomic_score": 0.05},
            "ans_state":   {"state": "SYMPATHETIC DOMINANT",
                            "polyvagal": "Sympathetic mobilisation",
                            "physiology": "Reduced vagal tone. Stress response."}
        }
    ]

    passed = 0
    failed = 0

    for i, t in enumerate(tests, 1):
        print(f"\n── Test {i}/3: {t['label']}")

        try:
            music = map_to_music(t["features"], t["state_space"], t["ans_state"])
            validate_music_params(music)

            print(f"  ✅ Validated")
            print(f"  Tempo:     {music.tempo_bpm} BPM")
            print(f"  Frequency: {music.frequency_hz} Hz")
            print(f"  Harmonics: {music.harmonic_complexity:.2f}")
            print(f"  Rhythm:    {music.rhythmic_density:.2f}")
            print(f"  Dynamics:  {music.dynamics:.2f}")
            print(f"  Binaural:  {music.binaural_offset_hz} Hz")
            print(f"  Rationale: {music.mapping_rationale}")
            passed += 1

        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1

        if i < len(tests):
            time.sleep(3)

    print(f"\n{'═' * 55}")
    print(f"  Results: {passed}/3 passed, {failed}/3 failed")
    if failed == 0:
        print("  ✅ Gemini Mapper v4 ready for pipeline.")
    print("═" * 55)