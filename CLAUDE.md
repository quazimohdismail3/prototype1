# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Project Vagus** — a physiological music mapping system that reads HRV (Heart Rate Variability) signals, classifies autonomic nervous system (ANS) state, and uses the Gemini LLM to map that state to therapeutic music parameters. Built as a research prototype for NTNU (Prof. Andreas Bergsland).

## Setup

No `requirements.txt` exists — install dependencies manually:

```bash
pip install neurokit2 numpy pydantic google-genai python-dotenv sounddevice
```

Requires a `.env` file with:
```
GEMINI_API_KEY=<your_google_gemini_api_key>
```

## Running the Pipeline

```bash
# Full demo: 5 cycles, 30s intervals (audio plays each cycle)
python vagus_pipeline.py

# Quick test: 3 cycles, 10s intervals
python vagus_pipeline.py --quick

# Test individual components in isolation
python hrv_engine.py       # HRV simulation + ANS classification only
python gemini_mapper.py    # Gemini API mapping only (3 hardcoded test cases)
python audio_engine.py     # Audio synthesis only (plays 3 ANS states, 6s each)

# Check available Gemini models
python CHECK_MODELS.py
```

## Architecture

Three-layer sequential pipeline, each module independently executable:

### Layer 1 & 2 — `hrv_engine.py`
Simulates 120s ECG at 1000 Hz using NeuroKit2, extracts 6 HRV metrics (RMSSD, SDNN, pNN50, SD1, SD2, SD2/SD1), normalizes them to a **5-dimensional state space** `[0,1]`, and classifies into one of 3 ANS states:
- **PARASYMPATHETIC DOMINANT** — autonomic_score ≥ 0.55
- **SYMPATHETIC DOMINANT** — autonomic_score ≤ 0.35
- **MIXED AUTONOMIC STATE** — between 0.35 and 0.55

The `autonomic_score` is a weighted composite: 45% RMSSD + 30% pNN50 + 15% SDNN + 10% inverted SD2/SD1.

### Layer 3 — `gemini_mapper.py`
Sends the HRV state space + ANS classification to **Gemini 2.5 Flash** (`temperature=0`) with a system prompt encoding Polyvagal Theory rules. Returns a Pydantic-validated `MusicParameters` object with bounds:
- `tempo_bpm`: 40–120
- `frequency_hz`: 110–440 (A2–A4)
- `harmonic_complexity`: 0.0–1.0
- `rhythmic_density`: 0.0–1.0
- `dynamics`: 0.0–1.0
- `binaural_offset_hz`: 0–14 (theta/alpha bands)

### Layer 4 — `audio_engine.py`
Generates real-time stereo audio from `MusicParameters` using `sounddevice`. Three synthesis layers applied in sequence:
1. **Additive synthesis** — `harmonic_complexity` controls partials (1=pure sine, 8=full overtone series with 1/n rolloff)
2. **Amplitude modulation** — `rhythmic_density` drives a cosine LFO at `tempo_bpm` rate (0=sustained, 75% max depth)
3. **Binaural beats** — left channel at `frequency_hz`, right channel at `frequency_hz + binaural_offset_hz` (inactive when offset ≤ 0.5 Hz)

Dynamics maps 0→1 to amplitude 0.05→0.45 to preserve headroom. 50 ms linear fades prevent clicks. `play_audio()` is non-blocking by default; `stop_audio()` halts playback immediately.

### Layer 5 — `vagus_pipeline.py`
Orchestrates the full loop: HRV → state space → Gemini → music params → terminal dashboard → audio. Audio plays non-blocking for `interval - 3` seconds per cycle; previous cycle's audio is stopped at the start of each new cycle. Gracefully disables audio if `sounddevice` is not installed. Tracks a rolling `deque(maxlen=5)` of autonomic scores for trend/sparkline visualization.

## Scientific Grounding

- HRV standards: Task Force ESC/NASPE 1996, Shaffer & Ginsberg 2017
- ANS model: Polyvagal Theory (Porges 2011)
- Music-physiology coupling: Bernardi 2006, Thayer & Lane 2000, McConnell et al. 2014

## Current Status & Next Steps

Day 2 complete: HRV engine + Gemini mapper + dashboard pipeline verified.
**Day 3 goal:** Implement `audio_engine.py` — actual audio synthesis from `MusicParameters`.
