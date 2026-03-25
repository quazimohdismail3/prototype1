"""
NTNU PROTOTYPE — Vagus Pipeline v2
=====================================
Author: Quazi Mohd Ismail, MBBS MD
Project: Project Vagus / NTNU Prototype
Target:  Prof. Andreas Bergsland, NTNU

Full closed-loop pipeline:
  HRV Simulation → ANS State Space → Gemini Mapping → Audio Parameters
  With trend tracking, clinical interpretation, and visual dashboard.

Run modes:
  python vagus_pipeline.py           → full demo (5 cycles, 30s interval)
  python vagus_pipeline.py --quick   → test mode (3 cycles, 10s interval)

Scientific Basis:
  HRV Engine:     Task Force ESC 1996, Shaffer & Ginsberg 2017
  ANS Mapping:    Porges Polyvagal Theory 2011
  Music Params:   Bernardi 2006, Thayer & Lane 2000, McConnell 2014
  Gemini Mapper:  Constrained LLM mapping, temperature=0, Pydantic schema
"""

import os
import sys
import time
from collections import deque
from dotenv import load_dotenv

from hrv_engine import (
    simulate_hrv,
    extract_hrv_features,
    compute_state_space,
    classify_ans_state,
)
from gemini_mapper import map_to_music, validate_music_params

# Audio engine — degrades gracefully if sounddevice not installed
try:
    from audio_engine import play_audio, stop_audio
    AUDIO_ENABLED = True
except ImportError:
    AUDIO_ENABLED = False
    def play_audio(*args, **kwargs): pass   # noqa: E731
    def stop_audio(): pass                  # noqa: E731

load_dotenv()

# ── Display constants ───────────────────────────────────
W          = 62          # total dashboard width
BAR        = 28          # progress bar width
DIVIDER    = "═" * W
THIN       = "─" * W
TREND_LEN  = 5           # cycles to track for trend


# ═══════════════════════════════════════════════════════
# DISPLAY UTILITIES
# ═══════════════════════════════════════════════════════

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def bar(value: float, width: int = BAR) -> str:
    """
    Convert 0.0-1.0 float to visual progress bar.
    0.0 → all empty  |  1.0 → all filled
    """
    value   = max(0.0, min(1.0, value))
    filled  = int(value * width)
    empty   = width - filled
    pct     = int(value * 100)
    return f"[{'█' * filled}{'░' * empty}] {pct:3d}%"


def trend_arrow(history: deque) -> str:
    """
    Compare last two autonomic scores.
    Returns arrow showing direction of ANS trend.
    ↑ = improving (more parasympathetic)
    ↓ = worsening (more sympathetic)
    → = stable
    """
    if len(history) < 2:
        return "→ (establishing baseline)"
    delta = history[-1] - history[-2]
    if delta > 0.05:
        return f"↑ IMPROVING  (+{delta:.3f})"
    elif delta < -0.05:
        return f"↓ WORSENING  ({delta:.3f})"
    else:
        return f"→ STABLE     ({delta:+.3f})"


def trend_summary(history: deque) -> str:
    """
    Summarise ANS trend across all recorded cycles.
    Compares first and last autonomic score.
    """
    if len(history) < 2:
        return "Insufficient data for trend analysis."
    start   = history[0]
    current = history[-1]
    delta   = current - start
    cycles  = len(history)
    if delta > 0.1:
        return f"Autonomic recovery in progress over {cycles} cycles (+{delta:.3f})"
    elif delta < -0.1:
        return f"Autonomic load increasing over {cycles} cycles ({delta:.3f})"
    else:
        return f"Autonomic state stable over {cycles} cycles ({delta:+.3f})"


def clinical_note(state: dict, features: dict, music: dict) -> str:
    """
    Generate a one-line clinical interpretation of the current cycle.
    Grounded in Polyvagal Theory (Porges 2011) and HRV research.

    Clinical thresholds:
    - RMSSD < 20ms: critically low vagal tone (Shaffer 2017)
    - RMSSD 20-40ms: reduced vagal tone, sympathetic dominance
    - RMSSD > 40ms: adequate-to-high vagal tone
    - pNN50 < 5%: minimal parasympathetic activity
    """
    rmssd   = features["rmssd"]
    pnn50   = features["pnn50"]
    score   = state["autonomic_score"]
    tempo   = music["tempo_bpm"]
    freq    = music["frequency_hz"]

    if rmssd < 20:
        return (
            f"⚠  Critical: RMSSD {rmssd:.1f}ms — severely reduced vagal tone. "
            f"Music targeting recovery: {tempo} BPM at {freq} Hz."
        )
    elif rmssd < 35 or score < 0.35:
        return (
            f"◈  Sympathetic load: RMSSD {rmssd:.1f}ms, pNN50 {pnn50:.1f}%. "
            f"Slow music intervention: {tempo} BPM at {freq} Hz."
        )
    elif score >= 0.55:
        return (
            f"◉  Good vagal tone: RMSSD {rmssd:.1f}ms, pNN50 {pnn50:.1f}%. "
            f"Maintenance music: {tempo} BPM at {freq} Hz."
        )
    else:
        return (
            f"◎  Balanced ANS: RMSSD {rmssd:.1f}ms, pNN50 {pnn50:.1f}%. "
            f"Transitional music: {tempo} BPM at {freq} Hz."
        )


def state_icon(state_label: str) -> str:
    if "PARASYMPATHETIC" in state_label:
        return "◉ PARASYMPATHETIC DOMINANT"
    elif "SYMPATHETIC" in state_label:
        return "◈ SYMPATHETIC DOMINANT"
    else:
        return "◎ MIXED AUTONOMIC STATE"


# ═══════════════════════════════════════════════════════
# DASHBOARD RENDERER
# ═══════════════════════════════════════════════════════

def render_dashboard(
    cycle:       int,
    total:       int,
    features:    dict,
    state_space: dict,
    ans:         dict,
    music:       dict,
    score_history: deque,
):
    """
    Render the full visual dashboard for one pipeline cycle.

    Sections:
    1. Header — cycle info, timestamp
    2. Raw HRV markers — with reference ranges
    3. ANS state space — progress bars
    4. ANS classification — Polyvagal label
    5. Music parameters — progress bars + values
    6. Clinical note — one-line interpretation
    7. Trend analysis — direction + summary
    """
    print(DIVIDER)
    print(f"  NTNU PROTOTYPE  ·  Project Vagus Pipeline v2")
    print(f"  Strategy B: HRV → Gemini → Music Parameters")
    print(DIVIDER)
    print(f"  Cycle {cycle}/{total}          {time.strftime('%H:%M:%S')}          gemini-2.5-flash")
    print(THIN)

    # ── Section 1: Raw HRV ──────────────────────────────
    print(f"\n  ◆ RAW HRV MARKERS          [reference: Shaffer 2017]")
    print(f"  {'RMSSD':<14} {features['rmssd']:>7.1f} ms      norm: 42 ± 15 ms")
    print(f"  {'SDNN':<14} {features['sdnn']:>7.1f} ms      norm: 50 ± 16 ms")
    print(f"  {'pNN50':<14} {features['pnn50']:>7.1f} %       norm: 19 ± 15 %")
    print(f"  {'SD2/SD1':<14} {features['sd2_sd1']:>7.3f}         sym load index")

    # ── Section 2: State Space ──────────────────────────
    print(f"\n  ◆ ANS STATE SPACE          [0 = sympathetic → 1 = parasympathetic]")
    print(f"  Vagal Tone    {bar(state_space['vagal_tone'])}")
    print(f"  Para Activity {bar(state_space['para_activity'])}")
    print(f"  Overall HRV   {bar(state_space['overall_hrv'])}")
    print(f"  Sym Load ↓    {bar(state_space['sym_load'])}")
    print(f"  {THIN[2:]}")
    print(f"  AUTONOMIC     {bar(state_space['autonomic_score'])}  ← composite")

    # ── Section 3: ANS Classification ──────────────────
    print(f"\n  ◆ ANS CLASSIFICATION       [Porges Polyvagal Theory 2011]")
    print(f"  {state_icon(ans['state'])}")
    print(f"  Polyvagal:  {ans['polyvagal']}")
    print(f"  Physiology: {ans['physiology']}")

    # ── Section 4: Music Parameters ────────────────────
    print(f"\n  ◆ MUSIC PARAMETERS         [Gemini Strategy B Output]")
    print(THIN)

    tempo_norm = (music["tempo_bpm"] - 40) / 80
    freq_norm  = (music["frequency_hz"] - 110) / 330
    bin_norm   = music["binaural_offset_hz"] / 14.0

    print(f"  Tempo         {bar(tempo_norm)}  {music['tempo_bpm']} BPM")
    print(f"  Frequency     {bar(freq_norm)}  {music['frequency_hz']} Hz")
    print(f"  Harmonics     {bar(music['harmonic_complexity'])}  {music['harmonic_complexity']:.2f}")
    print(f"  Rhythm        {bar(music['rhythmic_density'])}  {music['rhythmic_density']:.2f}")
    print(f"  Dynamics      {bar(music['dynamics'])}  {music['dynamics']:.2f}")
    print(f"  Binaural      {bar(bin_norm)}  {music['binaural_offset_hz']} Hz")
    print(THIN)
    print(f"  Rationale: {music['mapping_rationale']}")

    # ── Section 5: Clinical Note ────────────────────────
    print(f"\n  ◆ CLINICAL INTERPRETATION  [Porges 2011, Shaffer 2017]")
    print(f"  {clinical_note(state_space, features, music)}")

    # ── Section 6: Trend Analysis ───────────────────────
    print(f"\n  ◆ ANS TREND")
    print(f"  This cycle:  {trend_arrow(score_history)}")
    print(f"  Overall:     {trend_summary(score_history)}")

    # Sparkline of autonomic scores
    if len(score_history) > 1:
        sparkline = ""
        for s in score_history:
            if s >= 0.55:
                sparkline += "▂"
            elif s >= 0.35:
                sparkline += "▄"
            else:
                sparkline += "▆"
        print(f"  History:     {sparkline}  (low=sym, high=para)")

    print(f"\n{DIVIDER}")
    print(f"  Next cycle in {30 if total > 3 else 10}s  ·  Ctrl+C to stop")
    print(DIVIDER)


# ═══════════════════════════════════════════════════════
# PIPELINE RUNNER
# ═══════════════════════════════════════════════════════

def run_pipeline(cycles: int = 5, interval: int = 30):
    """
    Run the full closed-loop pipeline.

    Each cycle:
    1. Simulate 120s HRV window (NeuroKit2)
    2. Extract validated HRV features
    3. Compute normalised ANS state space
    4. Classify ANS state (3-state Polyvagal model)
    5. Map to music parameters via Gemini API
    6. Validate all parameters within scientific bounds
    7. Render visual dashboard
    8. Track autonomic trend across cycles
    9. Wait interval seconds before next cycle
    """
    score_history = deque(maxlen=TREND_LEN)

    clear()
    print(DIVIDER)
    print("  NTNU PROTOTYPE — Vagus Pipeline v2")
    print("  HRV → ANS State → Gemini → Music Parameters")
    print(DIVIDER)
    print(f"\n  Mode:      {'Quick Test' if cycles <= 3 else 'Full Demo'}")
    print(f"  Cycles:    {cycles}")
    print(f"  Interval:  {interval}s per cycle")
    print(f"  Strategy:  B — Constrained Gemini Mapping")
    print(f"  Model:     gemini-2.5-flash | temperature=0")
    if AUDIO_ENABLED:
        print(f"  Audio:     sounddevice additive synthesis + binaural beats")
    else:
        print(f"  Audio:     disabled  (pip install sounddevice)")
    print(f"\n  Starting in 3 seconds...")
    time.sleep(3)

    completed = 0
    failed    = 0

    for cycle in range(1, cycles + 1):
        stop_audio()   # stop audio from previous cycle (no-op on first)
        clear()
        print(f"\n  Processing cycle {cycle}/{cycles}...")
        print(f"  Step 1: Simulating 120s HRV window (NeuroKit2)...")

        try:
            # Step 1-2: HRV simulation + feature extraction
            signals, info = simulate_hrv(
                duration_seconds=120,
                sampling_rate=1000,
                heart_rate=68
            )
            features = extract_hrv_features(info, sampling_rate=1000)

            print(f"  Step 2: Computing ANS state space...")

            # Step 3-4: State space + classification
            state_space = compute_state_space(features)
            ans         = classify_ans_state(state_space["autonomic_score"])

            print(f"  Step 3: Calling Gemini API for music mapping...")

            # Step 5-6: Gemini mapping + validation
            music_obj = map_to_music(features, state_space, ans)
            validate_music_params(music_obj)
            music = music_obj.model_dump()

            # Step 7: Track trend
            score_history.append(state_space["autonomic_score"])

            # Step 8: Render dashboard
            clear()
            render_dashboard(
                cycle, cycles,
                features, state_space, ans, music,
                score_history
            )

            # Step 9: Play audio for this cycle (non-blocking)
            if AUDIO_ENABLED:
                audio_duration = max(3.0, float(interval) - 3.0)
                play_audio(music_obj, duration_seconds=audio_duration, blocking=False)
                right_freq = music_obj.frequency_hz + music_obj.binaural_offset_hz
                print(f"\n  ♪  {music_obj.frequency_hz} Hz (L) / {right_freq:.1f} Hz (R)"
                      f"  ·  {music_obj.tempo_bpm} BPM"
                      f"  ·  {music_obj.binaural_offset_hz:.1f} Hz binaural"
                      f"  ·  {audio_duration:.0f}s")

            completed += 1

        except KeyboardInterrupt:
            stop_audio()
            print("\n\n  Pipeline stopped by user.")
            break

        except Exception as e:
            print(f"\n  ❌ Cycle {cycle} error: {e}")
            print(f"  Retrying next interval...")
            failed += 1

        # Wait before next cycle
        if cycle < cycles:
            time.sleep(interval)

    # ── Final Summary ───────────────────────────────────
    stop_audio()
    clear()
    print(DIVIDER)
    print("  NTNU PROTOTYPE — Pipeline Complete")
    print(DIVIDER)
    print(f"\n  Cycles completed: {completed}/{cycles}")
    print(f"  Cycles failed:    {failed}/{cycles}")

    if score_history:
        avg_score = sum(score_history) / len(score_history)
        print(f"\n  Average autonomic score: {avg_score:.3f}")
        print(f"  Final trend: {trend_summary(score_history)}")

        if avg_score >= 0.55:
            print(f"\n  Overall state: PARASYMPATHETIC DOMINANT")
            print(f"  System in recovery. High vagal tone sustained.")
        elif avg_score <= 0.35:
            print(f"\n  Overall state: SYMPATHETIC DOMINANT")
            print(f"  Autonomic load detected. Music intervention active.")
        else:
            print(f"\n  Overall state: MIXED / TRANSITIONAL")
            print(f"  Balanced autonomic regulation observed.")

    print(f"\n  Strategy B verified across {completed} cycles.")
    if AUDIO_ENABLED:
        print(f"  Audio engine: additive synthesis + binaural beats active.")
    print(f"\n{DIVIDER}")


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    quick = "--quick" in sys.argv

    if quick:
        run_pipeline(cycles=3, interval=10)
    else:
        run_pipeline(cycles=5, interval=30)