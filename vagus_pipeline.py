"""
NTNU PROTOTYPE — Vagus Pipeline v1
====================================
Author: Quazi Mohd Ismail, MBBS MD
Purpose: Full pipeline — HRV → ANS State → Claude Mapping → Visual Output

Run this file to execute the complete Strategy B prototype:
  Step 1: Simulate HRV window (hrv_engine)
  Step 2: Compute ANS state space (hrv_engine)
  Step 3: Map to music parameters via Claude (claude_mapper)
  Step 4: Display visual dashboard in terminal
  Step 5: Loop continuously every 30 seconds
"""

import time
import os
import sys
from dotenv import load_dotenv

# ── Import from our own modules ─────────────────────────
from hrv_engine import simulate_hrv, extract_hrv_features, compute_state_space, classify_ans_state
from claude_mapper import map_to_music, validate_music_params

load_dotenv()

# ── Visual constants ────────────────────────────────────
WIDTH = 60
BAR_WIDTH = 30

def clear():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def progress_bar(value, width=BAR_WIDTH):
    """
    Convert 0.0-1.0 value to a visual progress bar.
    Example: 0.75 → [██████████████████████░░░░░░░░]  75%
    """
    filled = int(value * width)
    empty  = width - filled
    bar    = "█" * filled + "░" * empty
    pct    = int(value * 100)
    return f"[{bar}] {pct:3d}%"

def state_color_label(state):
    """Return a simple ASCII label for ANS state."""
    if "PARASYMPATHETIC" in state:
        return "◉ PARASYMPATHETIC DOMINANT"
    elif "SYMPATHETIC" in state:
        return "◈ SYMPATHETIC DOMINANT"
    else:
        return "◎ MIXED AUTONOMIC STATE"

def print_dashboard(cycle, features, state_space, ans, music):
    """
    Print a full visual dashboard showing the complete pipeline
    output for one HRV cycle.
    """
    divider  = "═" * WIDTH
    thin     = "─" * WIDTH

    print(divider)
    print(f"  NTNU PROTOTYPE — Project Vagus Pipeline")
    print(f"  Strategy B: HRV → Claude → Music")
    print(divider)
    print(f"  Cycle: {cycle}          Time: {time.strftime('%H:%M:%S')}")
    print(thin)

    # ── Section 1: Raw HRV ──────────────────────────────
    print(f"\n  ◆ RAW HRV MARKERS")
    print(f"  {'RMSSD':<14} {features['rmssd']:>7.1f} ms   [vagal tone]")
    print(f"  {'SDNN':<14} {features['sdnn']:>7.1f} ms   [total variability]")
    print(f"  {'pNN50':<14} {features['pnn50']:>7.1f} %    [para activity]")
    print(f"  {'SD2/SD1':<14} {features['sd2_sd1']:>7.3f}      [sym load index]")

    # ── Section 2: State Space ──────────────────────────
    print(f"\n  ◆ ANS STATE SPACE  (0 = sympathetic → 1 = parasympathetic)")
    print(f"  Vagal Tone   {progress_bar(state_space['vagal_tone'])}")
    print(f"  Para Index   {progress_bar(state_space['para_activity'])}")
    print(f"  Overall HRV  {progress_bar(state_space['overall_hrv'])}")
    print(f"  Sym Load↓    {progress_bar(state_space['sym_load'])}")
    print(thin)
    score = state_space['autonomic_score']
    print(f"  AUTONOMIC    {progress_bar(score)}  ← composite")

    # ── Section 3: ANS Classification ──────────────────
    print(f"\n  ◆ ANS CLASSIFICATION")
    print(f"  {state_color_label(ans['state'])}")
    print(f"  Polyvagal:  {ans['polyvagal']}")
    print(f"  Physiology: {ans['physiology']}")

    # ── Section 4: Music Parameters ────────────────────
    print(f"\n  ◆ MUSIC PARAMETERS  (Claude Strategy B Output)")
    print(thin)

    # Tempo bar
    tempo_norm = (music['tempo_bpm'] - 40) / (120 - 40)
    print(f"  Tempo        {progress_bar(tempo_norm)}  {music['tempo_bpm']} BPM")

    # Frequency bar
    freq_norm = (music['frequency_hz'] - 110) / (440 - 110)
    print(f"  Frequency    {progress_bar(freq_norm)}  {music['frequency_hz']} Hz")

    # Complexity bar
    print(f"  Harmonic     {progress_bar(music['harmonic_complexity'])}  {music['harmonic_complexity']:.2f}")

    # Rhythm bar
    print(f"  Rhythm       {progress_bar(music['rhythmic_density'])}  {music['rhythmic_density']:.2f}")

    # Dynamics bar
    print(f"  Dynamics     {progress_bar(music['dynamics'])}  {music['dynamics']:.2f}")

    # Binaural
    bin_norm = music['binaural_offset_hz'] / 14.0
    print(f"  Binaural     {progress_bar(bin_norm)}  {music['binaural_offset_hz']} Hz")

    print(thin)
    print(f"  Rationale: {music['mapping_rationale']}")

    print(f"\n{divider}")
    print(f"  Next cycle in 30s... Press Ctrl+C to stop.")
    print(divider)


def run_pipeline(cycles=5, interval_seconds=30):
    """
    Run the full pipeline for N cycles.

    Each cycle:
    1. Simulate 120s HRV window
    2. Extract features
    3. Compute state space
    4. Classify ANS state
    5. Map to music via Claude API
    6. Display dashboard
    7. Wait interval_seconds
    """
    print("═" * WIDTH)
    print("  NTNU PROTOTYPE — Starting Pipeline")
    print("  HRV → ANS State → Claude → Music Parameters")
    print("═" * WIDTH)
    print(f"\n  Cycles planned: {cycles}")
    print(f"  Interval:       {interval_seconds}s per cycle")
    print(f"  Strategy:       B (Constrained Claude Mapping)")
    print(f"  Model:          claude-sonnet-4-5, temperature=0")
    print("\n  Starting in 3 seconds...")
    time.sleep(3)

    for cycle in range(1, cycles + 1):

        clear()
        print(f"\n  Processing cycle {cycle}/{cycles}...")
        print(f"  Step 1/3: Simulating 120s HRV window...")

        # Step 1 + 2: HRV simulation and feature extraction
        signals, info = simulate_hrv(
            duration_seconds=120,
            sampling_rate=1000,
            heart_rate=68
        )
        features = extract_hrv_features(info, sampling_rate=1000)

        print(f"  Step 2/3: Computing ANS state space...")

        # Step 3 + 4: State space and classification
        state_space = compute_state_space(features)
        ans = classify_ans_state(state_space["autonomic_score"])

        print(f"  Step 3/3: Calling Claude API for music mapping...")

        # Step 5: Claude mapping
        try:
            music = map_to_music(features, state_space, ans)
            validate_music_params(music)
        except Exception as e:
            print(f"\n  ❌ Claude API error: {e}")
            print(f"  Skipping cycle {cycle}. Retrying next interval.")
            time.sleep(interval_seconds)
            continue

        # Step 6: Display dashboard
        clear()
        print_dashboard(cycle, features, state_space, ans, music)

        # Step 7: Wait before next cycle
        if cycle < cycles:
            time.sleep(interval_seconds)

    # Final summary
    clear()
    print("═" * WIDTH)
    print("  NTNU PROTOTYPE — Pipeline Complete")
    print(f"  {cycles} cycles completed successfully.")
    print("  Strategy B: Constrained Claude Mapping verified.")
    print("  Ready for audio_engine.py integration (Day 3).")
    print("═" * WIDTH)


# ── Entry point ─────────────────────────────────────────
if __name__ == "__main__":

    # Quick mode: 3 cycles, 10s interval (for testing)
    # Full mode:  5 cycles, 30s interval (for demo)

    # Detect if running in quick test mode
    quick = "--quick" in sys.argv

    if quick:
        print("\n  Running in QUICK TEST MODE (3 cycles, 10s interval)")
        run_pipeline(cycles=3, interval_seconds=10)
    else:
        run_pipeline(cycles=5, interval_seconds=30)