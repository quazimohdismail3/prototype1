"""
NTNU PROTOTYPE — ISO Principle 7-Day Arc Player
=================================================
Author: Quazi Mohd Ismail, MBBS MD

Applies the ISO Principle from music therapy to real WHOOP HRV data:

  ISO Principle:
    1. Mirror the patient's current ANS state (Day 1 anchor)
    2. Gradually steer music toward parasympathetic recovery (Days 2–7)
    3. The nervous system follows the music through entrainment

One cycle = one 7-day WHOOP window.
Each day: 15 seconds of audio with crossfade into the next day.
One full window: ~1 min 45 s of continuous music.

Run modes:
  python iso_player.py            → all weeks, week selection menu
  python iso_player.py --quick    → first 3 weeks, auto-advance
  python iso_player.py --sample   → 3 bundled demo weeks, selection menu
  python iso_player.py --sample --quick  → demo, auto-advance
"""

import os
import sys
import time
import statistics

from dotenv import load_dotenv
load_dotenv()

from gemini_mapper import MusicParameters, map_to_music
from hrv_engine    import compute_state_space, classify_ans_state
from music_state   import MusicState
from whoop_csv_loader import load_whoop_weekly_windows

try:
    from whoop_csv_loader import load_sample_weeks
    SAMPLE_LOADER_AVAILABLE = True
except ImportError:
    SAMPLE_LOADER_AVAILABLE = False
    def load_sample_weeks(): return []

try:
    import sounddevice as sd
    from iso_audio import (
        play_iso_audio, stop_iso_audio, generate_iso_audio,
        concat_iso_week, CROSSFADE_SEC, SAMPLE_RATE,
    )
    AUDIO_ENABLED = True
except ImportError:
    AUDIO_ENABLED = False
    CROSSFADE_SEC = 2.0
    SAMPLE_RATE   = 44100
    def play_iso_audio(*a, **kw): pass
    def stop_iso_audio(): pass
    def generate_iso_audio(*a, **kw): return None
    def concat_iso_week(*a, **kw): return None
    sd = None

# ── Parasympathetic recovery target ────────────────────────────────────────────
# Day 7 of every ISO arc converges here regardless of where Day 1 started.
# Values chosen at the deep end of the parasympathetic range:
#   tempo 50 BPM, A2-adjacent root (128 Hz), theta binaural (6 Hz)
PARA_TARGET = MusicParameters(
    tempo_bpm          = 50,
    frequency_hz       = 128,
    harmonic_complexity= 0.1,
    rhythmic_density   = 0.05,
    dynamics           = 0.25,
    binaural_offset_hz = 6.0,
    mapping_rationale  = "Parasympathetic recovery target — ISO arc endpoint"
)

# Display
W       = 58
DIVIDER = "═" * W
DAY_SEC = 15        # seconds of audio per day (crossfade overlaps between days)


# ═══════════════════════════════════════════════════════
# ARC CONSTRUCTION
# ═══════════════════════════════════════════════════════

def _interpolate(p1: MusicParameters, p2: MusicParameters,
                 alpha: float) -> MusicParameters:
    """Linear blend of all 6 numeric MusicParameters fields."""
    def lerp(a, b): return a + alpha * (b - a)
    return MusicParameters(
        tempo_bpm          = int(round(lerp(p1.tempo_bpm,           p2.tempo_bpm))),
        frequency_hz       = int(round(lerp(p1.frequency_hz,        p2.frequency_hz))),
        harmonic_complexity= round(lerp(p1.harmonic_complexity, p2.harmonic_complexity), 3),
        rhythmic_density   = round(lerp(p1.rhythmic_density,    p2.rhythmic_density),    3),
        dynamics           = round(lerp(p1.dynamics,             p2.dynamics),            3),
        binaural_offset_hz = round(lerp(p1.binaural_offset_hz,  p2.binaural_offset_hz),  2),
        mapping_rationale  = f"ISO arc day {alpha:.2f} — interpolated",
    )


def build_iso_arc(day1_music: MusicParameters,
                  n_days: int = 7) -> list[MusicParameters]:
    """
    Build the ISO arc: n_days MusicParameters objects from Day 1 match → PARA_TARGET.

    Day 1 (i=0, alpha=0.0): mirrors actual ANS state (ISO anchor)
    Day 7 (i=6, alpha=1.0): parasympathetic recovery target
    Days 2–6: linear interpolation between the two
    """
    arc = []
    for i in range(n_days):
        alpha = i / (n_days - 1) if n_days > 1 else 0.0
        if alpha == 0.0:
            arc.append(day1_music)
        elif alpha == 1.0:
            arc.append(PARA_TARGET)
        else:
            arc.append(_interpolate(day1_music, PARA_TARGET, alpha))
    return arc


# ═══════════════════════════════════════════════════════
# ISO START POINT — GEMINI MAPPING FOR DAY 1
# ═══════════════════════════════════════════════════════

def _get_day1_music(features: dict) -> tuple[MusicParameters, dict, bool]:
    """
    Map Day 1 HRV features to music parameters via Gemini.
    Returns (music_params, ans_state_dict, gemini_used).
    Falls back to state-based mapping if Gemini is unavailable.
    """
    state_space = compute_state_space(features)
    ans         = classify_ans_state(state_space["autonomic_score"])

    try:
        music = map_to_music(features, state_space, ans)
        return music, ans, True
    except Exception:
        rmssd        = features.get("rmssd", 50.0)
        state_params = MusicState().update(rmssd, "stable", 0)
        music        = MusicParameters(**state_params)
        return music, ans, False


# ═══════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════

def _clear():
    os.system("cls" if os.name == "nt" else "clear")


def _arc_bar(day_idx: int, n_days: int, width: int = 30) -> str:
    """Visual progress bar showing position on the ISO arc."""
    filled = round((day_idx / (n_days - 1)) * width) if n_days > 1 else 0
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def render_day_dashboard(
    window_idx:  int,
    n_windows:   int,
    window:      dict,
    day_idx:     int,
    n_days:      int,
    music:       MusicParameters,
    ans:         dict,
    gemini_used: bool,
) -> None:
    """Print the per-day status dashboard."""
    _clear()

    day_num  = day_idx + 1
    arc_pos  = day_idx / (n_days - 1) if n_days > 1 else 0.0

    from iso_audio import _RATIO_MATCH, _RATIO_TARGET
    ratio = _RATIO_MATCH + arc_pos * (_RATIO_TARGET - _RATIO_MATCH)

    if day_idx == 0:
        status = "MATCHING   — mirroring actual ANS state"
    elif day_idx == n_days - 1:
        status = "TARGET     — full parasympathetic"
    else:
        status = "GUIDING    — steering toward recovery"

    hrv_vals  = window.get("hrv_values", [])
    day_rmssd = hrv_vals[day_idx] if day_idx < len(hrv_vals) else window["mean_rmssd"]
    mapping_src = "Gemini 2.5-flash" if gemini_used else "state-based fallback"

    right_freq = music.frequency_hz + music.binaural_offset_hz

    print(DIVIDER)
    print(f"  NTNU PROTOTYPE — ISO Arc Player")
    print(DIVIDER)
    print(f"  Week {window_idx + 1}/{n_windows}  ·  "
          f"{window['date_start']} → {window['date_end']}")
    print(f"  Day {day_num}/{n_days}  ·  RMSSD = {day_rmssd:.0f} ms")
    print()
    print(f"  ANS Anchor (Day 1):  {ans['state']}")
    print(f"  Autonomic score:     "
          f"{compute_state_space(window)['autonomic_score']:.3f}")
    print(f"  Day-1 map source:    {mapping_src}")
    print()
    print(f"  ISO Arc:   {_arc_bar(day_idx, n_days)}  Day {day_num}/{n_days}")
    print(f"  Status:    {status}")
    print()
    print(f"  Music today:")
    print(f"    Tempo:            {music.tempo_bpm} BPM")
    print(f"    Frequency:        L {music.frequency_hz} Hz  "
          f"→  R {right_freq:.1f} Hz  (Δ {music.binaural_offset_hz:.1f} Hz binaural)")
    print(f"    Harmony interval: {ratio:.3f}  "
          f"({'major third' if arc_pos < 0.1 else 'perfect fifth' if arc_pos > 0.9 else 'gliding'})")
    print(f"    Harmonic layer:   {music.harmonic_complexity:.2f}")
    print(f"    Rhythmic pulse:   {music.rhythmic_density:.2f}  "
          f"({'accents active' if music.rhythmic_density > 0.15 else 'sustained only'})")
    print(f"    Dynamics:         {music.dynamics:.2f}")
    print()
    print(DIVIDER)
    if AUDIO_ENABLED:
        arrow = "→ next day" if day_num < n_days else "→ next week"
        print(f"  Playing {DAY_SEC}s with {CROSSFADE_SEC:.0f}s crossfade "
              f"[Day {day_num}]  ({arrow})")
    else:
        print(f"  [Audio disabled — pip install sounddevice]")
    print(DIVIDER)


# ═══════════════════════════════════════════════════════
# WEEK SELECTION MENU
# ═══════════════════════════════════════════════════════

def _categorise(windows: list[dict]) -> dict:
    """Group windows into HRV-level and variance categories."""
    para   = [w for w in windows if w["mean_rmssd"] >= 95]
    mixed  = [w for w in windows if 75 <= w["mean_rmssd"] < 95]
    sym    = [w for w in windows if w["mean_rmssd"] < 75]
    hv     = [w for w in windows
              if (statistics.stdev(w["hrv_values"]) >= 12
                  if len(w.get("hrv_values", [])) > 1 else False)]
    return {"para": para, "mixed": mixed, "sym": sym, "hv": hv}


def select_weeks_menu(windows: list[dict]) -> list[dict]:
    """
    Show an interactive category menu and return the selected week subset.

    Categories:
      Parasympathetic — mean RMSSD ≥ 95 ms
      Balanced        — mean RMSSD 75–95 ms
      Sympathetic     — mean RMSSD < 75 ms
      High variance   — intra-week std ≥ 12 ms (most dynamic, best for demo)
    """
    cats = _categorise(windows)

    while True:
        _clear()
        print(DIVIDER)
        print(f"  NTNU PROTOTYPE — Week Selection")
        print(DIVIDER)
        print(f"\n  Choose which weeks to play:\n")
        print(f"  [1] All weeks              ({len(windows)} total)")
        print(f"  [2] Parasympathetic        "
              f"(high HRV ≥ 95 ms,   {len(cats['para'])} weeks)")
        print(f"  [3] Balanced / Mixed       "
              f"(HRV 75–95 ms,       {len(cats['mixed'])} weeks)")
        print(f"  [4] Sympathetic            "
              f"(low HRV < 75 ms,    {len(cats['sym'])} weeks)")
        print(f"  [5] High variance          "
              f"(std ≥ 12 ms,        {len(cats['hv'])} weeks) ← best for demo")
        print(f"  [6] Choose individual week...")
        print()
        print(f"  Enter choice [1-6]: ", end="", flush=True)

        try:
            choice = input().strip()
        except (EOFError, KeyboardInterrupt):
            return windows

        if choice == "1":
            return windows
        elif choice == "2":
            sel = cats["para"]
        elif choice == "3":
            sel = cats["mixed"]
        elif choice == "4":
            sel = cats["sym"]
        elif choice == "5":
            sel = cats["hv"]
        elif choice == "6":
            sel = _pick_individual_week(windows)
        else:
            print("  Invalid choice — try again.")
            time.sleep(1)
            continue

        if not sel:
            print("  No weeks in that category — try another.")
            time.sleep(1.5)
            continue

        return sel


def _pick_individual_week(windows: list[dict]) -> list[dict]:
    """List all weeks and let the user pick one by number."""
    _clear()
    print(DIVIDER)
    print(f"  Individual week picker ({len(windows)} weeks available)")
    print(DIVIDER)
    for i, w in enumerate(windows):
        std = (statistics.stdev(w["hrv_values"])
               if len(w.get("hrv_values", [])) > 1 else 0)
        tag = ""
        if w["mean_rmssd"] >= 95:
            tag = "[↑ para]"
        elif w["mean_rmssd"] < 75:
            tag = "[↓ sym]"
        if std >= 12:
            tag += "[hv]"
        print(f"  [{i + 1:2d}] {w['date_start']} → {w['date_end']}  "
              f"mean={w['mean_rmssd']:.0f} ms  std={std:.1f}  {tag}")
    print()
    print(f"  Enter week number [1-{len(windows)}]: ", end="", flush=True)

    try:
        n = int(input().strip()) - 1
        if 0 <= n < len(windows):
            return [windows[n]]
    except (ValueError, EOFError, KeyboardInterrupt):
        pass
    return windows


# ═══════════════════════════════════════════════════════
# WINDOW PLAYBACK
# ═══════════════════════════════════════════════════════

def play_iso_window(window: dict, window_idx: int, n_windows: int) -> None:
    """
    Play the full 7-day ISO arc for one weekly window as continuous audio.

    Day 1: Gemini (or fallback) maps the actual ANS state.
    Days 2–7: Linear interpolation toward PARA_TARGET.

    All 7 days are pre-generated and concatenated with a CROSSFADE_SEC crossfade
    so the transition between days is smooth rather than a hard cut.
    Each day's dashboard is rendered on screen while the combined audio plays.
    """
    n_days    = window["day_count"]
    day1_feat = dict(window)

    day1_music, ans, gemini_used = _get_day1_music(day1_feat)
    iso_arc = build_iso_arc(day1_music, n_days=n_days)

    # Pre-generate all days and concatenate with crossfade
    if AUDIO_ENABLED:
        chunks = []
        for day_idx, music in enumerate(iso_arc):
            arc_pos = day_idx / (n_days - 1) if n_days > 1 else 0.0
            chunks.append(generate_iso_audio(music, float(DAY_SEC),
                                             arc_position=arc_pos))
        combined = concat_iso_week(chunks, crossfade_seconds=CROSSFADE_SEC)
        sd.play(combined, samplerate=SAMPLE_RATE)

    # Dashboard advances one day at a time while audio plays continuously
    for day_idx, music in enumerate(iso_arc):
        render_day_dashboard(
            window_idx  = window_idx,
            n_windows   = n_windows,
            window      = window,
            day_idx     = day_idx,
            n_days      = n_days,
            music       = music,
            ans         = ans,
            gemini_used = gemini_used,
        )
        # Sleep for DAY_SEC; last day needs no crossfade deduction
        if day_idx < n_days - 1:
            time.sleep(max(1.0, DAY_SEC - CROSSFADE_SEC))
        else:
            time.sleep(DAY_SEC)

    if AUDIO_ENABLED:
        stop_iso_audio()


# ═══════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════

def run_iso_pipeline(quick: bool = False, sample: bool = False) -> None:
    """
    Load weekly windows and play the ISO arc for each, pausing between weeks
    to ask permission (unless quick=True, which auto-advances).

    sample=True: use the 3 bundled high-variance demo weeks.
    """
    if sample:
        if not SAMPLE_LOADER_AVAILABLE:
            print("ERROR: load_sample_weeks() not found in whoop_csv_loader.py.")
            sys.exit(1)
        all_windows = load_sample_weeks()
        if not all_windows:
            print("ERROR: sample_whoop_data.csv not found or empty.")
            sys.exit(1)
        mode_label = "Sample mode (3 bundled high-variance weeks)"
    else:
        all_windows = load_whoop_weekly_windows(window_size=7)
        if not all_windows:
            print("ERROR: No WHOOP data found. Place whoop_data.csv in project root.")
            sys.exit(1)
        mode_label = f"WHOOP history ({len(all_windows)} weeks)"

    # Week selection menu (skipped for --quick or --sample without --quick)
    if quick:
        windows = all_windows[:3]
    elif sample:
        # Show menu so user can filter even within sample weeks
        windows = select_weeks_menu(all_windows)
    else:
        windows = select_weeks_menu(all_windows)

    n_windows = len(windows)

    _clear()
    print(DIVIDER)
    print("  NTNU PROTOTYPE — ISO Arc Player")
    print("  ISO Principle: Mirror → Guide → Recover")
    print(DIVIDER)
    print(f"\n  Mode:      {mode_label}")
    print(f"  Weeks:     {n_windows} × 7-day arcs")
    print(f"  Per day:   {DAY_SEC}s audio  ({CROSSFADE_SEC:.0f}s crossfade between days)")
    total_sec = n_windows * 7 * DAY_SEC
    print(f"  Per week:  ~{7 * DAY_SEC}s  ({7 * DAY_SEC // 60}m {7 * DAY_SEC % 60}s)")
    print(f"  Total:     ~{total_sec // 60}m")
    print(f"  Audio:     {'3-layer ISO synthesis + sub-bass + chord voicing'if AUDIO_ENABLED else 'DISABLED — pip install sounddevice'}")
    if not quick:
        print(f"  Advance:   manual (Enter to continue each week)")
    else:
        print(f"  Advance:   automatic (--quick mode)")
    print(f"\n  Starting in 3 seconds...")
    time.sleep(3)

    for w_idx, window in enumerate(windows):
        try:
            play_iso_window(window, w_idx, n_windows)
        except KeyboardInterrupt:
            stop_iso_audio()
            print("\n\n  Stopped by user.")
            return
        except Exception as e:
            print(f"\n  Week {w_idx + 1} error: {e}")
            continue

        # After each week: pause and ask permission (unless quick mode)
        if w_idx < n_windows - 1:
            if quick:
                time.sleep(1)
            else:
                _clear()
                nxt = windows[w_idx + 1]
                print(DIVIDER)
                print(f"  Week {w_idx + 1}/{n_windows} complete.")
                print(f"  Next: Week {w_idx + 2}  "
                      f"({nxt['date_start']} → {nxt['date_end']})  "
                      f"mean RMSSD = {nxt['mean_rmssd']:.0f} ms")
                print(DIVIDER)
                try:
                    input(f"\n  Press Enter to continue, or Ctrl+C to stop... ")
                except (KeyboardInterrupt, EOFError):
                    stop_iso_audio()
                    print("\n  Stopped.")
                    return

    _clear()
    print(DIVIDER)
    print("  ISO Arc Pipeline Complete")
    print(DIVIDER)
    print(f"\n  Played {n_windows} weekly windows.")
    print(f"  Recovery target each week: {PARA_TARGET.tempo_bpm} BPM, "
          f"{PARA_TARGET.frequency_hz} Hz, theta binaural {PARA_TARGET.binaural_offset_hz} Hz")
    print(f"\n{DIVIDER}")


# ═══════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    quick  = "--quick"  in sys.argv
    sample = "--sample" in sys.argv
    run_iso_pipeline(quick=quick, sample=sample)
