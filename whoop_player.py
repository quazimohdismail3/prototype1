"""
WHOOP HRV History Player — Project Vagus
==========================================
Author: Quazi Mohd Ismail, MBBS MD
Project: Project Vagus / NTNU Prototype

Interactive terminal player for WHOOP HRV history dataset.
Each day's physiology drives real-time music synthesis via Gemini + audio engine.

Data:    whoop_data.csv (WHOOP export)
Cache:   .whoop_player_cache.json (Gemini responses, persisted between runs)
Audio:   sounddevice additive synthesis + binaural beats (audio_engine.py)

Run:  python whoop_player.py
Keys: N=next  B=back  P=pause  D=date  T=today  W=worst  H=best  Q=quit
"""

import os
import sys
import csv
import json
import time
from pathlib import Path

# ── Platform keyboard ─────────────────────────────────────────────────────────
# Windows: msvcrt (non-blocking, no raw-mode needed)
# Unix:    tty/termios (raw mode per keypress, restored immediately)
if sys.platform == "win32":
    import msvcrt

    def _getch() -> str:
        """Block until a printable key is pressed. Returns uppercase char."""
        while True:
            ch = msvcrt.getch()
            if ch in (b"\x00", b"\xe0"):   # extended key (arrows etc) — skip
                msvcrt.getch()
                continue
            if ch == b"\x03":              # Ctrl-C
                raise KeyboardInterrupt
            decoded = ch.decode("utf-8", errors="ignore")
            if decoded.strip():            # skip CR / whitespace
                return decoded.upper()
else:
    import tty
    import termios

    def _getch() -> str:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch.upper()

# ── Pipeline imports ──────────────────────────────────────────────────────────
from hrv_engine import compute_state_space, classify_ans_state
from gemini_mapper import map_to_music, MusicParameters

try:
    from audio_engine import play_audio, stop_audio
    AUDIO_ENABLED = True
except ImportError:
    AUDIO_ENABLED = False
    def play_audio(*a, **kw): pass   # noqa
    def stop_audio(): pass           # noqa

# ── Paths & constants ─────────────────────────────────────────────────────────
CSV_PATH    = Path(__file__).parent / "whoop_data.csv"
CACHE_PATH  = Path(__file__).parent / ".whoop_player_cache.json"

W           = 64          # dashboard character width
BAR_W       = 28          # progress bar width
DIVIDER     = "═" * W
THIN        = "─" * W
SPARK_CHARS = "▁▂▃▄▅▆▇"
AUDIO_DUR   = 90.0        # seconds of audio per day (user navigates before it ends)

# ── Mutable audio state ───────────────────────────────────────────────────────
_audio_paused  = False
_current_music = None     # most recently played music dict


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_data() -> list[dict]:
    """
    Parse whoop_data.csv and return all valid cycles in chronological order.

    The CSV is newest-first; we reverse so Day 1 = oldest entry.
    Each cycle dict:
        date, rmssd, sdnn, pnn50, sd2_sd1  — HRV features (pipeline-compatible)
        rhr_bpm, recovery_pct              — display-only fields (may be None)
    """
    rows = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hrv_raw = row.get("Heart rate variability (ms)", "").strip()
            if not hrv_raw:
                continue
            try:
                rmssd = float(hrv_raw)
            except ValueError:
                continue

            rhr_raw  = row.get("Resting heart rate (bpm)", "").strip()
            rec_raw  = row.get("Recovery score %", "").strip()
            date     = row.get("Cycle start time", "").strip()

            # Validated approximations from whoop_csv_loader.py
            sdnn    = round(rmssd * 1.2, 2)
            pnn50   = round(max(0.0, (rmssd - 20.0) / 1.5), 2)
            sd2_sd1 = round(max(0.5, 2.0 - (rmssd / 80.0)), 3)

            rows.append({
                "date":         date,
                "rmssd":        round(rmssd, 2),
                "sdnn":         sdnn,
                "pnn50":        pnn50,
                "sd2_sd1":      sd2_sd1,
                "rhr_bpm":      float(rhr_raw) if rhr_raw else None,
                "recovery_pct": float(rec_raw)  if rec_raw  else None,
            })

    rows.reverse()   # CSV is newest-first → chronological
    return rows


def precompute_ans(cycles: list[dict]) -> None:
    """
    Compute state_space and ans dicts for every cycle in-place.
    Pure math (no API calls) — runs at startup.
    """
    for c in cycles:
        ss  = compute_state_space(c)
        ans = classify_ans_state(ss["autonomic_score"])
        c["state_space"] = ss
        c["ans"]         = ans


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI CACHE
# ═══════════════════════════════════════════════════════════════════════════════

def load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


def get_music(cycle: dict, cache: dict) -> dict | None:
    """
    Return music parameters for this cycle.
    Hits Gemini API only on first encounter; caches result by date (YYYY-MM-DD).
    Returns parameter dict or None on API failure.
    """
    key = cycle["date"][:10]
    if key in cache:
        return cache[key]
    try:
        obj   = map_to_music(cycle, cycle["state_space"], cycle["ans"])
        music = obj.model_dump()
        cache[key] = music
        save_cache(cache)
        return music
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def bar(value: float, width: int = BAR_W) -> str:
    """Float 0–1 → visual progress bar."""
    value  = max(0.0, min(1.0, value))
    filled = int(value * width)
    empty  = width - filled
    return f"[{'█' * filled}{'░' * empty}] {int(value * 100):3d}%"


def sparkline(values: list[float]) -> str:
    """Map a list of HRV values to ▁▂▃▄▅▆▇ sparkline."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span   = max(hi - lo, 1.0)
    return "".join(SPARK_CHARS[min(6, int((v - lo) / span * 7))] for v in values)


def binaural_label(hz: float) -> str:
    if hz <= 0.5: return "off"
    if hz < 4:    return "delta"
    if hz < 8:    return "theta"
    if hz < 14:   return "alpha"
    return "beta"


def state_icon(label: str) -> str:
    if "PARASYMPATHETIC" in label: return "◉ PARASYMPATHETIC DOMINANT"
    if "SYMPATHETIC"     in label: return "◈ SYMPATHETIC DOMINANT"
    return "◎ MIXED AUTONOMIC STATE"


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD RENDERERS
# ═══════════════════════════════════════════════════════════════════════════════

def render_mapping_screen(day: int, total: int, date_str: str) -> None:
    """Shown while waiting for the Gemini API call to complete."""
    clear()
    print(DIVIDER)
    print(f"  WHOOP HRV HISTORY PLAYER — Project Vagus")
    print(DIVIDER)
    print(f"  Day {day}/{total}   ·   {date_str}")
    print(THIN)
    print(f"  Mapping...   calling gemini-2.5-flash")
    print(f"  Please wait — this is cached after the first call.")
    print(THIN)
    print(DIVIDER)


def render_dashboard(
    day:          int,
    total:        int,
    cycle:        dict,
    music:        dict | None,
    hrv_win:      list[float],
    audio_status: str,
    note:         str = "",
) -> None:
    """Full terminal dashboard for one WHOOP day."""
    clear()
    ss  = cycle["state_space"]
    ans = cycle["ans"]

    rec   = cycle.get("recovery_pct")
    rhr   = cycle.get("rhr_bpm")
    rec_s = f"{rec:.0f}%" if rec is not None else "N/A"
    rhr_s = f"{rhr:.0f} bpm" if rhr is not None else "N/A"

    # ── Header ────────────────────────────────────────────
    print(DIVIDER)
    print(f"  WHOOP HRV HISTORY PLAYER — Project Vagus")
    print(DIVIDER)
    print(f"  Day {day}/{total}   ·   {cycle['date'][:10]}   ·   {time.strftime('%H:%M:%S')}")
    print(THIN)

    # ── HRV vitals ────────────────────────────────────────
    print(f"  HRV  {cycle['rmssd']:>6.1f} ms    Recovery  {rec_s:>5}    RHR  {rhr_s}")
    print(THIN)

    # ── 7-Day sparkline ───────────────────────────────────
    spark = sparkline(hrv_win)
    lo_s  = f"{min(hrv_win):.0f}" if hrv_win else "–"
    hi_s  = f"{max(hrv_win):.0f}" if hrv_win else "–"
    n_days = len(hrv_win)
    print(f"  {n_days}-Day HRV   {spark}   ({lo_s}–{hi_s} ms)")
    print(THIN)

    # ── ANS state ─────────────────────────────────────────
    print(f"  ANS   {state_icon(ans['state'])}")
    print(f"  Score {bar(ss['autonomic_score'])}   autonomic composite")
    print(THIN)

    # ── Music parameters ──────────────────────────────────
    if music:
        tempo_n = (music["tempo_bpm"] - 40) / 80
        freq_n  = (music["frequency_hz"] - 110) / 330
        bin_n   = music["binaural_offset_hz"] / 14.0
        bin_lbl = binaural_label(music["binaural_offset_hz"])

        print(f"  ◆ MUSIC PARAMETERS   (gemini-2.5-flash · cached)")
        print(f"  Tempo     {bar(tempo_n)}  {music['tempo_bpm']} BPM")
        print(f"  Frequency {bar(freq_n)}  {music['frequency_hz']} Hz")
        print(f"  Binaural  {bar(bin_n)}  {music['binaural_offset_hz']:.1f} Hz ({bin_lbl})")
    else:
        print(f"  ◆ MUSIC PARAMETERS")
        print(f"  {note if note else 'Mapping unavailable — check API key / network'}")
        print()

    print(THIN)

    # ── Audio status ──────────────────────────────────────
    if AUDIO_ENABLED and music:
        rf = music["frequency_hz"] + music["binaural_offset_hz"]
        print(f"  ♪ {audio_status:<8}  "
              f"{music['frequency_hz']} Hz (L)  /  {rf:.1f} Hz (R)   "
              f"{music['tempo_bpm']} BPM")
    elif not AUDIO_ENABLED:
        print(f"  ♪ AUDIO DISABLED   (pip install sounddevice)")
    else:
        print(f"  ♪ {audio_status}")

    print(THIN)
    print(f"  N=next  B=back  P=pause  D=date  T=today  W=worst  H=best  Q=quit")
    print(DIVIDER)


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _start_audio(music: dict) -> None:
    """Stop any current audio and start new audio for this day's parameters."""
    global _audio_paused, _current_music
    _current_music = music
    _audio_paused  = False
    if not AUDIO_ENABLED:
        return
    stop_audio()
    time.sleep(0.05)   # small gap for clean transition (50ms fade-in follows)
    play_audio(MusicParameters(**music), duration_seconds=AUDIO_DUR, blocking=False)


def _toggle_pause() -> str:
    """Toggle play/pause. Returns new status string."""
    global _audio_paused, _current_music
    if not AUDIO_ENABLED:
        return "DISABLED"
    if _audio_paused:
        if _current_music:
            stop_audio()
            play_audio(
                MusicParameters(**_current_music),
                duration_seconds=AUDIO_DUR,
                blocking=False,
            )
        _audio_paused = False
        return "PLAYING"
    else:
        stop_audio()
        _audio_paused = True
        return "PAUSED"


def _audio_status() -> str:
    if not AUDIO_ENABLED:
        return "DISABLED"
    return "PAUSED" if _audio_paused else "PLAYING"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTERACTIVE LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    # ── Startup splash ────────────────────────────────────
    clear()
    print(DIVIDER)
    print(f"  WHOOP HRV HISTORY PLAYER — Project Vagus")
    print(DIVIDER)
    print(f"\n  Use headphones for binaural beat effect.")
    print(f"\n  Loading WHOOP dataset...")

    cycles = load_all_data()
    if not cycles:
        print("  ERROR: No valid data found in whoop_data.csv")
        sys.exit(1)

    total = len(cycles)
    print(f"  {total} cycles loaded.")
    print(f"  Computing ANS state spaces (no API)...")
    precompute_ans(cycles)
    print(f"  Loading Gemini response cache...")
    cache = load_cache()
    print(f"  {len(cache)} days pre-mapped in cache.")
    print(f"\n  Starting in 2 seconds...")
    time.sleep(2)

    current_idx = total - 1   # default: most recent day

    # ── Helper: build 7-day HRV window ending at idx ──────
    def hrv_window(idx: int) -> list[float]:
        start = max(0, idx - 6)
        return [cycles[i]["rmssd"] for i in range(start, idx + 1)]

    # ── Navigate to a day, trigger mapping + audio ─────────
    def navigate(new_idx: int) -> None:
        nonlocal current_idx
        current_idx = max(0, min(new_idx, total - 1))
        cycle       = cycles[current_idx]
        day         = current_idx + 1
        date_key    = cycle["date"][:10]

        # Gemini mapping (cached after first call)
        if date_key not in cache:
            render_mapping_screen(day, total, date_key)
            music = get_music(cycle, cache)
        else:
            music = cache[date_key]

        # Audio: start for new day (respects pause state)
        if music and not _audio_paused:
            _start_audio(music)
        elif not music:
            stop_audio()

        render_dashboard(
            day, total, cycle, music,
            hrv_window(current_idx), _audio_status()
        )

    # ── Initial render ────────────────────────────────────
    navigate(current_idx)

    # ── Event loop ────────────────────────────────────────
    while True:
        try:
            key = _getch()
        except KeyboardInterrupt:
            key = "Q"

        if key == "Q":
            stop_audio()
            clear()
            print(DIVIDER)
            print(f"  WHOOP HRV HISTORY PLAYER — Project Vagus")
            print(DIVIDER)
            print(f"\n  Session ended at Day {current_idx + 1}/{total}.")
            print(f"  {len(cache)} days mapped and cached.")
            print(f"\n{DIVIDER}")
            break

        elif key == "N":
            if current_idx < total - 1:
                navigate(current_idx + 1)

        elif key == "B":
            if current_idx > 0:
                navigate(current_idx - 1)

        elif key == "P":
            status = _toggle_pause()
            cycle  = cycles[current_idx]
            day    = current_idx + 1
            music  = cache.get(cycle["date"][:10])
            render_dashboard(
                day, total, cycle, music,
                hrv_window(current_idx), status
            )

        elif key == "T":
            navigate(total - 1)

        elif key == "W":
            worst = min(range(total), key=lambda i: cycles[i]["rmssd"])
            navigate(worst)

        elif key == "H":   # H = Highest (best) HRV — B is taken by "back"
            best = max(range(total), key=lambda i: cycles[i]["rmssd"])
            navigate(best)

        elif key == "D":
            # Date jump: show dashboard + inline prompt
            cycle  = cycles[current_idx]
            day    = current_idx + 1
            music  = cache.get(cycle["date"][:10])
            render_dashboard(
                day, total, cycle, music,
                hrv_window(current_idx), _audio_status()
            )
            print(f"\n  Enter date (YYYY-MM-DD): ", end="", flush=True)
            try:
                date_input = input().strip()
            except (EOFError, KeyboardInterrupt):
                date_input = ""

            if date_input:
                found = next(
                    (i for i, c in enumerate(cycles) if c["date"][:10] == date_input),
                    None,
                )
                if found is not None:
                    navigate(found)
                else:
                    render_dashboard(
                        day, total, cycle, music,
                        hrv_window(current_idx), _audio_status(),
                        note=f"Date '{date_input}' not found in dataset",
                    )


if __name__ == "__main__":
    main()
