"""
NTNU PROTOTYPE — ISO Audio Engine
====================================
Author: Quazi Mohd Ismail, MBBS MD

Three-layer psychoacoustic synthesiser for ISO Principle arc playback.
Used exclusively by iso_player.py — audio_engine.py is unchanged.

Synthesis layers:
  Layer 1 — Drone        : root tone + harmonics + binaural beat
  Layer 2 — Harmony      : consonant interval above root, gliding major
                           third → perfect fifth as arc_position 0→1
  Layer 3 — Rhythmic Pulse: bar-structured Gaussian beat accents at
                            octave-above-root, scaled by rhythmic_density

Musical structure:
  4/4 time: bar = 4 beats, phrase = 4 bars
  Phrase envelope: smooth raised-cosine arch over each 4-bar phrase
                   (floor 0.3 — never fully silent)

Psychoacoustic grounding:
  Perfect fifth (3:2) — most consonant, associated with rest/parasympathetic
  Major third (5:4)   — warm but slightly tense, mirrors sympathetic onset
  Binaural beats       — theta 4–8 Hz for parasympathetic (McConnell 2014)
  Phrase envelope      — mirrors natural respiratory cycle pacing (Bernardi 2006)
"""

import numpy as np
import sounddevice as sd

from gemini_mapper import MusicParameters

SAMPLE_RATE   = 44100   # Hz
CROSSFADE_SEC = 2.0     # seconds of crossfade between days in a week arc

# Psychoacoustic just-intonation interval ratios
_RATIO_MATCH   = 5 / 4   # major third  — arc_position = 0.0 (mirroring stress)
_RATIO_TARGET  = 3 / 2   # perfect fifth — arc_position = 1.0 (parasympathetic)


# ═══════════════════════════════════════════════════════
# INTERNAL PRIMITIVES
# ═══════════════════════════════════════════════════════

def _phrase_envelope(t: np.ndarray, tempo_bpm: float) -> np.ndarray:
    """
    Raised-cosine arch over each 4-bar phrase.

    Gives the sound a natural "breathing" shape — swells gently in the
    middle of each phrase and recedes at the boundaries.

    Floor at 0.3: never fully silent, just quieter at phrase seams.
    """
    beat_sec   = 60.0 / max(tempo_bpm, 20.0)
    phrase_sec = beat_sec * 16                        # 4 beats × 4 bars
    t_in       = t % phrase_sec
    arch       = 0.5 * (1.0 - np.cos(2.0 * np.pi * t_in / phrase_sec))
    return 0.3 + 0.7 * arch                          # [0.3, 1.0]


def _beat_envelope(t: np.ndarray, tempo_bpm: float,
                   rhythmic_density: float) -> np.ndarray:
    """
    Bar-structured Gaussian pulses at beat positions.

    Beat 1 of each bar (downbeat) is 1.5× stronger — gives a sense of
    musical metre rather than an undifferentiated pulse stream.

    Width of each pulse ≈ 15% of one beat duration.
    """
    if rhythmic_density < 0.15:
        return np.zeros_like(t)

    beat_sec = 60.0 / max(tempo_bpm, 20.0)
    bar_sec  = beat_sec * 4
    sigma    = beat_sec * 0.15

    env = np.zeros_like(t)
    duration = float(t[-1]) + beat_sec            # slightly past end
    beat_positions = np.arange(0.0, duration, beat_sec)

    for i, bp in enumerate(beat_positions):
        strength = 1.5 if (i % 4 == 0) else 1.0  # downbeat emphasis
        env += strength * np.exp(-0.5 * ((t - bp) / sigma) ** 2)

    # Normalise peak to rhythmic_density
    peak = env.max()
    if peak > 0:
        env = env / peak * rhythmic_density
    return env


def _apply_fade(stereo: np.ndarray, fade_ms: float = 50.0) -> np.ndarray:
    """50 ms linear fade-in / fade-out on both channels."""
    n = min(int(fade_ms * SAMPLE_RATE / 1000), stereo.shape[0] // 4)
    if n < 2:
        return stereo
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    stereo[:n,  0] *= ramp
    stereo[:n,  1] *= ramp
    stereo[-n:, 0] *= ramp[::-1]
    stereo[-n:, 1] *= ramp[::-1]
    return stereo


# ═══════════════════════════════════════════════════════
# CROSSFADE UTILITY
# ═══════════════════════════════════════════════════════

def concat_iso_week(
    chunks: list,
    crossfade_seconds: float = CROSSFADE_SEC,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Concatenate per-day audio chunks with a linear crossfade between each pair.

    Instead of hard-cutting between days, the end of day N fades out while
    day N+1 fades in simultaneously over `crossfade_seconds`.

    Args:
        chunks:            List of stereo float32 arrays (one per day).
        crossfade_seconds: Duration of overlap between adjacent days.
        sample_rate:       Audio sample rate (must match how chunks were generated).

    Returns:
        Single stereo float32 array covering the full week with crossfades.
    """
    if not chunks:
        return np.zeros((0, 2), dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0].astype(np.float32)

    xf   = int(sample_rate * crossfade_seconds)
    ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)[:, None]  # (xf, 1) for broadcasting
    out  = chunks[0].astype(np.float32, copy=False)

    for nxt in chunks[1:]:
        nxt   = nxt.astype(np.float32, copy=False)
        xf_n  = min(xf, out.shape[0] // 4, nxt.shape[0] // 4)
        if xf_n < 2:
            out = np.concatenate([out, nxt], axis=0)
            continue
        r = np.linspace(0.0, 1.0, xf_n, dtype=np.float32)[:, None]
        mixed = out[-xf_n:] * (1.0 - r) + nxt[:xf_n] * r
        out = np.concatenate([out[:-xf_n], mixed, nxt[xf_n:]], axis=0)

    return out


# ═══════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════

def generate_iso_audio(
    music:            MusicParameters,
    duration_seconds: float,
    arc_position:     float = 0.0,
    sample_rate:      int   = SAMPLE_RATE,
) -> np.ndarray:
    """
    Generate a stereo float32 buffer using three psychoacoustic layers.

    Args:
        music:            MusicParameters for this day of the ISO arc
        duration_seconds: Length of output buffer in seconds
        arc_position:     0.0 = Day 1 ISO match, 1.0 = Day 7 para target
                          Controls harmony interval: major third → perfect fifth
        sample_rate:      Audio sample rate (default 44100 Hz)

    Returns:
        ndarray shape (n_samples, 2), dtype float32, clipped to [-1, 1]
    """
    arc_position = float(np.clip(arc_position, 0.0, 1.0))
    n  = int(duration_seconds * sample_rate)
    t  = np.linspace(0.0, duration_seconds, n, endpoint=False)

    # Master amplitude envelope
    amp          = 0.05 + float(music.dynamics) * 0.35
    phrase_env   = _phrase_envelope(t, float(music.tempo_bpm))

    freq  = float(music.frequency_hz)
    hc    = float(music.harmonic_complexity)

    # Slow breathing LFO shared across layers (quarter of tempo rate)
    breath_hz  = float(music.tempo_bpm) / 240.0
    breath_lfo = 0.7 + 0.3 * np.cos(2.0 * np.pi * breath_hz * t)

    # ── Layer 0: Sub-bass ──────────────────────────────────────
    # One octave below root — adds warmth and grounding.
    # Pulses at half the breath rate so it feels like a slow foundation.
    sub_freq   = freq / 2.0
    sub_breath = 0.55 + 0.45 * np.cos(2.0 * np.pi * breath_hz * 0.5 * t)
    sub        = np.sin(2.0 * np.pi * sub_freq * t) * sub_breath * phrase_env
    sub_amp    = amp * 0.15

    # ── Layer 1: Drone ─────────────────────────────────────────
    # 1–4 harmonics with 1/n natural rolloff; binaural on fundamental
    n_harmonics = max(1, round(1 + hc * 3))           # 1–4 partials
    drone_l = np.zeros(n)
    drone_r = np.zeros(n)
    total_w = 0.0

    for h in range(1, n_harmonics + 1):
        w = 1.0 / h
        drone_l += w * np.sin(2.0 * np.pi * freq * h * t)
        # Binaural offset only on the fundamental (h=1)
        r_freq = freq + float(music.binaural_offset_hz) if h == 1 else freq * h
        drone_r += w * np.sin(2.0 * np.pi * r_freq * t)
        total_w += w

    drone_l = (drone_l / total_w) * phrase_env
    drone_r = (drone_r / total_w) * phrase_env
    drone_amp = amp * 0.42

    # ── Layer 2: Harmony (chord voicing) ───────────────────────
    # Root interval glides major third → perfect fifth (arc 0→1).
    # A second voice adds a fifth above the root interval, creating a
    # full triad and making the harmony sound like actual chords.
    ratio      = _RATIO_MATCH + arc_position * (_RATIO_TARGET - _RATIO_MATCH)
    harm_freq  = freq * ratio
    fifth_freq = harm_freq * 1.5                       # fifth above harmony

    harm_root  = np.sin(2.0 * np.pi * harm_freq  * t) * breath_lfo * phrase_env
    harm_fifth = np.sin(2.0 * np.pi * fifth_freq * t) * breath_lfo * phrase_env * 0.6

    harm_amp = amp * 0.30
    # Stereo spread: root slightly left, fifth slightly right
    harm_l = harm_root * 1.10 + harm_fifth * 0.90
    harm_r = harm_root * 0.90 + harm_fifth * 1.10

    # ── Layer 3: Rhythmic Pulse ────────────────────────────────
    # Downbeats (beat 1) sound at the octave above root.
    # Off-beats (beats 2/4) sound at the fifth above root.
    # Mixed together, this gives a simple bass-line feel instead of
    # an undifferentiated pulse stream.
    beat_env   = _beat_envelope(t, float(music.tempo_bpm),
                                float(music.rhythmic_density))
    pulse_tone = (0.65 * np.sin(2.0 * np.pi * freq * 2.0 * t) +   # octave
                  0.35 * np.sin(2.0 * np.pi * freq * 1.5 * t))     # fifth
    pulse      = pulse_tone * beat_env
    pulse_amp  = amp * 0.13

    # ── Mix ────────────────────────────────────────────────────
    left  = (sub_amp * sub + drone_amp * drone_l
             + harm_amp * harm_l + pulse_amp * pulse)
    right = (sub_amp * sub + drone_amp * drone_r
             + harm_amp * harm_r + pulse_amp * pulse * 0.6)

    stereo = np.column_stack([left, right]).astype(np.float32)
    stereo = _apply_fade(stereo)
    return np.clip(stereo, -1.0, 1.0)


def play_iso_audio(
    music:            MusicParameters,
    duration_seconds: float,
    arc_position:     float = 0.0,
    blocking:         bool  = False,
    sample_rate:      int   = SAMPLE_RATE,
) -> None:
    """Generate and play ISO audio via sounddevice."""
    audio = generate_iso_audio(music, duration_seconds, arc_position, sample_rate)
    sd.play(audio, samplerate=sample_rate)
    if blocking:
        sd.wait()


def stop_iso_audio() -> None:
    """Immediately stop any active sounddevice playback."""
    sd.stop()


# ═══════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("═" * 58)
    print("  ISO Audio Engine — Layer verification")
    print("  3 arc positions × 4s each  (use headphones)")
    print("═" * 58)

    tests = [
        (0.0, MusicParameters(tempo_bpm=100, frequency_hz=360,
            harmonic_complexity=0.7, rhythmic_density=0.65,
            dynamics=0.70, binaural_offset_hz=10.0,
            mapping_rationale="ISO match: sympathetic dominant")),
        (0.5, MusicParameters(tempo_bpm=75,  frequency_hz=244,
            harmonic_complexity=0.4, rhythmic_density=0.35,
            dynamics=0.48, binaural_offset_hz=8.0,
            mapping_rationale="ISO mid-arc: transitioning")),
        (1.0, MusicParameters(tempo_bpm=50,  frequency_hz=128,
            harmonic_complexity=0.1, rhythmic_density=0.05,
            dynamics=0.25, binaural_offset_hz=6.0,
            mapping_rationale="ISO target: parasympathetic")),
    ]

    for arc_pos, music in tests:
        interval_ratio = _RATIO_MATCH + arc_pos * (_RATIO_TARGET - _RATIO_MATCH)
        print(f"\n  arc={arc_pos:.1f}  tempo={music.tempo_bpm}  "
              f"freq={music.frequency_hz}Hz  interval={interval_ratio:.3f}")
        audio = generate_iso_audio(music, 4.0, arc_position=arc_pos)
        print(f"  Buffer: {audio.shape}  max={audio.max():.3f}  "
              f"min={audio.min():.3f}", end="  ")
        assert audio.dtype == np.float32,  "dtype must be float32"
        assert audio.max()  <= 1.0,        "clip failed — too loud"
        assert audio.shape[1] == 2,        "must be stereo"
        print("OK")
        sd.play(audio, samplerate=SAMPLE_RATE)
        sd.wait()

    print("\n  All layers verified.")
    print("═" * 58)
