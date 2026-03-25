"""
NTNU PROTOTYPE — Audio Engine v1
==================================
Author: Quazi Mohd Ismail, MBBS MD
Project: Project Vagus / NTNU Prototype

Generates real-time therapeutic audio from MusicParameters using
additive synthesis and binaural beats via sounddevice.

Audio Model:
  Additive synthesis   → harmonic_complexity controls overtone richness
  Binaural beats       → binaural_offset_hz splits L/R carrier frequencies
  Amplitude modulation → rhythmic_density pulses at tempo_bpm rate
  Dynamics             → scales overall amplitude (headroom preserved)

Scientific Basis:
  Binaural beats:    McConnell et al. 2014 (theta 4-8 Hz, alpha 8-14 Hz)
  Harmonic richness: Thayer & Lane 2000 (pure sine = calm, overtones = arousal)
  Tempo entrainment: Bernardi 2006 (slow tempo → HRV coherence at 0.1 Hz)
  432 Hz base:       Calamassi & Pomponi 2019 (HRV coupling study)
"""

import sys
import time
import numpy as np
import sounddevice as sd

from gemini_mapper import MusicParameters

SAMPLE_RATE = 44100   # Hz — CD quality, compatible with all sounddevice backends


# ═══════════════════════════════════════════════════════
# SYNTHESIS PRIMITIVES
# ═══════════════════════════════════════════════════════

def _synthesize(freq_hz: float, harmonic_complexity: float, t: np.ndarray) -> np.ndarray:
    """
    Additive synthesis from harmonic series.

    harmonic_complexity 0.0 → 1 partial (pure sine)
    harmonic_complexity 1.0 → 8 partials with 1/n amplitude rolloff

    All partials summed and peak-normalised to [-1, 1].
    """
    num_harmonics = 1 + round(harmonic_complexity * 7)   # 1 – 8
    signal    = np.zeros(len(t))
    total_amp = 0.0
    for n in range(1, num_harmonics + 1):
        amp = 1.0 / n                                     # natural 1/n rolloff
        signal    += amp * np.sin(2.0 * np.pi * freq_hz * n * t)
        total_amp += amp
    return signal / total_amp                             # normalise


def _apply_rhythm(
    signal:          np.ndarray,
    rhythmic_density: float,
    tempo_bpm:        float,
    t:               np.ndarray,
) -> np.ndarray:
    """
    Cosine LFO amplitude modulation at tempo rate.

    rhythmic_density 0.0 → sustained, no modulation
    rhythmic_density 1.0 → 75 % depth pulse at tempo_bpm
    """
    if rhythmic_density < 0.05:
        return signal
    pulse_hz = tempo_bpm / 60.0
    depth    = rhythmic_density * 0.75                    # max 75 % mod depth
    lfo      = 1.0 - depth * 0.5 * (1.0 - np.cos(2.0 * np.pi * pulse_hz * t))
    return signal * lfo


def _apply_fade(signal: np.ndarray, sample_rate: int, fade_ms: float = 50.0) -> np.ndarray:
    """Linear 50 ms fade-in / fade-out — prevents audible clicks."""
    n = min(int(fade_ms * sample_rate / 1000), len(signal) // 4)
    ramp           = np.linspace(0.0, 1.0, n)
    signal[:n]    *= ramp
    signal[-n:]   *= ramp[::-1]
    return signal


# ═══════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════

def generate_audio(
    music:            MusicParameters,
    duration_seconds: float,
    sample_rate:      int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Generate a stereo float32 audio array from MusicParameters.

    Left channel:  carrier at frequency_hz
    Right channel: carrier at frequency_hz + binaural_offset_hz
                   (identical to left when binaural_offset_hz <= 0.5 Hz)

    Returns ndarray of shape (num_samples, 2), dtype float32.
    """
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0.0, duration_seconds, num_samples, endpoint=False)

    # ── Left channel ────────────────────────────────────
    left = _synthesize(music.frequency_hz, music.harmonic_complexity, t)
    left = _apply_rhythm(left, music.rhythmic_density, music.tempo_bpm, t)

    # ── Right channel — binaural offset if present ──────
    if music.binaural_offset_hz > 0.5:
        right_freq = music.frequency_hz + music.binaural_offset_hz
        right = _synthesize(right_freq, music.harmonic_complexity, t)
        right = _apply_rhythm(right, music.rhythmic_density, music.tempo_bpm, t)
    else:
        right = left.copy()

    # ── Amplitude — dynamics 0→1 maps to 0.05→0.45 ─────
    # Capped at 0.45 to preserve headroom and avoid clipping
    amplitude = 0.05 + music.dynamics * 0.40
    left  = _apply_fade(left  * amplitude, sample_rate)
    right = _apply_fade(right * amplitude, sample_rate)

    return np.stack([left, right], axis=1).astype(np.float32)


def play_audio(
    music:            MusicParameters,
    duration_seconds: float,
    blocking:         bool = False,
    sample_rate:      int  = SAMPLE_RATE,
) -> None:
    """
    Play audio derived from MusicParameters via sounddevice.

    Args:
        music:            Validated MusicParameters from gemini_mapper
        duration_seconds: Playback duration in seconds
        blocking:         Block until playback completes (default False)
        sample_rate:      Audio sample rate (default 44100 Hz)
    """
    audio = generate_audio(music, duration_seconds, sample_rate)
    sd.play(audio, samplerate=sample_rate)
    if blocking:
        sd.wait()


def stop_audio() -> None:
    """Immediately stop any active sounddevice playback."""
    sd.stop()


# ═══════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":

    sys.stdout.reconfigure(encoding="utf-8")
    DEMO_SECS = 6.0   # seconds per ANS state demo

    print("═" * 58)
    print("  NTNU PROTOTYPE — Audio Engine v1")
    print("  Additive synthesis + binaural beats via sounddevice")
    print(f"  Sample rate: {SAMPLE_RATE} Hz  |  {DEMO_SECS:.0f}s per state")
    print("═" * 58)
    print("\n  Use headphones for binaural beat effect.\n")

    # Representative parameters for the three ANS states
    tests = [
        {
            "label": "PARASYMPATHETIC DOMINANT",
            "params": MusicParameters(
                tempo_bpm          = 52,
                frequency_hz       = 110,
                harmonic_complexity= 0.1,
                rhythmic_density   = 0.1,
                dynamics           = 0.25,
                binaural_offset_hz = 6.0,
                mapping_rationale  = "High vagal tone: slow, low, pure sine, theta binaural."
            ),
        },
        {
            "label": "MIXED AUTONOMIC STATE",
            "params": MusicParameters(
                tempo_bpm          = 75,
                frequency_hz       = 220,
                harmonic_complexity= 0.4,
                rhythmic_density   = 0.4,
                dynamics           = 0.50,
                binaural_offset_hz = 9.0,
                mapping_rationale  = "Balanced ANS: moderate tempo, mid-frequency, alpha binaural."
            ),
        },
        {
            "label": "SYMPATHETIC DOMINANT",
            "params": MusicParameters(
                tempo_bpm          = 105,
                frequency_hz       = 380,
                harmonic_complexity= 0.7,
                rhythmic_density   = 0.7,
                dynamics           = 0.75,
                binaural_offset_hz = 12.0,
                mapping_rationale  = "Sympathetic load: fast, high-frequency, dense harmonics, beta binaural."
            ),
        },
    ]

    passed = 0
    failed = 0

    for i, test in enumerate(tests, 1):
        music = test["params"]
        right_freq = music.frequency_hz + music.binaural_offset_hz
        num_harmonics = 1 + round(music.harmonic_complexity * 7)

        print(f"── Test {i}/3: {test['label']}")
        print(f"   Tempo:     {music.tempo_bpm} BPM")
        print(f"   Frequency: L {music.frequency_hz} Hz  →  R {right_freq:.1f} Hz  "
              f"(Δ {music.binaural_offset_hz:.1f} Hz binaural)")
        print(f"   Harmonics: {num_harmonics} partials  "
              f"(complexity {music.harmonic_complexity:.1f})")
        print(f"   Rhythm:    depth {music.rhythmic_density:.1f}  at {music.tempo_bpm} BPM")
        print(f"   Dynamics:  {music.dynamics:.2f}  "
              f"(amplitude {0.05 + music.dynamics * 0.40:.2f})")
        print(f"   Playing {DEMO_SECS:.0f}s... ", end="", flush=True)

        try:
            play_audio(music, duration_seconds=DEMO_SECS, blocking=True)
            print("done.")
            passed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

        if i < len(tests):
            time.sleep(0.8)
        print()

    print("═" * 58)
    print(f"  Results: {passed}/3 passed, {failed}/3 failed")
    if failed == 0:
        print("  Audio Engine v1 ready for vagus_pipeline.py.")
    else:
        print("  Check sounddevice installation: pip install sounddevice")
    print("═" * 58)
