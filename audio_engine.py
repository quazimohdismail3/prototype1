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
from dataclasses import dataclass

from gemini_mapper import MusicParameters

SAMPLE_RATE = 44100   # Hz — CD quality, compatible with all sounddevice backends

# Stores the last played parameters for smooth transitions.
previous_music: MusicParameters | None = None


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
# SMOOTH TRANSITIONS
# ═══════════════════════════════════════════════════════

@dataclass
class _MusicLike:
    tempo_bpm: float
    frequency_hz: float
    harmonic_complexity: float
    rhythmic_density: float
    dynamics: float
    binaural_offset_hz: float


def _generate_audio_core(
    music: _MusicLike,
    duration_seconds: float,
    sample_rate: int,
    apply_fade: bool,
) -> np.ndarray:
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0.0, duration_seconds, num_samples, endpoint=False)

    left = _synthesize(float(music.frequency_hz), float(music.harmonic_complexity), t)
    left = _apply_rhythm(left, float(music.rhythmic_density), float(music.tempo_bpm), t)

    if float(music.binaural_offset_hz) > 0.5:
        right_freq = float(music.frequency_hz) + float(music.binaural_offset_hz)
        right = _synthesize(right_freq, float(music.harmonic_complexity), t)
        right = _apply_rhythm(right, float(music.rhythmic_density), float(music.tempo_bpm), t)
    else:
        right = left.copy()

    amplitude = 0.05 + float(music.dynamics) * 0.40
    stereo = np.stack([left * amplitude, right * amplitude], axis=1).astype(np.float32)
    if apply_fade:
        stereo[:, 0] = _apply_fade(stereo[:, 0], sample_rate)
        stereo[:, 1] = _apply_fade(stereo[:, 1], sample_rate)
    return stereo


def _concat_with_crossfade(
    chunks: list[np.ndarray],
    sample_rate: int,
    crossfade_ms: float = 30.0,
) -> np.ndarray:
    if not chunks:
        return np.zeros((0, 2), dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]

    xf = int(sample_rate * crossfade_ms / 1000.0)
    xf = max(0, min(xf, min(c.shape[0] for c in chunks) // 4))
    if xf == 0:
        return np.concatenate(chunks, axis=0)

    out = chunks[0].astype(np.float32, copy=False)
    ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)[:, None]

    for nxt in chunks[1:]:
        nxt = nxt.astype(np.float32, copy=False)
        a = out[:-xf]
        b_tail = out[-xf:]
        n_head = nxt[:xf]
        n_rest = nxt[xf:]
        mixed = b_tail * (1.0 - ramp) + n_head * ramp
        out = np.concatenate([a, mixed, n_rest], axis=0)

    return out


def _music_like_from_params(m: MusicParameters) -> _MusicLike:
    return _MusicLike(
        tempo_bpm=float(m.tempo_bpm),
        frequency_hz=float(m.frequency_hz),
        harmonic_complexity=float(m.harmonic_complexity),
        rhythmic_density=float(m.rhythmic_density),
        dynamics=float(m.dynamics),
        binaural_offset_hz=float(m.binaural_offset_hz),
    )


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
    core = _music_like_from_params(music)
    return _generate_audio_core(core, duration_seconds, sample_rate, apply_fade=True)


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
    global previous_music

    target = _music_like_from_params(music)

    # Smooth transition: interpolate key parameters over ~2.5s (30 steps)
    transition_seconds = 2.5
    steps = 30

    if previous_music is None or duration_seconds <= 0:
        audio = _generate_audio_core(target, duration_seconds, sample_rate, apply_fade=True)
    else:
        prev = _music_like_from_params(previous_music)
        trans_dur = min(float(duration_seconds), float(transition_seconds))
        chunk_dur = trans_dur / float(steps)

        chunks: list[np.ndarray] = []
        for alpha in np.linspace(0.0, 1.0, steps):
            interp = _MusicLike(
                frequency_hz=prev.frequency_hz + (target.frequency_hz - prev.frequency_hz) * float(alpha),
                tempo_bpm=prev.tempo_bpm + (target.tempo_bpm - prev.tempo_bpm) * float(alpha),
                dynamics=prev.dynamics + (target.dynamics - prev.dynamics) * float(alpha),
                binaural_offset_hz=prev.binaural_offset_hz + (target.binaural_offset_hz - prev.binaural_offset_hz) * float(alpha),
                harmonic_complexity=target.harmonic_complexity,
                rhythmic_density=target.rhythmic_density,
            )
            chunks.append(_generate_audio_core(interp, chunk_dur, sample_rate, apply_fade=False))

        transition_audio = _concat_with_crossfade(chunks, sample_rate=sample_rate, crossfade_ms=30.0)

        remaining = max(0.0, float(duration_seconds) - trans_dur)
        if remaining > 0.0:
            steady = _generate_audio_core(target, remaining, sample_rate, apply_fade=False)
            audio = np.concatenate([transition_audio, steady], axis=0)
        else:
            audio = transition_audio

        # Fade the full buffer once (avoid per-chunk fade dips)
        audio[:, 0] = _apply_fade(audio[:, 0], sample_rate)
        audio[:, 1] = _apply_fade(audio[:, 1], sample_rate)

    sd.play(audio, samplerate=sample_rate)
    if blocking:
        sd.wait()

    previous_music = music


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
