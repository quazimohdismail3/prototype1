# Project Vagus — ISO Principle Music Arc

A bioadaptive music system that reads HRV (Heart Rate Variability) data,
classifies your autonomic nervous system state, and generates therapeutic
music that gradually guides you from stress toward parasympathetic calm —
following the **ISO Principle** from music therapy.

Built as a research prototype for **Prof. Andreas Bergsland, NTNU**.

---

## What you will hear

Each **week** of data becomes a **7-day musical arc**:

| Day | Role | Music character |
|-----|------|----------------|
| Day 1 | **Mirror** — matches your actual ANS state | Mirrors stress: faster tempo, higher frequency, denser harmonics |
| Days 2–5 | **Guide** — gradual steering | Slowly decelerates, lowers pitch, simplifies texture |
| Day 7 | **Target** — parasympathetic calm | 50 BPM, 128 Hz root, theta binaural (6 Hz), pure tones |

Each day is **15 seconds** of audio, with a **2-second crossfade** into the next day so the transition is seamless. One full week arc plays in about **1 minute 45 seconds**.

The sound has four layers:
- **Sub-bass** — one octave below root, pulses slowly for grounding
- **Drone** — root tone with 1–4 harmonics + binaural beat between ears
- **Harmony** — a chord that glides from a major third (tension) toward a perfect fifth (rest) as the arc progresses
- **Rhythmic pulse** — gentle octave/fifth accents at the tempo, fading toward stillness

Use headphones for the binaural effect.

---

## The ISO Principle

The ISO Principle (Altshuler 1948) is a foundational music therapy technique:
start with music that **matches** the listener's current emotional/physiological state,
then **gradually** steer the music toward the desired target state.
The nervous system follows the music through entrainment rather than resisting an abrupt change.

---

## Quick start (3 minutes)

**1. Clone the repository**
```bash
git clone https://github.com/your-username/ntnu-prototype.git
cd ntnu-prototype
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

> Requires Python 3.10 or later. On Windows use `python` instead of `python3`.

**3. Get a Gemini API key** *(free tier is sufficient)*

Go to [Google AI Studio](https://aistudio.google.com/apikey) → Create API key → copy it.

**4. Create your `.env` file**
```bash
cp .env.example .env
# Open .env in any text editor and paste your key:
# GEMINI_API_KEY=your_key_here
```

**5. Run the demo**
```bash
python iso_player.py --sample
```

This plays **3 pre-bundled weeks** of real HRV data (high-variance weeks chosen for maximum musical contrast). A week selection menu appears — choose any category or play all three.

---

## Run modes

| Command | What it does |
|---------|-------------|
| `python iso_player.py --sample` | Demo: 3 curated high-variance weeks, selection menu |
| `python iso_player.py --sample --quick` | Demo, auto-advances (no permission prompts) |
| `python iso_player.py` | Your full WHOOP history, week selection menu |
| `python iso_player.py --quick` | First 3 weeks of your history, auto-advance |
| `python iso_audio.py` | Test the audio engine alone (3 arc positions, 4 s each) |
| `python vagus_pipeline.py` | Original pipeline with simulated HRV + terminal dashboard |
| `python vagus_pipeline.py --quick` | Quick test (3 cycles, 10 s intervals) |

**Between weeks** the player pauses and asks your permission before continuing (skip with `--quick`).

---

## Week selection menu

When you run `iso_player.py`, a menu groups weeks by autonomic state:

```
  [1] All weeks              (43 total)
  [2] Parasympathetic        (high HRV ≥ 95 ms,  N weeks)
  [3] Balanced / Mixed       (HRV 75–95 ms,      N weeks)
  [4] Sympathetic            (low HRV < 75 ms,   N weeks)
  [5] High variance          (std ≥ 12 ms,       N weeks)  ← best for demo
  [6] Choose individual week...
```

---

## Using your own WHOOP data

1. Open the WHOOP app → Profile → Export Data → download the CSV
2. Place the file as `whoop_data.csv` in the project root
3. Run `python iso_player.py`

The loader reads these columns (all others are ignored):
- `Heart rate variability (ms)` — RMSSD (required)
- `Resting heart rate (bpm)` — optional
- `Cycle start time` — date for navigation

---

## Architecture

```
WHOOP CSV (or simulated HRV)
    → whoop_csv_loader.py   Parse daily RMSSD, group into 7-day windows
    → hrv_engine.py         HRV feature extraction + ANS state space
    → gemini_mapper.py      Polyvagal Theory → MusicParameters (Gemini 2.5 Flash)
    → iso_player.py         ISO arc construction (Day 1 mirror → Day 7 target)
    → iso_audio.py          4-layer psychoacoustic synthesis + crossfade
```

Each module is independently runnable for testing.

---

## System requirements

- Python 3.10+
- Working audio output (headphones strongly recommended for binaural beats)
- Internet connection for Gemini API calls (only once per week, on Day 1)
- Gemini API key (free tier, no billing required for this usage level)
- WHOOP CSV export (optional — `--sample` mode works without it)

---

## Scientific grounding

- HRV standards: Task Force ESC/NASPE 1996; Shaffer & Ginsberg 2017
- ANS model: Polyvagal Theory (Porges 2011)
- ISO Principle: Altshuler 1948; Grocke & Wigram 2007
- Music–physiology coupling: Bernardi 2006; Thayer & Lane 2000; McConnell et al. 2014
- Binaural beats: theta 4–8 Hz (parasympathetic), alpha 8–14 Hz (balanced)
- 432/128 Hz root: Calamassi & Pomponi 2019 (HRV coupling study)
- Perfect fifth (3:2) — most consonant interval, associated with rest/safety
- Major third (5:4) — warm but slightly tense, mirrors mild sympathetic activation

---

## Roadmap

- [x] HRV feature extraction + 3-state ANS classification
- [x] Gemini LLM Polyvagal → music parameter mapping
- [x] Real-time additive synthesis + binaural beats
- [x] Full pipeline with terminal dashboard
- [x] WHOOP CSV input mode
- [x] ISO Principle 7-day arc player
- [x] 4-layer psychoacoustic synthesis (sub-bass, drone, chord harmony, rhythm)
- [x] Continuous crossfade between days within a week arc
- [x] Week selection menu with autonomic-state categories
- [x] 3-week bundled demo (high-variance real HRV data)
- [ ] Real-time BLE input from WHOOP band
- [ ] Session logging (CSV/JSON export for analysis)
- [ ] NTNU population validation study
