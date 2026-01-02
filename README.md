# Counter Speech Detection and Speaker Recognition

This project evaluates how synthetic audio perturbations impact automatic speech recognition (ASR) and speaker recognition. It preprocesses a small MP3 dataset, applies attack transformations, transcribes audio with Whisper, computes word error rate (WER), and measures speaker embedding similarity with SpeechBrain.

## Project Structure
- `python codes/randomnoise.py`: Generates and plays randomized audio noises/tones (white noise, sine, chirp, glitch, overlapping sines) using `sounddevice`.
- `python codes/testing_attacks phase 1.py`: Phase 1 pipeline (signal-processing attacks + metrics).
- `python codes/testing_attacks phase 2.py`: Phase 2 pipeline (synthetic-noise attacks + metrics).
- `small_data/`: Input dataset (MP3) and transcript CSV.
- `testing_attacks phase 1/`: Phase 1 outputs (cleaned WAV, attacked audio, CSV metrics).
- `testing_attacks phase 2/`: Phase 2 outputs (cleaned WAV, attacked audio, CSV metrics).

## Data
`small_data/` contains:
- `sample-*.mp3`: Audio clips.
- `small_data.csv`: Ground-truth transcripts with file paths and metadata.

The scripts read transcripts from `small_data/small_data.csv` and match by filename.

## Technical Overview
Both attack phases follow this pipeline:
1. **Load MP3 audio** via `librosa`, resample to 16 kHz, mono.
2. **Normalize** audio amplitude to `[-1, 1]`.
3. **Save** normalized WAVs into a phase-specific processed folder.
4. **Apply attacks** to each WAV and save per-attack subfolders.
5. **Transcribe** clean and attacked audio using Whisper (`tiny`).
6. **Compute metrics**:
   - WER: `jiwer.wer` (true vs clean, true vs attack, clean vs attack).
   - SNR (dB): compares clean vs attacked samples.
   - L2 norm: Phase 1 only, magnitude of perturbation.
7. **Speaker similarity** (Phase 1 and Phase 2):
   - SpeechBrain ECAPA embeddings + cosine similarity between clean and attacked audio.

### Phase 1 Attacks
Implemented in `python codes/testing_attacks phase 1.py`:
- Time stretch (slower/faster)
- Pitch shift
- Additive noise (target SNR)
- Low-pass filter
- Combined attack: pitch + time stretch + noise

Outputs:
- `testing_attacks phase 1/processed_small_data/` (clean WAVs)
- `testing_attacks phase 1/indiv_attacked_processed_small_data/`
- `testing_attacks phase 1/mult_attacked_processed_small_data/`
- `indiv_attack_metrics.csv`
- `mult_attack_metrics.csv`
- `indiv_attack_speaker_recognition.csv`
- `mult_attack_speaker_recognition.csv`

### Phase 2 Attacks
Implemented in `python codes/testing_attacks phase 2.py` and aligned with `randomnoise.py`:
- White noise
- Single sine wave
- Chirp
- Glitch bursts
- Overlapping sine waves

Outputs:
- `testing_attacks phase 2/processed_small_data/` (clean WAVs)
- `testing_attacks phase 2/indiv_attacked_processed_small_data/`
- `testing_attacks phase 2/indiv_attack_metrics.csv`
- `testing_attacks phase 2/indiv_attack_speaker_recognition.csv`

## Requirements
Python 3.9+ is recommended. Core dependencies:
- `torch`, `torchaudio`
- `numpy`, `pandas`
- `librosa`, `soundfile`, `scipy`
- `whisper` (OpenAI Whisper)
- `jiwer`
- `speechbrain`
- `sounddevice` (only needed for `randomnoise.py`)

Note: Whisper and SpeechBrain download model weights at first run (network access required).

## How to Run
From the project root:

```powershell
python "python codes\\testing_attacks phase 1.py"
python "python codes\\testing_attacks phase 2.py"
```

Optional noise generator (audio playback):

```powershell
python "python codes\\randomnoise.py"
```

## Outputs and Metrics
- **WER**: Quantifies transcription degradation (0 is perfect match).
- **SNR (dB)**: Measures noise level relative to clean signal.
- **L2 norm** (Phase 1): Perturbation magnitude.
- **Speaker cosine similarity**: Higher values indicate closer speaker identity after attack.

## Notes
- All outputs are written under their respective `testing_attacks phase */` folders.
- If you change `sample_rate` or attack parameters, re-run to regenerate results.
