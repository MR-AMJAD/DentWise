#!/usr/bin/env python3
"""
finetune_whisper.py — Clinician Speaker Adaptation for Dental ASR
==================================================================
Personalises OpenAI Whisper to a specific clinician's accent using
LoRA (Low-Rank Adaptation) on the encoder attention Q/V projections.

USAGE
-----
  # Step 1: record enrollment audio (run this script with --enroll)
  python finetune_whisper.py --enroll --enrollment_dir enrollment/

  # Step 2: fine-tune
  python finetune_whisper.py --enrollment_dir enrollment/ --model_size small

  # Step 3: adapted model loads automatically on next session start.
  #   File: enrollment/adapted_model.pt

ADDING NEW COMMANDS IN FUTURE
------------------------------
  1. Add the new phrase strings to STANDARD_PHRASES below.
  2. Run --enroll again — existing recordings are skipped, only new ones prompted.
  3. Run fine-tune again. It overwrites adapted_model.pt.
  4. Restart live_charting_final.py — it auto-loads the updated model.
  Also update DentalFSG / PHONEME_VARIANTS in live_charting_final.py
  so the recogniser knows what to do with the new word.

REQUIREMENTS
------------
  pip install openai-whisper torch torchaudio soundfile SpeechRecognition pyaudio
  pip install audiomentations   # optional — better augmentation

FIXES in this version
----------------------
  1. LoRA A/B kept float32 for stability; forward casts result to x.dtype
     (fixes Half vs float mismatch after weight merge)
  2. LoRA A/B created on same device as base weight (fixes cuda/cpu mismatch)
  3. evaluate() uses temp WAV + whisper.transcribe() (correct API)
  4. --enroll built into this script (no longer needs live_charting_final.py)
"""

import os, sys, json, math, argparse, time, random, warnings, re
from pathlib import Path
from typing import List
from dataclasses import dataclass

import numpy as np
warnings.filterwarnings('ignore', category=UserWarning)

# ── Number words for 1–32 ─────────────────────────────────────────────────────
_NUM_WORDS = {
    1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven',
    8:'eight', 9:'nine', 10:'ten', 11:'eleven', 12:'twelve', 13:'thirteen',
    14:'fourteen', 15:'fifteen', 16:'sixteen', 17:'seventeen', 18:'eighteen',
    19:'nineteen', 20:'twenty', 21:'twenty one', 22:'twenty two',
    23:'twenty three', 24:'twenty four', 25:'twenty five', 26:'twenty six',
    27:'twenty seven', 28:'twenty eight', 29:'twenty nine', 30:'thirty',
    31:'thirty one', 32:'thirty two',
}


def _build_phrase_list() -> List[str]:
    """
    Comprehensive enrollment phrase list for DentWise voice fine-tuning.

    Covers every command the system accepts, in multiple natural sentence
    forms so Whisper learns them in clinical context, not isolation.
    Total: ~350 phrases across 7 sections.

    Sections:
      1.  Core single-word commands (on/off/stop/start/end/go)
      2.  Mode switching (stitching / charting / panorama)
      3.  Stitching session commands
      4.  Charting session commands
      5.  Surface commands (buccal / lingual)
      6.  Tooth numbers 1-32 with B/L — short form
      7.  Tooth numbers with sentence context — especially 20+
      8.  Clinical workflow phrases
    """
    phrases = []

    # ── Section 1: Core single-word commands ─────────────────────────────────
    # Most misrecognised because they are short and context-free.
    # Record each one 3x: once slow, once normal speed, once in a sentence.
    section1 = [
        # ON
        "on",
        # OFF
        "off",
        # STOP
        "stop",
        # START
        "start",
        # GO
        "go",
        # END
        "end",
        # PAUSE
        "pause",
        # RESUME
        "resume",
        # SAVE
        "save",
        # DISCARD
        "discard",
        # YES
        "yes",
        # NO
        "no",
        # DONE
        "done",
        # FINISH
        "finish",
        # BEGIN
        "begin",
        # ACTIVATE
        "activate",
        # CONFIRM
        "confirm",
        # REPEAT
        "repeat",
        # VIEW
        "view",
        # CANCEL
        "cancel",
        # CONTINUE
        "continue",
        # READY
        "ready",
    ]
    phrases.extend(section1)

    # ── Section 2: Mode switching ─────────────────────────────────────────────
    section2 = [
        # STITCHING — full word and sentence forms
        "stitching",
        "switch to stitching",
        "go to stitching",
        "stitching mode",
        "stitching please",
        "start stitching",
        "open stitching",
        "panorama stitching",
        # CHARTING — full word and sentence forms
        "charting",
        "switch to charting",
        "go to charting",
        "charting mode",
        "charting please",
        "start charting",
        "open charting",
        "dental charting",
        "begin charting",
        # PANORAMA — used as stitching synonym
        "panorama",
        "panorama mode",
        "take panorama",
        "start panorama",
        "panorama on",
        "panorama off",
        "panorama start",
        "panorama stop",
        "panorama save",
        "panorama done",
        "panorama ready",
    ]
    phrases.extend(section2)

    # ── Section 3: Stitching session commands ────────────────────────────────
    section3 = [
        # ON variations
        "on",
        "turn on",
        "switch on",
        "recording on",
        "panorama on",
        "stitching on",
        "start the panorama",
        "start recording now",
        "begin recording",
        # OFF variations
        "off",
        "turn off",
        "stop recording",
        "panorama off",
        "stitching off",
        "stop the panorama",
        "end the panorama",
        "finish the panorama",
        "stop stitching now",
        "end stitching",
        "finish stitching",
        # SAVE
        "save",
        "save panorama",
        "save the panorama",
        "save and stop",
        "save session",
        "stitching save",
        "panorama save",
        # PAUSE / RESUME
        "pause",
        "pause recording",
        "hold on",
        "stand by",
        "resume",
        "resume recording",
        "continue recording",
        "carry on",
        "go ahead",
        # DISCARD
        "discard",
        "discard session",
        "throw it away",
        "clear this",
        "start over",
        # VIEW / IMU
        "view",
        "imu view",
        "show imu",
        "toggle imu",
        # CALIBRATE
        "calibrate",
        "calibrate imu",
        "set scale",
    ]
    phrases.extend(section3)

    # ── Section 4: Charting session commands ─────────────────────────────────
    section4 = [
        # GO / ACTIVATE
        "go",
        "go ahead",
        "start now",
        "activate",
        "begin scanning",
        "start scanning",
        "record this tooth",
        "chart this tooth",
        "measure this tooth",
        "scan this tooth",
        # STOP
        "stop",
        "stop now",
        "okay stop",
        "please stop",
        "stop scanning",
        "stop charting",
        "finished scanning",
        "done scanning",
        # END
        "end",
        "end now",
        "end session",
        "end charting",
        "end all",
        "all done",
        "finish charting",
        "finish session",
        "charting complete",
        "charting done",
        "that is all",
        # CONFIRM / YES / NO
        "confirm",
        "yes confirm",
        "confirmed",
        "yes",
        "no",
        "no repeat",
        # REPEAT / REDO
        "repeat",
        "repeat that",
        "redo",
        "try again",
        "again",
        # NEXT
        "next tooth",
        "move to next",
        "next please",
        "go to next",
        "next one",
        # PREVIOUS
        "previous tooth",
        "go back",
        "previous one",
    ]
    phrases.extend(section4)

    # ── Section 5: Surface commands ──────────────────────────────────────────
    section5 = [
        "buccal",
        "lingual",
        "buccal surface",
        "lingual surface",
        "buccal side",
        "lingual side",
        "record buccal",
        "record lingual",
        "scan buccal",
        "scan lingual",
        "buccal please",
        "lingual please",
        "switch to buccal",
        "switch to lingual",
        "buccal surface please",
        "lingual surface please",
    ]
    phrases.extend(section5)

    # ── Section 6: Tooth numbers 1-32, short form B and L ────────────────────
    for n in range(1, 33):
        phrases.append(f"{_NUM_WORDS[n]} B")
    for n in range(1, 33):
        phrases.append(f"{_NUM_WORDS[n]} L")
    # Also with buccal / lingual spelled out
    for n in range(1, 33):
        phrases.append(f"{_NUM_WORDS[n]} buccal")
    for n in range(1, 33):
        phrases.append(f"{_NUM_WORDS[n]} lingual")

    # ── Section 7: Sentence-level tooth phrases (especially 20+) ─────────────
    # Whisper is far more accurate with sentence context than isolated words.
    # Twenty-something numbers are the most commonly misrecognised in practice.
    section7 = []
    # Every tooth 20-32 with B and L in multiple sentence forms
    for n in range(20, 33):
        w = _NUM_WORDS[n]
        section7 += [
            f"{w} B",
            f"{w} L",
            f"{w} buccal",
            f"{w} lingual",
            f"tooth {w} buccal",
            f"tooth {w} lingual",
            f"tooth {w} B",
            f"tooth {w} L",
            f"{w} B stop",
            f"{w} L stop",
            f"{w} buccal stop",
            f"{w} lingual stop",
            f"go {w} buccal",
            f"go {w} lingual",
        ]
    # Same for teeth 1-10 and 11-19 with sentence context
    for n in list(range(1,11)) + list(range(11,20)):
        w = _NUM_WORDS[n]
        section7 += [
            f"tooth {w} buccal",
            f"tooth {w} lingual",
            f"{w} buccal stop",
            f"{w} lingual stop",
        ]
    phrases.extend(section7)

    # ── Section 8: Clinical workflow and ambient phrases ─────────────────────
    section8 = [
        # Workflow
        "probe stable",
        "probe in place",
        "probe ready",
        "good contact",
        "upper jaw",
        "lower jaw",
        "upper teeth",
        "lower teeth",
        "right side",
        "left side",
        "front teeth",
        "back teeth",
        "ready",
        "proceed",
        "mark anchor",
        "set anchor",
        "bad frame",
        "bad region",
        "skip bad",
        "skip this",
        # IMU
        "imu on",
        "imu off",
        "imu connected",
        "imu ready",
        # Common ambient phrases the model should NOT confuse with commands
        "thank you",
        "open wide",
        "rinse please",
        "spit please",
        "almost done",
        "one moment",
        "hold still",
        "do not move",
        "relax",
    ]
    phrases.extend(section8)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in phrases:
        key = p.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique


STANDARD_PHRASES = _build_phrase_list()

LORA_RANK       = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05
LEARNING_RATE   = 3e-4
MAX_EPOCHS      = 20   # default epochs — enough for accent adaptation, not overfitting
EARLY_STOP_PAT  = 5
BATCH_SIZE      = 4
AUG_FACTOR      = 4
SAMPLE_RATE     = 16000
MAX_AUDIO_LEN_S = 5.0


# ─── LoRA as proper nn.Module ─────────────────────────────────────────────────

def _make_lora_linear(base_linear, rank: int, alpha: float, dropout: float):
    """
    Wraps a Linear layer with LoRA A/B matrices.
    Built inside a function so imports are guaranteed available.
    forward(x) = base(x) + dropout(x @ A.T) @ B.T * (alpha/rank)
    """
    import torch, torch.nn as nn

    d_out, d_in = base_linear.weight.shape
    scale = alpha / rank

    class LoRALinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.base    = base_linear
            self.scale   = scale
            dev   = base_linear.weight.device
            dtype = base_linear.weight.dtype
            self.lora_A  = nn.Parameter(torch.empty(rank, d_in,  device=dev, dtype=torch.float32))
            self.lora_B  = nn.Parameter(torch.zeros(d_out, rank, device=dev, dtype=torch.float32))
            self.dropout = nn.Dropout(p=dropout)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # Store target dtype so forward can cast correctly
            self._dtype  = dtype

        def forward(self, x):
            # Compute LoRA delta in float32 for numerical stability,
            # then cast result to match x (may be float16 on GPU)
            x32    = x.float()
            delta  = self.dropout(x32 @ self.lora_A.T) @ self.lora_B.T * self.scale
            return self.base(x) + delta.to(x.dtype)

        def merge_weights(self):
            with torch.no_grad():
                delta = self.scale * (self.lora_B @ self.lora_A)
                self.base.weight.data += delta.to(self.base.weight.dtype)

    return LoRALinear()


def inject_lora(model, rank: int, alpha: float, dropout: float) -> list:
    """
    Inject LoRA into Q and V projections of every encoder attention block.

    Whisper's MultiHeadAttention stores projections as:
        .query  .key  .value  .out  (all Linear-like)

    We use runtime isinstance check against both nn.Linear and whisper's
    custom Linear (if present) so this works across all Whisper versions.
    """
    import torch.nn as nn

    # Collect all Linear base types present in this Whisper version
    linear_types = [nn.Linear]
    try:
        import whisper.model as _wm
        if hasattr(_wm, 'Linear'):
            linear_types.append(_wm.Linear)
    except Exception:
        pass
    linear_types = tuple(linear_types)

    lora_layers = []
    seen        = set()

    for block_name, block in model.encoder.named_modules():
        # Each ResidualAttentionBlock has a .attn MultiHeadAttention
        attn = getattr(block, 'attn', None)
        if attn is None:
            continue
        for proj_name in ('query', 'value'):
            proj = getattr(attn, proj_name, None)
            if proj is None or not isinstance(proj, linear_types):
                continue
            uid = id(proj)
            if uid in seen:
                continue
            seen.add(uid)
            lora = _make_lora_linear(proj, rank, alpha, dropout)
            setattr(attn, proj_name, lora)
            lora_layers.append(lora)

    print(f"[LoRA] Injected {len(lora_layers)} adapters  "
          f"(rank={rank}, alpha={alpha})")
    return lora_layers


def get_lora_params(lora_layers: list) -> list:
    params = []
    for L in lora_layers:
        params.extend([L.lora_A, L.lora_B])
    return params


# ─── Audio augmentation ───────────────────────────────────────────────────────

def augment_audio(wav: np.ndarray, sr: int, aug_idx: int) -> np.ndarray:
    try:
        import audiomentations as AM
        aug = [
            AM.TimeStretch(min_rate=0.9, max_rate=1.1, p=1.0),
            AM.PitchShift(min_semitones=-2, max_semitones=2, p=1.0),
            AM.AddGaussianNoise(min_amplitude=0.002, max_amplitude=0.01, p=1.0),
            AM.RoomSimulator(p=1.0),
        ][aug_idx % 4]
        return aug(samples=wav.astype(np.float32), sample_rate=sr)
    except Exception:
        pass
    wav = wav.astype(np.float32)
    if aug_idx % 4 == 0:
        f = random.uniform(0.90, 1.10)
        wav = np.interp(np.linspace(0, len(wav)-1, int(len(wav)/f)),
                        np.arange(len(wav)), wav)
    elif aug_idx % 4 == 1:
        f = 2 ** (random.uniform(-2, 2) / 12)
        n = int(len(wav) / f)
        wav = np.interp(np.linspace(0, len(wav)-1, n), np.arange(len(wav)), wav)
    elif aug_idx % 4 == 2:
        rms = np.sqrt(np.mean(wav**2)) + 1e-8
        wav = wav + np.random.randn(len(wav)).astype(np.float32) * rms * 0.05
    else:
        ir_len = int(sr * 0.3)
        ir = (np.exp(-8*np.linspace(0,1,ir_len)) *
              np.random.randn(ir_len)).astype(np.float32)
        ir /= np.abs(ir).max() + 1e-8
        wav = np.convolve(wav, ir, mode='full')[:len(wav)]
    peak = np.abs(wav).max()
    return (wav / peak * 0.95).astype(np.float32) if peak > 1e-6 else wav


# ─── Dataset ──────────────────────────────────────────────────────────────────

@dataclass
class Sample:
    wav: np.ndarray
    transcript: str


def load_dataset(recordings_dir: Path, phrases: List[str],
                 aug_factor: int) -> List[Sample]:
    try:
        import soundfile as sf
    except ImportError:
        print("ERROR: pip install soundfile"); sys.exit(1)

    max_s    = int(MAX_AUDIO_LEN_S * SAMPLE_RATE)
    samples  = []
    wav_files = sorted(recordings_dir.glob("phrase_*.wav"))
    if not wav_files:
        print(f"ERROR: No WAV files in {recordings_dir}"); sys.exit(1)
    print(f"[Dataset] Found {len(wav_files)} recordings")

    for wav_path in wav_files:
        try:
            idx = int(wav_path.stem.replace("phrase_", ""))
        except ValueError:
            continue
        if idx >= len(phrases):
            continue
        transcript = phrases[idx].lower().strip()
        try:
            wav, sr = sf.read(str(wav_path), dtype='float32', always_2d=False)
        except Exception as e:
            print(f"  ⚠ {wav_path.name}: {e}"); continue
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE:
            try:
                import torchaudio, torch
                t = torch.from_numpy(wav).unsqueeze(0)
                wav = torchaudio.functional.resample(t, sr, SAMPLE_RATE).squeeze(0).numpy()
            except Exception:
                n = int(len(wav) * SAMPLE_RATE / sr)
                wav = np.interp(np.linspace(0, len(wav)-1, n), np.arange(len(wav)), wav)
        # Trim silence
        e = wav**2; ce = np.cumsum(e); te = ce[-1]
        if te > 1e-8:
            s0 = np.searchsorted(ce, te*0.01)
            s1 = np.searchsorted(ce, te*0.99)
            wav = wav[max(0,s0-160): min(len(wav), s1+160)]
        wav = wav[:max_s]
        samples.append(Sample(wav=wav.copy(), transcript=transcript))
        for ai in range(aug_factor):
            samples.append(Sample(wav=augment_audio(wav, SAMPLE_RATE, ai)[:max_s],
                                  transcript=transcript))

    print(f"[Dataset] {len(samples)} total samples ({aug_factor}× aug)")
    random.shuffle(samples)
    return samples


def samples_to_batches(samples: List[Sample], model, tokenizer,
                        batch_size: int, device) -> List[dict]:
    import torch, whisper as _w
    sot_seq = list(tokenizer.sot_sequence)
    eot     = tokenizer.eot
    batches, mels_buf, labels_buf = [], [], []

    for i, s in enumerate(samples):
        mel = _w.log_mel_spectrogram(_w.pad_or_trim(s.wav.astype(np.float32)))
        mels_buf.append(mel.to(device))
        toks = tokenizer.encode(s.transcript)
        labels_buf.append(torch.tensor(sot_seq + toks + [eot],
                                        dtype=torch.long, device=device))
        if len(mels_buf) == batch_size or i == len(samples)-1:
            maxl = max(t.shape[0] for t in labels_buf)
            pad  = torch.full((len(labels_buf), maxl), -100,
                               dtype=torch.long, device=device)
            for j, t in enumerate(labels_buf):
                pad[j, :t.shape[0]] = t
            batches.append({'mels': torch.stack(mels_buf), 'labels': pad})
            mels_buf, labels_buf = [], []
    return batches


# ─── Evaluation ───────────────────────────────────────────────────────────────

def compute_wer(ref: str, hyp: str) -> float:
    r = ref.lower().split(); h = hyp.lower().split()
    if not r: return 0.0 if not h else 1.0
    d = np.zeros((len(r)+1, len(h)+1), int)
    for i in range(len(r)+1): d[i,0] = i
    for j in range(len(h)+1): d[0,j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            d[i,j] = d[i-1,j-1] if r[i-1]==h[j-1] \
                      else 1+min(d[i-1,j], d[i,j-1], d[i-1,j-1])
    return d[-1,-1] / len(r)


def evaluate(model, eval_samples: List[Sample], device) -> float:
    import torch, whisper as _w, soundfile as sf, tempfile
    model.eval()
    wers = []
    fp16 = (str(device) != 'cpu')
    with torch.no_grad():
        for s in eval_samples[:12]:
            tmp = tempfile.mktemp(suffix='.wav')
            try:
                sf.write(tmp, s.wav.astype(np.float32), SAMPLE_RATE)
                res = _w.transcribe(model, tmp, language='en',
                                     condition_on_previous_text=False, fp16=fp16)
                hyp = res['text'].strip().lower()
            except Exception as e:
                hyp = ""
                print(f"  [Eval] {e}")
            finally:
                try: os.unlink(tmp)
                except: pass
            wer = compute_wer(s.transcript, hyp)
            wers.append(wer)
            print(f"  ref: '{s.transcript}'  |  hyp: '{hyp}'  WER={wer:.0%}")
    return float(np.mean(wers)) if wers else 1.0


# ─── Training ─────────────────────────────────────────────────────────────────

def train(model, lora_layers, batches, lr, max_epochs, patience, device, output_path,
          model_size="small"):
    import torch, torch.nn as nn

    lora_params = get_lora_params(lora_layers)
    n_lora  = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[Train] LoRA params: {n_lora:,}  ({100*n_lora/n_total:.3f}%)")
    if n_lora == 0:
        print("[Train] ERROR: 0 LoRA parameters — injection failed"); return float('inf')

    for p in model.parameters(): p.requires_grad_(False)
    for p in lora_params:        p.requires_grad_(True)

    opt  = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.01)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=lr*0.1)
    crit = nn.CrossEntropyLoss(ignore_index=-100)

    best_loss, no_improve, best_state = float('inf'), 0, None
    print(f"\n[Train] lr={lr}  epochs={max_epochs}  batches/ep={len(batches)}\n")
    print(f"{'Epoch':>6}  {'Loss':>8}  {'Best':>8}  {'LR':>10}  Time")
    print("-"*50)

    for epoch in range(1, max_epochs+1):
        t0 = time.time(); model.train(); epoch_loss = 0.0
        random.shuffle(batches)
        for batch in batches:
            mels, labels = batch['mels'], batch['labels']
            af     = model.encoder(mels)
            din    = labels[:, :-1].clamp(min=0)
            dtgt   = labels[:, 1:]
            logits = model.decoder(din, af)
            loss   = crit(logits.reshape(-1, logits.shape[-1]), dtgt.reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step(); epoch_loss += loss.item()

        sch.step()
        avg  = epoch_loss / max(len(batches), 1)
        ok   = avg < best_loss - 1e-5
        print(f"{epoch:>6}  {avg:>8.4f}  {best_loss:>8.4f}  "
              f"{sch.get_last_lr()[0]:>10.2e}  {time.time()-t0:.1f}s"
              + (" ←" if ok else ""))
        if ok:
            best_loss  = avg; no_improve = 0
            best_state = {n: p.data.clone()
                          for n,p in model.named_parameters() if p.requires_grad}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n[Train] Early stop at epoch {epoch}"); break

    if best_state:
        for n, p in model.named_parameters():
            if p.requires_grad and n in best_state:
                p.data.copy_(best_state[n])

    # Save LoRA adapter weights ONLY — do NOT merge into base weights.
    # Merging causes "enforce fail at inline_container.cc" corruption on some
    # PyTorch versions when tensor sizes don't align cleanly.
    # At inference time, LoRA weights are re-applied to a fresh base model.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lora_state = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }
    torch.save({
        "lora_state_dict": lora_state,
        "model_size":      model_size,    # passed in from finetune_one_user
        "lora_rank":       LORA_RANK,
        "lora_alpha":      LORA_ALPHA,
    }, output_path)
    print(f"[LoRA] ✓ Adapter saved (no merge) → {output_path}")
    return best_loss


# ─── Main ─────────────────────────────────────────────────────────────────────

def record_enrollment(enrollment_dir: Path, phrases: List[str],
                       mic_device=None):
    """
    Interactive CLI enrollment: records each phrase as a WAV file.
    Already-recorded phrases are skipped — safe to re-run after adding phrases.

    Uses fixed-duration raw recording (not energy-gated) so short single words
    like "go", "end", "yes" are captured reliably regardless of energy threshold.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        print("ERROR: pip install SpeechRecognition pyaudio"); sys.exit(1)

    try:
        import pyaudio
    except ImportError:
        print("ERROR: pip install pyaudio"); sys.exit(1)

    recordings_dir = enrollment_dir / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)

    CHUNK       = 1024
    FORMAT      = pyaudio.paInt16
    CHANNELS    = 1
    RATE        = 16000
    # Short phrases (≤2 words): 2.0 s — long phrases (>2 words): 3.5 s
    # This bypasses energy-gate timeouts entirely.

    print(f"\n{'='*65}")
    print(f"  ENROLLMENT — {len(phrases)} phrases to record")
    print(f"{'='*65}")
    print("  Say each phrase clearly at normal clinical speaking pace.")
    print("  Press ENTER, then speak IMMEDIATELY — recording starts at once.")
    print("  Short phrases (go, stop…): 2 sec.  Long phrases: 3.5 sec.")
    print("  Already-recorded phrases are skipped automatically.")
    print(f"  Recordings saved to: {recordings_dir}\n")

    recorded = skipped = failed = 0
    pa = pyaudio.PyAudio()

    # Find device index
    dev_index = mic_device
    if dev_index is None:
        # Use default input device
        try:
            dev_index = pa.get_default_input_device_info()['index']
        except Exception:
            dev_index = 0

    for i, phrase in enumerate(phrases):
        wav_path = recordings_dir / f"phrase_{i:03d}.wav"
        if wav_path.exists():
            print(f"  [{i+1:>3}/{len(phrases)}] ✓ (exists)  '{phrase}'")
            skipped += 1
            continue

        # Duration: 2s for 1-word phrases, 3.5s for longer
        n_words  = len(phrase.split())
        duration = 2.0 if n_words <= 2 else 3.5
        n_frames = int(RATE * duration / CHUNK)

        retry = True
        while retry:
            input(f"\n  [{i+1:>3}/{len(phrases)}] Say: \"{phrase}\"\n"
                  f"  Press ENTER then speak immediately ({duration:.0f}s)…")
            print("  🎤 Recording…", end='', flush=True)

            try:
                stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                                 input=True, input_device_index=dev_index,
                                 frames_per_buffer=CHUNK)
                frames = []
                for _ in range(n_frames):
                    frames.append(stream.read(CHUNK, exception_on_overflow=False))
                stream.stop_stream()
                stream.close()

                # Write WAV
                import wave, io
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(pa.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                wav_path.write_bytes(buf.getvalue())
                print(f" ✓  saved")
                recorded += 1
                retry = False

            except Exception as e:
                print(f" ✗  error: {e}")
                ans = input("  Retry? (Enter = yes, n = skip): ").strip().lower()
                if ans == 'n':
                    failed += 1
                    retry = False
                # else loop again

    pa.terminate()

    # Save metadata
    meta = {
        'n_phrases':  len(phrases),
        'n_recorded': recorded,
        'n_skipped':  skipped,
        'n_failed':   failed,
        'timestamp':  time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(enrollment_dir / "enrollment_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Recorded: {recorded}  Skipped (existing): {skipped}  Failed: {failed}")
    print(f"  Total available: {recorded + skipped} / {len(phrases)}")
    if failed > 0:
        print(f"  Re-run --enroll to retry failed phrases.")
    print(f"{'='*65}\n")
    return recorded + skipped


def main():
    p = argparse.ArgumentParser(
        description="Whisper LoRA fine-tuning — one adapted model per clinician")
    p.add_argument("--user",           type=str, default=None,
                   help="Clinician name/ID, e.g. 'dr_smith'.  "
                        "Creates enrollment/<user>/ so each person's model is kept separately.  "
                        "If omitted, falls back to --enrollment_dir directly.")
    p.add_argument("--enrollment_dir", default="enrollment",
                   help="Root enrollment folder (default: enrollment/).  "
                        "With --user this becomes enrollment/<user>/")
    p.add_argument("--enroll",         action="store_true",
                   help="Record enrollment audio then exit (run before fine-tuning)")
    p.add_argument("--list_users",     action="store_true",
                   help="List all enrolled users and exit")
    p.add_argument("--mic_device",     type=int, default=None)
    p.add_argument("--model_size",     default="small",
                   choices=["tiny","base","small","medium"])
    p.add_argument("--output_path",    default=None)
    p.add_argument("--lora_rank",      type=int,   default=LORA_RANK)
    p.add_argument("--lora_alpha",     type=float, default=LORA_ALPHA)
    p.add_argument("--lora_dropout",   type=float, default=LORA_DROPOUT)
    p.add_argument("--lr",             type=float, default=LEARNING_RATE)
    p.add_argument("--epochs",         type=int,   default=MAX_EPOCHS)
    p.add_argument("--patience",       type=int,   default=EARLY_STOP_PAT)
    p.add_argument("--aug_factor",     type=int,   default=AUG_FACTOR)
    p.add_argument("--eval_split",     type=float, default=0.15)
    p.add_argument("--cpu",            action="store_true")
    p.add_argument("--phrases_json",   default=None)
    args = p.parse_args()

    root_dir = Path(args.enrollment_dir)

    # ── List users ─────────────────────────────────────────────────────────────
    if args.list_users:
        print(f"\nEnrolled users in {root_dir}:")
        found = False
        for d in sorted(root_dir.iterdir()) if root_dir.exists() else []:
            model_pt = d / "adapted_model.pt"
            recs     = list((d / "recordings").glob("phrase_*.wav")) if (d/"recordings").exists() else []
            meta_f   = d / "enrollment_meta.json"
            meta     = json.load(open(meta_f)) if meta_f.exists() else {}
            if d.is_dir() and (model_pt.exists() or recs):
                status  = "✓ model ready" if model_pt.exists() else "⚠ recorded, not yet fine-tuned"
                ts      = meta.get('timestamp', '—')
                print(f"  {d.name:<20}  {len(recs):>3} recordings  {status}  ({ts})")
                found = True
        if not found:
            print("  (none found)")
        print()
        return

    # ── Resolve per-user enrollment dir ───────────────────────────────────────
    if args.user:
        # Sanitise: lowercase, spaces→underscores, strip non-alphanum
        user_id      = re.sub(r'[^a-z0-9_]', '', args.user.lower().replace(' ', '_'))
        enrollment_dir = root_dir / user_id
        print(f"[User] '{args.user}'  →  {enrollment_dir}")
    else:
        enrollment_dir = root_dir
    enrollment_dir.mkdir(parents=True, exist_ok=True)

    phrases = STANDARD_PHRASES
    if args.phrases_json:
        phrases = json.load(open(args.phrases_json))

    # ── Enrollment mode ────────────────────────────────────────────────────────
    if args.enroll:
        if not args.user:
            print("TIP: use --user <name> to keep each clinician's recordings separate")
            print("     e.g.  python finetune_whisper.py --enroll --user dr_smith\n")
        print(f"[Phrases] {len(phrases)} total phrases")
        record_enrollment(enrollment_dir, phrases, mic_device=args.mic_device)
        if args.user:
            print(f"  Now run:  python finetune_whisper.py --user {args.user} --model_size {args.model_size}")
        return

    # ── Fine-tuning mode ───────────────────────────────────────────────────────
def finetune_one_user(enrollment_dir: Path, user_label: str,
                       model_size: str, device, phrases: List[str],
                       lora_rank: int, lora_alpha: float, lora_dropout: float,
                       lr: float, max_epochs: int, patience: int,
                       aug_factor: int, eval_split: float,
                       output_path: Path) -> dict:
    """
    Fine-tune Whisper for one user's enrollment_dir.
    Returns summary dict. Raises on hard errors so --all_users can continue.
    """
    import whisper as _w

    recordings_dir = enrollment_dir / "recordings"
    if not recordings_dir.exists():
        raise FileNotFoundError(f"{recordings_dir} not found — run --enroll first")

    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {user_label}")
    print(f"  Model={model_size}  Device={device}  rank={lora_rank}")
    print(f"{'='*60}")

    print(f"[Whisper] Loading '{model_size}'…")
    model = _w.load_model(model_size, device=device)

    try:
        tokenizer = _w.tokenizer.get_tokenizer(
            multilingual=model.is_multilingual, language='en', task='transcribe')
    except Exception:
        tokenizer = _w.decoding.get_tokenizer(
            multilingual=model.is_multilingual, language='en', task='transcribe')

    originals = load_dataset(recordings_dir, phrases, aug_factor=0)
    if len(originals) < 5:
        raise ValueError(f"Only {len(originals)} recordings — need ≥5")

    n_eval  = max(1, int(len(originals) * eval_split))
    random.shuffle(originals)
    eval_s  = originals[:n_eval]
    train_o = originals[n_eval:]
    max_s   = int(MAX_AUDIO_LEN_S * SAMPLE_RATE)

    train_s = list(train_o)
    for s in train_o:
        for ai in range(aug_factor):
            train_s.append(Sample(wav=augment_audio(s.wav, SAMPLE_RATE, ai)[:max_s],
                                   transcript=s.transcript))
    random.shuffle(train_s)
    print(f"[Dataset] Train: {len(train_s)}  Eval: {len(eval_s)}")

    print("\n[Eval] Baseline WER…")
    baseline_wer = evaluate(model, eval_s, device)
    print(f"[Eval] Baseline WER: {baseline_wer:.1%}\n")

    lora_layers = inject_lora(model, lora_rank, lora_alpha, lora_dropout)
    if not lora_layers:
        raise RuntimeError("LoRA injection returned 0 layers")

    batches = samples_to_batches(train_s, model, tokenizer, BATCH_SIZE, device)

    best_loss = train(model, lora_layers, batches,
                      lr=lr, max_epochs=max_epochs, patience=patience,
                      device=device, output_path=output_path)

    print("\n[Eval] WER after fine-tuning…")
    adapted_wer = evaluate(model, eval_s, device)
    imp = (baseline_wer - adapted_wer) / max(baseline_wer, 1e-6) * 100
    print(f"[Eval] Adapted WER: {adapted_wer:.1%}  ({imp:+.1f}%)")

    summary = {
        'user':                user_label,
        'model_size':          model_size,
        'lora_rank':           lora_rank,
        'n_train':             len(train_s),
        'n_eval':              len(eval_s),
        'best_train_loss':     round(best_loss, 4),
        'baseline_wer':        round(baseline_wer, 4),
        'adapted_wer':         round(adapted_wer, 4),
        'wer_improvement_pct': round(imp, 1),
        'timestamp':           time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(enrollment_dir / "finetune_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  ✓ {user_label}: {baseline_wer:.1%} → {adapted_wer:.1%}  ({imp:+.1f}%)")
    print(f"  Model: {output_path}")

    # Free GPU memory before next user
    import torch
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return summary


def main():
    p = argparse.ArgumentParser(
        description="Whisper LoRA fine-tuning — one adapted model per clinician")
    p.add_argument("--user",           type=str, default=None,
                   help="Clinician name/ID, e.g. 'dr_smith'. "
                        "Creates enrollment/<user>/ so each person's model is separate.")
    p.add_argument("--all_users",      action="store_true",
                   help="Fine-tune ALL enrolled users sequentially. "
                        "Each user's model is saved independently. "
                        "Skips users whose recordings folder is missing.")
    p.add_argument("--enrollment_dir", default="enrollment",
                   help="Root enrollment folder (default: enrollment/). "
                        "With --user becomes enrollment/<user>/")
    p.add_argument("--enroll",         action="store_true",
                   help="Record enrollment audio then exit (run before fine-tuning)")
    p.add_argument("--list_users",     action="store_true",
                   help="List all enrolled users and exit")
    p.add_argument("--mic_device",     type=int, default=None)
    p.add_argument("--model_size",     default="small",
                   choices=["tiny","base","small","medium"])
    p.add_argument("--output_path",    default=None)
    p.add_argument("--lora_rank",      type=int,   default=LORA_RANK)
    p.add_argument("--lora_alpha",     type=float, default=LORA_ALPHA)
    p.add_argument("--lora_dropout",   type=float, default=LORA_DROPOUT)
    p.add_argument("--lr",             type=float, default=LEARNING_RATE)
    p.add_argument("--epochs",         type=int,   default=MAX_EPOCHS)
    p.add_argument("--patience",       type=int,   default=EARLY_STOP_PAT)
    p.add_argument("--aug_factor",     type=int,   default=AUG_FACTOR)
    p.add_argument("--eval_split",     type=float, default=0.15)
    p.add_argument("--cpu",            action="store_true")
    p.add_argument("--phrases_json",   default=None)
    args = p.parse_args()

    root_dir = Path(args.enrollment_dir)

    # ── List users ─────────────────────────────────────────────────────────────
    if args.list_users:
        print(f"\nEnrolled users in {root_dir}:")
        found = False
        for d in sorted(root_dir.iterdir()) if root_dir.exists() else []:
            model_pt = d / "adapted_model.pt"
            recs     = list((d / "recordings").glob("phrase_*.wav")) if (d/"recordings").exists() else []
            meta_f   = d / "enrollment_meta.json"
            meta     = json.load(open(meta_f)) if meta_f.exists() else {}
            summ_f   = d / "finetune_summary.json"
            summ     = json.load(open(summ_f)) if summ_f.exists() else {}
            if d.is_dir() and (model_pt.exists() or recs):
                status = "✓ model ready" if model_pt.exists() else "⚠ not yet fine-tuned"
                wer    = f"  WER {summ['adapted_wer']:.0%}" if 'adapted_wer' in summ else ""
                ts     = summ.get('timestamp', meta.get('timestamp', '—'))
                print(f"  {d.name:<20}  {len(recs):>3} recordings  {status}{wer}  ({ts})")
                found = True
        if not found:
            print("  (none found)")
        print()
        return

    phrases = STANDARD_PHRASES
    if args.phrases_json:
        phrases = json.load(open(args.phrases_json))

    # ── Enrollment mode ────────────────────────────────────────────────────────
    if args.enroll:
        if args.user:
            user_id        = re.sub(r'[^a-z0-9_]', '', args.user.lower().replace(' ', '_'))
            enrollment_dir = root_dir / user_id
        else:
            print("TIP: use --user <name> to keep each clinician's recordings separate")
            enrollment_dir = root_dir
        enrollment_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Phrases] {len(phrases)} total phrases  "
              f"({len([p for p in phrases if ' B' in p or ' L' in p])} letter-form + word-form)")
        record_enrollment(enrollment_dir, phrases, mic_device=args.mic_device)
        if args.user:
            print(f"  Now run:  python finetune_whisper.py --user {args.user}")
        return

    # ── Fine-tuning: shared setup ──────────────────────────────────────────────
    try:    import torch
    except: print("ERROR: pip install torch"); sys.exit(1)
    try:    import whisper as _w
    except: print("ERROR: pip install openai-whisper"); sys.exit(1)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    ft_kwargs = dict(
        model_size   = args.model_size,
        device       = device,
        phrases      = phrases,
        lora_rank    = args.lora_rank,
        lora_alpha   = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        lr           = args.lr,
        max_epochs   = args.epochs,
        patience     = args.patience,
        aug_factor   = args.aug_factor,
        eval_split   = args.eval_split,
    )

    # ── Fine-tune: default is all users ───────────────────────────────────────
    # If neither --user nor --all_users given, behave as --all_users.
    if not args.user and not args.all_users:
        args.all_users = True

    # ── Fine-tune ALL users ────────────────────────────────────────────────────
    if args.all_users:
        if not root_dir.exists():
            print(f"ERROR: {root_dir} does not exist — no users enrolled yet")
            sys.exit(1)
        user_dirs = [d for d in sorted(root_dir.iterdir())
                     if d.is_dir() and (d / "recordings").exists()]
        if not user_dirs:
            print(f"ERROR: no user subdirectories with recordings found in {root_dir}")
            sys.exit(1)

        print(f"\n[All users] Found {len(user_dirs)} enrolled users: "
              f"{[d.name for d in user_dirs]}")
        print(f"[All users] Fine-tuning sequentially on {device}…\n")

        results = []
        for user_dir in user_dirs:
            out = user_dir / "adapted_model.pt"
            try:
                summary = finetune_one_user(
                    enrollment_dir=user_dir,
                    user_label=user_dir.name,
                    output_path=out,
                    **ft_kwargs)
                results.append(summary)
            except Exception as e:
                print(f"\n  ✗ {user_dir.name} failed: {e}\n")
                results.append({'user': user_dir.name, 'error': str(e)})

        print(f"\n{'='*60}")
        print(f"  ALL USERS COMPLETE")
        print(f"{'='*60}")
        for r in results:
            if 'error' in r:
                print(f"  ✗ {r['user']:<20}  ERROR: {r['error']}")
            else:
                imp = r.get('wer_improvement_pct', 0)
                print(f"  ✓ {r['user']:<20}  "
                      f"{r.get('baseline_wer',0):.0%} → {r.get('adapted_wer',0):.0%}  "
                      f"({imp:+.0f}%)")
        print(f"{'='*60}\n")
        return

    # ── Fine-tune single user ──────────────────────────────────────────────────
    if args.user:
        user_id        = re.sub(r'[^a-z0-9_]', '', args.user.lower().replace(' ', '_'))
        enrollment_dir = root_dir / user_id
        user_label     = args.user
    else:
        enrollment_dir = root_dir
        user_label     = "(default)"
    enrollment_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output_path) if args.output_path \
                  else enrollment_dir / "adapted_model.pt"

    try:
        summary = finetune_one_user(
            enrollment_dir=enrollment_dir,
            user_label=user_label,
            output_path=output_path,
            **ft_kwargs)
    except Exception as e:
        print(f"\nERROR: {e}"); sys.exit(1)

    print(f"\n  Run with this model:")
    if args.user:
        print(f"    python live_charting_final.py --user {args.user} --model_path <ckpt>")
    print(f"  Run --list_users to see all clinicians.\n")


if __name__ == "__main__":
    main()
