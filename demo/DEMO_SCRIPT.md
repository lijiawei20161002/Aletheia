# Aletheia — Demo

`demo/aletheia_demo.mp4`  ·  1:43  ·  1280×720  ·  ambient music

---

## What's in the video

| Timestamp | Segment | Visual |
|---|---|---|
| 0:00–0:05 | Title card | "Aletheia — AI Propaganda Detection with Built-in Trust Auditing" |
| 0:05–0:23 | Problem hook | Slides building to "HIDDEN REASONING" |
| 0:23–0:39 | Case 1 — TRUSTWORTHY | Terminal output streams in, green verdict |
| 0:39–1:05 | Case 2 — HIDDEN_REASONING | Reasoning contradicts output; explainer panel overlaid |
| 1:05–1:19 | Cases 3 & 4 + summary table | HONEST_FAILURE, UNRELIABLE, all four classified |
| 1:19–1:33 | Key insight | "The AI's explanation cannot be used as evidence — even if the score is right" |
| 1:33–1:47 | Architecture diagram | ASCII diagram from README, colour-coded trust labels |
| 1:47–1:44 | Closing | GitHub URL + tagline |

The centrepiece is **Case 2 (HIDDEN_REASONING)**: the model gives a correct propaganda score
while its reasoning explicitly denies finding any patterns — a failure mode invisible to
either CoTShield or the pattern verifier alone, caught only by their cross-check.

---

## How to regenerate

**Requirements:** Python 3.9+, ffmpeg

```bash
# 1. Install Python deps (from repo root)
pip install Pillow moviepy numpy

# 2. Generate silent video
python3 make_demo.py
# → demo/aletheia_demo_silent.mp4

# 3. Add ambient music
python3 demo/add_audio.py
# → demo/aletheia_demo.mp4
```

`add_audio.py` has no external dependencies beyond `numpy` and `ffmpeg` — music is
synthesised entirely from numpy sine waves (D minor pad, sub-bass drone, shimmer arpeggio).

To add a voice-over, install `edge-tts` and adapt the narration segments at the top of
`add_audio.py` using `edge_tts.Communicate` — see the narration timing table below.

---

## Narration timing (if re-adding voice)

| # | Segment | Duration | Narration |
|---|---|---|---|
| 1 | Title | 5.0 s | "Aletheia. Open-source AI propaganda detection with built-in trust auditing." |
| 2 | Problem | 18.5 s | "AI detectors give you a score. But can you trust how the AI got there? There's a failure mode nobody talks about: Hidden Reasoning. The model gives you the right answer — for the wrong reasons." |
| 3 | Case 1 | 6.7 s | "Case one: Trustworthy. Clean reasoning, patterns verified. Trust score 0.89." |
| 4 | Case 2 | 29.0 s | "Case two — the critical one. Same text. Watch the reasoning: 'fear appeal cannot be found.' The output says: score 8 out of 10 — fear appeal present. Pattern verifier: fear IS in the text. CoTShield flags the reversal. Verdict: Hidden Reasoning. Trust 0.20. Neither check alone catches this." |
| 5 | Cases 3 & 4 | 9.3 s | "Case three: Honest Failure. Case four: Unreliable — both wrong. All four correctly classified." |
| 6 | Key insight | 11.1 s | "In legal or journalistic contexts, an AI's explanation cannot be used as evidence — even if the score is right. Audit the reasoning, not just the output." |
| 7 | Architecture | 13.4 s | "Under the hood: Claude analyses the text. CoTShield checks the reasoning chain. The pattern verifier confirms the claimed techniques are present. Their cross-check produces one of five trust labels." |
| 8 | Closing | 10.1 s | "Aletheia is open source and running today. AI that explains itself — and can prove it." |
