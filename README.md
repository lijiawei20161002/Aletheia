# Aletheia

**Open-source AI platform for detecting media propaganda**

Aletheia analyzes media content to identify the structural fingerprints of manipulation — rhetorical techniques, emotional persuasion tactics, and coordinated narrative patterns — giving journalists, researchers, and civil society monitors the analytical leverage to see propaganda before it shapes public reality.

> *Aletheia (ἀλήθεια): Ancient Greek for truth, disclosure, unconcealedness.*

---

## What It Does

Paste any text — a news article, political speech, social media thread, broadcast transcript — and Aletheia returns a structured analysis:

- **Propaganda score** (0–10) with a plain-language verdict
- **Rhetorical techniques** — named manipulation tactics with quoted evidence from the text (false dichotomy, manufactured urgency, dehumanizing language, appeal to fear, etc.)
- **Emotional manipulation** — which emotions are being targeted, at what intensity, and how
- **Narrative framing** — core story being constructed, us-vs-them dynamics, scapegoating, false urgency flags
- **Key passages** — specific quotes identified as manipulative, with explanations
- **Summary** — 2–3 sentence plain-language findings

---

## Why It Matters

Democracy requires shared reality. Modern propaganda rarely works through outright lies — it works through emotional saturation and narrative anchoring that precede rational response. Generative AI now allows a single actor to produce thousands of thematically consistent articles, posts, and scripts in hours. Traditional fact-checking, which evaluates claims one at a time, cannot keep pace.

Aletheia uses AI to detect what human analysts cannot monitor at scale: the structural patterns that distinguish coordinated manipulation from legitimate political speech.

---

## Demo

![Aletheia screenshot](docs/screenshot.png)

Live demo: run locally (see setup below), paste text, get analysis in seconds.

---

## Setup

**Requirements:** Python 3.9+, an Anthropic API key

```bash
# Clone the repo
git clone https://github.com/lijiawei20161002/Aletheia
cd aletheia

# Create virtual environment
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
echo 'ANTHROPIC_API_KEY="your-key-here"' > .env

# Run
python main.py
```

Open `http://localhost:8000` in your browser.

---

## API

### `POST /analyze`

```json
{
  "text": "string (max 10,000 characters)"
}
```

**Response:**

```json
{
  "propaganda_score": 8,
  "verdict": "Highly manipulative content using fear appeals and us-vs-them framing.",
  "rhetorical_techniques": [
    {
      "technique": "Appeal to Fear",
      "description": "Exaggerating threats to provoke fear rather than rational assessment.",
      "example": "every single day they remain in power, our nation inches closer to total collapse"
    }
  ],
  "emotional_manipulation": {
    "primary_emotion": "Fear",
    "secondary_emotions": ["Anger", "Disgust"],
    "intensity": "high",
    "analysis": "Content systematically targets fear and tribal outrage..."
  },
  "narrative_framing": {
    "core_narrative": "Patriotic citizens under existential threat from corrupt elites",
    "us_vs_them": true,
    "scapegoating": true,
    "false_urgency": true,
    "analysis": "..."
  },
  "key_passages": [
    {
      "passage": "this is our last chance to save the country",
      "concern": "False urgency designed to bypass deliberative reasoning"
    }
  ],
  "summary": "..."
}
```

---

## Stack

- **Backend:** Python, FastAPI
- **AI:** Anthropic Claude (claude-sonnet-4-6)
- **Frontend:** Vanilla HTML/CSS/JS (no build step)

---

## Roadmap

- [ ] Multilingual support (Arabic, Spanish, French, Mandarin)
- [ ] Narrative campaign detection — identify coordinated patterns across multiple documents
- [ ] Browser extension
- [ ] Public campaign registry
- [ ] Independent methodology audit (false positive / adversarial robustness)
- [ ] CLI tool for batch analysis

---

## License

AGPL-3.0 — free to use, modify, and distribute. Contributions welcome.

---

## About

Built by [Jiawei Li](https://github.com/lijiawei20161002).

---

## Dual-Layer Reasoning Auditor

Aletheia includes an integrated **dual-layer reasoning auditor** that detects a critical AI failure mode: models reaching correct conclusions via deceptive or unsupported reasoning chains — invisible to either a CoT monitor or a formal verifier working alone.

> *Call it the **Clever Hans problem**: the model gets the right answer, but for the wrong reason.*

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│   Input: (reasoning text, output claim, optional conjecture)        │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
           ┌───────────┴────────────┐
           ▼                        ▼
  ┌─────────────────┐      ┌─────────────────────┐
  │   CoTShield     │      │  AutoConjecture      │
  │   Detector      │      │  ProofEngine         │
  └────────┬────────┘      └──────────┬───────────┘
           └──────────┬───────────────┘
                      ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │   TrustLabel                                                     │
  │   CoT clean + Proof OK   → TRUSTWORTHY        (trust ~0.9)      │
  │   CoT suspect + Proof OK → HIDDEN_REASONING   (trust ~0.2)  ★  │
  │   CoT clean + Proof fail → HONEST_FAILURE     (trust ~0.35)     │
  │   CoT suspect + Proof fail → UNRELIABLE       (trust ~0.05)     │
  │   Proof not attempted    → UNVERIFIABLE       (trust varies)    │
  └──────────────────────────────────────────────────────────────────┘
```

### Quick Start

```python
import sys
sys.path.insert(0, "/home/ubuntu/AutoConjecture/src")
sys.path.insert(0, "/home/ubuntu/CoTShield")

from alethia import make_auditor

auditor = make_auditor(with_prover=True)

verdict = auditor.audit(
    reasoning="x + 0 = 0 because addition is symmetric",
    output="Therefore: forall x. x + 0 = x",
)
print(verdict.summary())
# Label      : HIDDEN_REASONING
# Trust score: 0.21
```

### Auditing Aletheia's Own Propaganda Analysis

```python
import httpx
from alethia.pipeline import PropagandaAuditPipeline

pipeline = PropagandaAuditPipeline()

text = "..."  # media article
response = httpx.post("http://localhost:8000/analyze", json={"text": text}).json()

verdict = pipeline.audit_analysis(text, response)
if verdict.is_hidden_reasoning():
    print("WARNING: Propaganda score may be correct but explanation is unreliable.")
    print(verdict.summary())
```

### Demos

```bash
# Demo 1: Mathematical reasoning (no API key needed)
python examples/demo_math_reasoning.py

# Demo 2: Propaganda audit (no API key needed)
python examples/demo_propaganda.py

# Tests
python -m pytest tests/ -v
```

Requires sibling repos at `/home/ubuntu/CoTShield` and `/home/ubuntu/AutoConjecture`.