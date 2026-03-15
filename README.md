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