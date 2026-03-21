import os
import json
import sys
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from anthropic import Anthropic
from dotenv import load_dotenv
from typing import Optional

load_dotenv(os.path.expanduser("~/Desktop/.env"))

# ---------------------------------------------------------------------------
# Sibling repo paths (CoTShield + AutoConjecture)
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "..", "CoTShield"))
sys.path.insert(0, os.path.join(_ROOT, "..", "AutoConjecture"))

app = FastAPI(title="Aletheia")
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ---------------------------------------------------------------------------
# Lazy-loaded subsystems — initialised on first request so startup is instant
# ---------------------------------------------------------------------------
_audit_pipeline = None
_campaign_detector = None
_adversary = None


def _get_audit_pipeline():
    global _audit_pipeline
    if _audit_pipeline is None:
        try:
            from alethia.pipeline import PropagandaAuditPipeline
            _audit_pipeline = PropagandaAuditPipeline()
        except Exception:
            pass
    return _audit_pipeline


def _get_campaign_detector():
    global _campaign_detector
    if _campaign_detector is None:
        try:
            from alethia.campaign import CampaignDetector
            _campaign_detector = CampaignDetector()
        except Exception:
            pass
    return _campaign_detector


def _get_adversary():
    global _adversary
    if _adversary is None:
        try:
            from alethia.adversary import AdversarialCritique
            _adversary = AdversarialCritique(client=client)
        except Exception:
            pass
    return _adversary


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Aletheia, an expert propaganda and media manipulation analyst.
Analyze the provided text for propaganda techniques, emotional manipulation, and narrative framing.

Return ONLY valid JSON with this exact structure (no markdown, no extra text):
{
  "propaganda_score": <integer 0-10, where 0=no propaganda, 10=extreme propaganda>,
  "verdict": "<one sentence overall assessment>",
  "rhetorical_techniques": [
    {
      "technique": "<technique name>",
      "description": "<what it is>",
      "example": "<short quoted passage from the text>"
    }
  ],
  "emotional_manipulation": {
    "primary_emotion": "<main emotion being targeted>",
    "secondary_emotions": ["<emotion>"],
    "intensity": "<low|medium|high>",
    "analysis": "<brief explanation>"
  },
  "narrative_framing": {
    "core_narrative": "<what story is being told>",
    "us_vs_them": <true|false>,
    "scapegoating": <true|false>,
    "false_urgency": <true|false>,
    "analysis": "<brief explanation>"
  },
  "key_passages": [
    {
      "passage": "<quoted text>",
      "concern": "<why this passage is manipulative>"
    }
  ],
  "summary": "<2-3 sentence plain language summary of findings>"
}

Be precise and evidence-based. If the text shows no propaganda, say so honestly with a low score."""


# ---------------------------------------------------------------------------
# Core analysis logic (shared by /analyze and /analyze-url)
# ---------------------------------------------------------------------------

def _run_analysis(text: str, source_url: Optional[str] = None) -> dict:
    """
    Run the full Aletheia pipeline on `text`:
      1. Claude (extended thinking) — propaganda analysis
      2. AdversarialCritique         — challenge the analysis
      3. PropagandaAuditPipeline     — dual-layer trust audit on real CoT
      4. CampaignDetector            — cross-document narrative matching
    """
    # ------------------------------------------------------------------
    # Step 1: Claude with extended thinking
    # ------------------------------------------------------------------
    thinking_text = ""
    try:
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=12000,
            thinking={"type": "enabled", "budget_tokens": 8000},
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Analyze this text for propaganda:\n\n{text}",
            }],
        )
        # Extract thinking block and text block
        for block in message.content:
            if block.type == "thinking":
                thinking_text = block.thinking
            elif block.type == "text":
                result = json.loads(block.text)
    except (AttributeError, TypeError):
        # Model doesn't support extended thinking — fall back to standard call
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Analyze this text for propaganda:\n\n{text}",
            }],
        )
        result = json.loads(message.content[0].text)

    # ------------------------------------------------------------------
    # Step 2: Adversarial critique
    # ------------------------------------------------------------------
    adversary_critique = None
    adversary = _get_adversary()
    if adversary is not None:
        try:
            adversary_critique = adversary.critique(text, result)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Step 3: Dual-layer reasoning audit
    # ------------------------------------------------------------------
    audit_dict = None
    pipeline = _get_audit_pipeline()
    if pipeline is not None:
        try:
            if thinking_text:
                verdict = pipeline.audit_with_thinking(
                    text, result, thinking_text, adversary_critique
                )
            else:
                verdict = pipeline.audit_analysis(text, result)

            audit_dict = {
                "label": verdict.label.value,
                "trust_score": round(verdict.trust_score, 3),
                "cot_risk": round(verdict.cot.risk_score, 3),
                "cot_flags": verdict.cot.flag_count,
                "explanation": verdict.explanation,
                "prover_result": verdict.prover.result,
                "prover_steps": verdict.prover.proof_steps,
                "adversary": adversary_critique,
            }
        except Exception as e:
            audit_dict = {"error": str(e)}

    # ------------------------------------------------------------------
    # Step 4: Campaign detection
    # ------------------------------------------------------------------
    campaign_dict = None
    detector = _get_campaign_detector()
    if detector is not None:
        try:
            detector.add_document(
                text,
                propaganda_score=result.get("propaganda_score", 0),
                source_url=source_url,
            )
            campaign_result = detector.find_similar(text)
            campaign_dict = campaign_result.to_dict()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Assemble response
    # ------------------------------------------------------------------
    if audit_dict:
        result["audit"] = audit_dict
    if campaign_dict:
        result["campaign"] = campaign_dict

    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    text: str


class UrlRequest(BaseModel):
    url: str


@app.post("/analyze")
async def analyze(req: AnalysisRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    if len(req.text) > 10000:
        raise HTTPException(status_code=400, detail="Text too long (max 10,000 characters)")

    try:
        result = await asyncio.to_thread(_run_analysis, req.text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse analysis result")

    return result


@app.post("/analyze-url")
async def analyze_url(req: UrlRequest):
    """Fetch an article by URL and analyze it for propaganda."""
    try:
        import trafilatura
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="trafilatura not installed. Run: pip install trafilatura",
        )

    try:
        downloaded = trafilatura.fetch_url(req.url, timeout=15)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e}")

    if not downloaded:
        raise HTTPException(status_code=400, detail="Could not fetch content from URL")

    metadata = trafilatura.extract(downloaded, output_format="python", favor_precision=True)
    if not metadata or not metadata.get("text"):
        raise HTTPException(status_code=400, detail="Could not extract article text from URL")

    text = metadata["text"]
    if len(text) > 10000:
        text = text[:10000]

    try:
        result = await asyncio.to_thread(_run_analysis, text, req.url)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse analysis result")

    result["source"] = {
        "url": req.url,
        "title": metadata.get("title"),
        "author": metadata.get("author"),
        "date": str(metadata.get("date") or ""),
        "hostname": metadata.get("hostname"),
    }
    return result


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aletheia — Propaganda Detection</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: #0d1117;
    color: #c9d1d9;
    min-height: 100vh;
    line-height: 1.6;
  }

  header {
    border-bottom: 1px solid #21262d;
    padding: 20px 40px;
    display: flex;
    align-items: baseline;
    gap: 16px;
  }

  header h1 { font-size: 1.5rem; font-weight: 700; color: #f0f6fc; letter-spacing: -0.5px; }
  header span { font-size: 0.85rem; color: #6e7681; }

  .container { max-width: 900px; margin: 40px auto; padding: 0 24px; }

  .input-tabs { display: flex; gap: 4px; margin-bottom: 12px; }

  .tab-btn {
    padding: 6px 16px;
    background: transparent;
    color: #8b949e;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.15s;
  }
  .tab-btn.active { background: #21262d; color: #f0f6fc; border-color: #8b949e; }
  .tab-btn:hover:not(.active) { border-color: #8b949e; }

  .input-pane { display: none; }
  .input-pane.active { display: block; }

  .input-section label {
    display: block;
    font-size: 0.85rem;
    color: #8b949e;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  textarea {
    width: 100%;
    height: 180px;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px;
    color: #c9d1d9;
    font-size: 0.95rem;
    font-family: inherit;
    resize: vertical;
    outline: none;
    transition: border-color 0.2s;
  }
  textarea:focus { border-color: #388bfd; }
  textarea::placeholder { color: #484f58; }

  .url-input {
    width: 100%;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 14px;
    color: #c9d1d9;
    font-size: 0.95rem;
    font-family: inherit;
    outline: none;
    transition: border-color 0.2s;
  }
  .url-input:focus { border-color: #388bfd; }
  .url-input::placeholder { color: #484f58; }

  .btn {
    margin-top: 12px;
    padding: 10px 24px;
    background: #238636;
    color: #fff;
    border: none;
    border-radius: 6px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.2s;
  }
  .btn:hover { background: #2ea043; }
  .btn:disabled { background: #21262d; color: #484f58; cursor: not-allowed; }

  .sample-btn {
    margin-top: 12px;
    margin-left: 10px;
    padding: 10px 18px;
    background: transparent;
    color: #8b949e;
    border: 1px solid #30363d;
    border-radius: 6px;
    font-size: 0.9rem;
    cursor: pointer;
    transition: all 0.2s;
  }
  .sample-btn:hover { border-color: #8b949e; color: #c9d1d9; }

  #loading {
    display: none;
    margin-top: 32px;
    text-align: center;
    color: #8b949e;
    font-size: 0.95rem;
  }
  .spinner {
    width: 32px; height: 32px;
    border: 3px solid #21262d;
    border-top-color: #388bfd;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 12px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  #results { display: none; margin-top: 32px; }

  .score-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 24px;
    display: flex;
    align-items: center;
    gap: 24px;
    margin-bottom: 24px;
  }
  .score-badge {
    width: 80px; height: 80px;
    border-radius: 50%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    font-weight: 700; font-size: 1.8rem;
    flex-shrink: 0; border: 3px solid;
  }
  .score-label { font-size: 0.65rem; font-weight: 600; letter-spacing: 0.5px; margin-top: 2px; }
  .score-low    { color: #3fb950; border-color: #3fb950; background: rgba(63,185,80,0.1); }
  .score-medium { color: #d29922; border-color: #d29922; background: rgba(210,153,34,0.1); }
  .score-high   { color: #f85149; border-color: #f85149; background: rgba(248,81,73,0.1); }
  .score-info h2 { font-size: 1.1rem; color: #f0f6fc; margin-bottom: 6px; }
  .score-info p  { font-size: 0.9rem; color: #8b949e; }

  .meter { width: 100%; height: 6px; background: #21262d; border-radius: 3px; margin-top: 10px; overflow: hidden; }
  .meter-fill { height: 100%; border-radius: 3px; transition: width 0.6s ease; }

  .section {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 16px;
  }
  .section h3 {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #8b949e;
    margin-bottom: 16px;
    font-weight: 600;
  }

  /* Audit panel */
  .audit-panel {
    border-color: #388bfd44;
  }
  .audit-panel h3 { color: #79c0ff; }

  .trust-row { display: flex; align-items: center; gap: 16px; margin-bottom: 14px; }

  .trust-badge {
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    flex-shrink: 0;
  }
  .trust-TRUSTWORTHY     { background: rgba(63,185,80,0.15);  color: #3fb950; border: 1px solid #3fb95044; }
  .trust-HIDDEN_REASONING{ background: rgba(248,81,73,0.15);  color: #f85149; border: 1px solid #f8514944; }
  .trust-HONEST_FAILURE  { background: rgba(56,139,253,0.15); color: #79c0ff; border: 1px solid #388bfd44; }
  .trust-UNRELIABLE      { background: rgba(210,153,34,0.15); color: #d29922; border: 1px solid #d2992244; }
  .trust-UNVERIFIABLE    { background: rgba(110,118,129,0.15);color: #8b949e; border: 1px solid #8b949e44; }

  .trust-score-mini { font-size: 0.9rem; color: #8b949e; }
  .trust-score-mini strong { color: #c9d1d9; }

  .audit-explanation {
    font-size: 0.88rem;
    color: #8b949e;
    line-height: 1.6;
    background: #0d1117;
    padding: 12px;
    border-radius: 6px;
    margin-top: 8px;
  }

  .adversary-toggle {
    margin-top: 12px;
    background: transparent;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 7px 14px;
    font-size: 0.83rem;
    color: #8b949e;
    cursor: pointer;
    transition: all 0.15s;
  }
  .adversary-toggle:hover { border-color: #8b949e; color: #c9d1d9; }

  .adversary-detail {
    display: none;
    margin-top: 12px;
    border-top: 1px solid #21262d;
    padding-top: 12px;
  }
  .adversary-detail.open { display: block; }

  .challenge-item {
    padding: 8px 12px;
    margin-bottom: 8px;
    border-radius: 6px;
    font-size: 0.85rem;
  }
  .challenge-valid         { background: rgba(63,185,80,0.08);  border-left: 3px solid #3fb950; }
  .challenge-over_interpreted { background: rgba(210,153,34,0.08); border-left: 3px solid #d29922; }
  .challenge-unsupported   { background: rgba(248,81,73,0.08);  border-left: 3px solid #f85149; }
  .challenge-plausible     { background: rgba(56,139,253,0.08); border-left: 3px solid #388bfd; }

  .challenge-item strong { color: #f0f6fc; display: block; margin-bottom: 3px; }

  /* Campaign alert */
  .campaign-alert {
    border-color: #d2992244;
    background: rgba(210,153,34,0.05);
  }
  .campaign-alert h3 { color: #d29922; }
  .campaign-match {
    background: #0d1117;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.85rem;
  }
  .campaign-match .sim { color: #d29922; font-weight: 600; margin-right: 8px; }
  .campaign-match .snip { color: #8b949e; font-style: italic; }

  .technique {
    border-left: 3px solid #f85149;
    padding: 10px 14px;
    margin-bottom: 12px;
    background: rgba(248,81,73,0.05);
    border-radius: 0 6px 6px 0;
  }
  .technique strong { color: #f0f6fc; font-size: 0.95rem; }
  .technique .desc  { font-size: 0.85rem; color: #8b949e; margin: 4px 0; }
  .technique .quote { font-size: 0.85rem; color: #c9d1d9; font-style: italic; background: #0d1117; padding: 6px 10px; border-radius: 4px; margin-top: 6px; }

  .tag { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 500; margin: 3px; }
  .tag-red    { background: rgba(248,81,73,0.15);  color: #f85149; }
  .tag-yellow { background: rgba(210,153,34,0.15); color: #d29922; }
  .tag-blue   { background: rgba(56,139,253,0.15); color: #79c0ff; }
  .tag-green  { background: rgba(63,185,80,0.15);  color: #3fb950; }

  .flag-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px; }
  .flag { padding: 4px 12px; border-radius: 4px; font-size: 0.82rem; font-weight: 600; }
  .flag-true  { background: rgba(248,81,73,0.15); color: #f85149; }
  .flag-false { background: rgba(63,185,80,0.15); color: #3fb950; }

  .passage {
    border-left: 3px solid #d29922;
    padding: 10px 14px;
    margin-bottom: 12px;
    background: rgba(210,153,34,0.05);
    border-radius: 0 6px 6px 0;
  }
  .passage blockquote { font-style: italic; color: #c9d1d9; font-size: 0.9rem; }
  .passage .concern   { font-size: 0.82rem; color: #8b949e; margin-top: 6px; }

  .summary-text {
    font-size: 0.95rem; color: #c9d1d9; line-height: 1.7;
    background: #0d1117; padding: 14px; border-radius: 6px;
  }
  .em-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .em-item label { font-size: 0.75rem; color: #6e7681; text-transform: uppercase; letter-spacing: 0.4px; }
  .em-item .val  { font-size: 0.95rem; color: #c9d1d9; margin-top: 2px; }

  .source-bar {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 20px;
    font-size: 0.85rem;
    color: #8b949e;
  }
  .source-bar a { color: #79c0ff; text-decoration: none; }
  .source-bar a:hover { text-decoration: underline; }

  @media (max-width: 600px) {
    .score-card { flex-direction: column; text-align: center; }
    .em-grid    { grid-template-columns: 1fr; }
    header      { padding: 16px 20px; flex-direction: column; gap: 4px; }
    .container  { padding: 0 16px; }
    .trust-row  { flex-direction: column; align-items: flex-start; }
  }
</style>
</head>
<body>

<header>
  <h1>Aletheia</h1>
  <span>Open-source Propaganda Detection + Reasoning Audit</span>
</header>

<div class="container">
  <div class="input-section">

    <div class="input-tabs">
      <button class="tab-btn active" onclick="switchTab('text', this)">Paste text</button>
      <button class="tab-btn" onclick="switchTab('url', this)">Analyze URL</button>
    </div>

    <div id="pane-text" class="input-pane active">
      <label>Article, speech, or media text</label>
      <textarea id="inputText" placeholder="Paste any media content here — news article, political speech, social media post, broadcast transcript..."></textarea>
      <div>
        <button class="btn" onclick="analyzeText()">Analyze</button>
        <button class="sample-btn" onclick="loadSample()">Load sample text</button>
      </div>
    </div>

    <div id="pane-url" class="input-pane">
      <label>Article URL</label>
      <input type="url" class="url-input" id="inputUrl" placeholder="https://example.com/article">
      <div>
        <button class="btn" onclick="analyzeUrl()">Fetch &amp; Analyze</button>
      </div>
    </div>

  </div>

  <div id="loading">
    <div class="spinner"></div>
    Analyzing content...
  </div>

  <div id="results">

    <div id="sourceBar" class="source-bar" style="display:none"></div>

    <div class="score-card">
      <div class="score-badge" id="scoreBadge">
        <span id="scoreNum">—</span>
        <span class="score-label">/ 10</span>
      </div>
      <div class="score-info" style="flex:1">
        <h2 id="verdictText"></h2>
        <div class="meter">
          <div class="meter-fill" id="meterFill"></div>
        </div>
      </div>
    </div>

    <!-- Audit panel -->
    <div class="section audit-panel" id="auditSection" style="display:none">
      <h3>Reasoning Audit</h3>
      <div class="trust-row">
        <span class="trust-badge" id="trustBadge"></span>
        <span class="trust-score-mini">
          Trust <strong id="trustScore"></strong> &nbsp;·&nbsp;
          CoT risk <strong id="cotRisk"></strong> &nbsp;·&nbsp;
          <span id="cotFlags"></span>
        </span>
      </div>
      <div class="audit-explanation" id="auditExplanation"></div>
      <div id="adversaryBlock"></div>
    </div>

    <!-- Campaign alert -->
    <div class="section campaign-alert" id="campaignSection" style="display:none">
      <h3>⚠ Coordinated Campaign Detected</h3>
      <div id="campaignMatches"></div>
    </div>

    <div class="section" id="summarySection">
      <h3>Summary</h3>
      <div class="summary-text" id="summaryText"></div>
    </div>

    <div class="section" id="techniquesSection">
      <h3>Rhetorical Techniques Detected</h3>
      <div id="techniquesList"></div>
    </div>

    <div class="section">
      <h3>Emotional Manipulation</h3>
      <div class="em-grid">
        <div class="em-item">
          <label>Primary Emotion Targeted</label>
          <div class="val" id="emPrimary"></div>
        </div>
        <div class="em-item">
          <label>Intensity</label>
          <div class="val" id="emIntensity"></div>
        </div>
        <div class="em-item" style="grid-column: span 2">
          <label>Secondary Emotions</label>
          <div id="emSecondary" style="margin-top:6px"></div>
        </div>
        <div class="em-item" style="grid-column: span 2">
          <label>Analysis</label>
          <div class="val" id="emAnalysis" style="color:#8b949e; font-size:0.88rem"></div>
        </div>
      </div>
    </div>

    <div class="section">
      <h3>Narrative Framing</h3>
      <div style="margin-bottom:10px; font-size:0.95rem; color:#c9d1d9" id="coreNarrative"></div>
      <div class="flag-row">
        <div>
          <span style="font-size:0.8rem; color:#6e7681; margin-right:6px">Us vs. Them</span>
          <span class="flag" id="flagUsVsThem"></span>
        </div>
        <div>
          <span style="font-size:0.8rem; color:#6e7681; margin-right:6px">Scapegoating</span>
          <span class="flag" id="flagScapegoat"></span>
        </div>
        <div>
          <span style="font-size:0.8rem; color:#6e7681; margin-right:6px">False Urgency</span>
          <span class="flag" id="flagUrgency"></span>
        </div>
      </div>
      <div style="margin-top:12px; font-size:0.85rem; color:#8b949e" id="narrativeAnalysis"></div>
    </div>

    <div class="section" id="passagesSection">
      <h3>Key Passages</h3>
      <div id="passagesList"></div>
    </div>

  </div>
</div>

<script>
const SAMPLE_TEXT = `The radical left elites have declared war on your family. While hardworking patriots struggle to afford groceries,
these corrupt globalists are flooding our borders and handing YOUR tax dollars to illegal aliens.
Our children are being indoctrinated in schools taken over by extremists who hate America.
Every single day they remain in power, our nation inches closer to total collapse.
Real Americans know the truth: this is our last chance to save the country our forefathers bled for.
The mainstream media won't tell you this because they are part of the conspiracy.
Stand up now—before it's too late—or lose everything forever.`;

let activeTab = 'text';

function switchTab(tab, btn) {
  activeTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.input-pane').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('pane-' + tab).classList.add('active');
}

function loadSample() {
  document.getElementById('inputText').value = SAMPLE_TEXT;
  switchTab('text', document.querySelector('.tab-btn'));
}

function setLoading(on) {
  document.getElementById('results').style.display = 'none';
  document.getElementById('loading').style.display = on ? 'block' : 'none';
  document.querySelectorAll('.btn').forEach(b => b.disabled = on);
}

async function analyzeText() {
  const text = document.getElementById('inputText').value.trim();
  if (!text) return;
  setLoading(true);
  try {
    const res = await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Analysis failed'); }
    renderResults(await res.json(), null);
  } catch (e) { alert('Error: ' + e.message); }
  finally { setLoading(false); }
}

async function analyzeUrl() {
  const url = document.getElementById('inputUrl').value.trim();
  if (!url) return;
  setLoading(true);
  try {
    const res = await fetch('/analyze-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Analysis failed'); }
    const data = await res.json();
    renderResults(data, data.source || null);
  } catch (e) { alert('Error: ' + e.message); }
  finally { setLoading(false); }
}

function scoreClass(s) {
  if (s <= 3) return 'score-low';
  if (s <= 6) return 'score-medium';
  return 'score-high';
}
function meterColor(s) {
  if (s <= 3) return '#3fb950';
  if (s <= 6) return '#d29922';
  return '#f85149';
}
function intensityTag(i) {
  const map = { low: 'tag-green', medium: 'tag-yellow', high: 'tag-red' };
  return `<span class="tag ${map[i] || 'tag-blue'}">${i}</span>`;
}

function renderAudit(audit) {
  if (!audit || audit.error) {
    document.getElementById('auditSection').style.display = 'none';
    return;
  }
  const label = audit.label || 'UNVERIFIABLE';
  const labelMap = {
    trustworthy: 'TRUSTWORTHY',
    hidden_reasoning: 'HIDDEN REASONING',
    honest_failure: 'HONEST FAILURE',
    unreliable: 'UNRELIABLE',
    unverifiable: 'UNVERIFIABLE',
  };
  const trustBadge = document.getElementById('trustBadge');
  trustBadge.textContent = labelMap[label] || label.toUpperCase();
  trustBadge.className = 'trust-badge trust-' + label.toUpperCase();

  document.getElementById('trustScore').textContent = (audit.trust_score * 100).toFixed(0) + '%';
  document.getElementById('cotRisk').textContent = (audit.cot_risk * 100).toFixed(0) + '%';
  document.getElementById('cotFlags').textContent = audit.cot_flags + ' CoT flag(s)';
  document.getElementById('auditExplanation').textContent = audit.explanation || '';

  // Adversary block
  const ab = document.getElementById('adversaryBlock');
  ab.innerHTML = '';
  const adv = audit.adversary;
  if (adv && adv.technique_challenges && adv.technique_challenges.length) {
    const btn = document.createElement('button');
    btn.className = 'adversary-toggle';
    btn.textContent = `Adversarial review: ${adv.overall_assessment || '?'} (score adjustment: ${adv.confidence_adjustment >= 0 ? '+' : ''}${adv.confidence_adjustment})`;
    ab.appendChild(btn);

    const detail = document.createElement('div');
    detail.className = 'adversary-detail';
    if (adv.summary) {
      const p = document.createElement('p');
      p.style.cssText = 'font-size:0.88rem;color:#8b949e;margin-bottom:10px';
      p.textContent = adv.summary;
      detail.appendChild(p);
    }
    adv.technique_challenges.forEach(ch => {
      const d = document.createElement('div');
      d.className = 'challenge-item challenge-' + (ch.verdict || 'plausible');
      d.innerHTML = `<strong>${ch.technique}</strong>${ch.reason || ''}`;
      detail.appendChild(d);
    });
    if (adv.missed_techniques && adv.missed_techniques.length) {
      const d = document.createElement('div');
      d.style.cssText = 'font-size:0.83rem;color:#8b949e;margin-top:8px';
      d.textContent = 'Missed: ' + adv.missed_techniques.join(', ');
      detail.appendChild(d);
    }
    ab.appendChild(detail);
    btn.addEventListener('click', () => detail.classList.toggle('open'));
  }

  document.getElementById('auditSection').style.display = 'block';
}

function renderCampaign(campaign) {
  const sec = document.getElementById('campaignSection');
  if (!campaign || !campaign.is_campaign || !campaign.matches.length) {
    sec.style.display = 'none';
    return;
  }
  const list = document.getElementById('campaignMatches');
  list.innerHTML = `<p style="font-size:0.88rem;color:#d29922;margin-bottom:12px">
    ${campaign.matches.length} similar document(s) found — possible coordinated narrative
    (confidence: ${(campaign.campaign_score * 100).toFixed(0)}%)</p>`;
  campaign.matches.slice(0, 3).forEach(m => {
    list.innerHTML += `<div class="campaign-match">
      <span class="sim">${(m.similarity * 100).toFixed(0)}% similar</span>
      ${m.source_url ? `<a href="${m.source_url}" target="_blank" style="color:#79c0ff;font-size:0.82rem;margin-right:8px">${m.source_url}</a>` : ''}
      <span class="snip">"${m.snippet}"</span>
    </div>`;
  });
  sec.style.display = 'block';
}

function renderResults(d, source) {
  // Source bar (URL analysis)
  const sb = document.getElementById('sourceBar');
  if (source && source.url) {
    sb.innerHTML = `<strong style="color:#c9d1d9">${source.title || 'Article'}</strong>
      &nbsp;·&nbsp; <a href="${source.url}" target="_blank">${source.hostname || source.url}</a>
      ${source.author ? `&nbsp;·&nbsp; ${source.author}` : ''}
      ${source.date ? `&nbsp;·&nbsp; ${source.date}` : ''}`;
    sb.style.display = 'block';
  } else {
    sb.style.display = 'none';
  }

  // Score
  const score = d.propaganda_score;
  const cls = scoreClass(score);
  const badge = document.getElementById('scoreBadge');
  badge.className = `score-badge ${cls}`;
  document.getElementById('scoreNum').textContent = score;
  document.getElementById('verdictText').textContent = d.verdict;
  const fill = document.getElementById('meterFill');
  fill.style.width = (score * 10) + '%';
  fill.style.background = meterColor(score);

  // Audit
  renderAudit(d.audit);

  // Campaign
  renderCampaign(d.campaign);

  // Summary
  document.getElementById('summaryText').textContent = d.summary;

  // Techniques
  const tl = document.getElementById('techniquesList');
  tl.innerHTML = '';
  if (d.rhetorical_techniques && d.rhetorical_techniques.length) {
    d.rhetorical_techniques.forEach(t => {
      tl.innerHTML += `<div class="technique">
        <strong>${t.technique}</strong>
        <div class="desc">${t.description}</div>
        ${t.example ? `<div class="quote">"${t.example}"</div>` : ''}
      </div>`;
    });
  } else {
    tl.innerHTML = '<div style="color:#8b949e; font-size:0.9rem">No significant rhetorical techniques detected.</div>';
  }

  // Emotional manipulation
  document.getElementById('emPrimary').textContent   = d.emotional_manipulation.primary_emotion || '—';
  document.getElementById('emIntensity').innerHTML   = intensityTag(d.emotional_manipulation.intensity);
  document.getElementById('emAnalysis').textContent  = d.emotional_manipulation.analysis || '';
  const sec = document.getElementById('emSecondary');
  sec.innerHTML = '';
  (d.emotional_manipulation.secondary_emotions || []).forEach(e => {
    sec.innerHTML += `<span class="tag tag-blue">${e}</span>`;
  });

  // Narrative framing
  document.getElementById('coreNarrative').textContent      = d.narrative_framing.core_narrative || '';
  document.getElementById('narrativeAnalysis').textContent  = d.narrative_framing.analysis || '';
  function setFlag(id, val) {
    const el = document.getElementById(id);
    el.textContent = val ? 'Yes' : 'No';
    el.className   = `flag ${val ? 'flag-true' : 'flag-false'}`;
  }
  setFlag('flagUsVsThem',  d.narrative_framing.us_vs_them);
  setFlag('flagScapegoat', d.narrative_framing.scapegoating);
  setFlag('flagUrgency',   d.narrative_framing.false_urgency);

  // Key passages
  const pl = document.getElementById('passagesList');
  pl.innerHTML = '';
  if (d.key_passages && d.key_passages.length) {
    d.key_passages.forEach(p => {
      pl.innerHTML += `<div class="passage">
        <blockquote>"${p.passage}"</blockquote>
        <div class="concern">${p.concern}</div>
      </div>`;
    });
  } else {
    pl.innerHTML = '<div style="color:#8b949e; font-size:0.9rem">No specific key passages flagged.</div>';
  }

  document.getElementById('results').style.display = 'block';
  document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
