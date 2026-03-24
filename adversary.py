"""
Adversarial critique agent for Aletheia propaganda analysis.

Spawns a second Claude call with a skeptical system prompt to challenge the
initial analysis. The critique feeds into DualLayerAuditor as a richer signal:
instead of auditing only Claude's self-reported reasoning, we audit the full
analyst ↔ adversary dialogue — exposing cases where the analyst's stated
reasons don't survive challenge even when the final score is plausible.

Architecture
------------
  1. Aletheia (analyst)  →  structured propaganda analysis
  2. Adversary agent     →  targeted challenges: over-interpretation?
                             innocent reading? circular evidence?
  3. build_audit_reasoning() merges both into a CoT string for CoTShield

The key HIDDEN_REASONING signal: analyst reaches score 8/10, adversary finds
no logical support for the cited evidence, yet the score is pattern-verified.
Neither system alone catches this — the cross-check does.
"""

from __future__ import annotations
import json
from typing import Optional


ADVERSARY_SYSTEM = """You are a meticulous devil's advocate auditing an AI propaganda analysis.

Your role is NOT to defend propaganda — it is to ensure the analysis is intellectually rigorous.

For each claimed technique, test:
1. Does the quoted passage actually use that technique, or is it being over-interpreted?
2. Could the passage have an innocent or benign explanation?
3. Does the cited evidence logically support the claim, or is the reasoning circular?
4. Are there alternative rhetorical framings the analyst missed or mislabeled?
5. Is the propaganda score calibrated to the evidence, or inflated/deflated?

Be specific. Quote exact phrases. If a technique claim is solid, say so — do not
manufacture objections. Focus on genuine analytical weaknesses.

Return ONLY valid JSON with this exact structure (no markdown, no extra text):
{
  "overall_assessment": "agree | partially_agree | disagree",
  "confidence_adjustment": <float from -3 to +3, negative = lower the score>,
  "technique_challenges": [
    {
      "technique": "<name from analyst's list>",
      "verdict": "valid | over_interpreted | unsupported | plausible",
      "reason": "<one sentence citing specific text>"
    }
  ],
  "missed_techniques": ["<technique name>"],
  "unchallenged_score": <int 0-10, your own independent propaganda score>,
  "summary": "<2-3 sentence adversarial summary>"
}"""


class AdversarialCritique:
    """
    Runs an adversarial Claude agent to challenge a propaganda analysis.

    Parameters
    ----------
    client : anthropic.Anthropic
        Shared Anthropic client.
    model : str
        Model to use for the adversary (defaults to same model as analyst).
    """

    def __init__(self, client, model: str = "claude-sonnet-4-6"):
        self.client = client
        self.model = model

    def critique(
        self,
        text: str,
        analysis: dict,
        max_tokens: int = 1024,
    ) -> dict:
        """
        Challenge an Aletheia analysis from a skeptical perspective.

        Parameters
        ----------
        text     : original media text
        analysis : Aletheia /analyze JSON response
        max_tokens : token budget for the adversary

        Returns
        -------
        dict with fields: overall_assessment, confidence_adjustment,
        technique_challenges, missed_techniques, unchallenged_score, summary
        """
        user_content = (
            f"Original text:\n\"\"\"\n{text}\n\"\"\"\n\n"
            f"Analyst's verdict:\n{json.dumps(analysis, indent=2)}"
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=ADVERSARY_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1] if len(parts) > 1 else raw
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            return json.loads(raw)

        except (json.JSONDecodeError, Exception) as e:
            return {
                "overall_assessment": "unknown",
                "confidence_adjustment": 0.0,
                "technique_challenges": [],
                "missed_techniques": [],
                "unchallenged_score": analysis.get("propaganda_score", 0),
                "summary": f"Adversarial critique unavailable: {e}",
            }

    def build_audit_reasoning(self, analysis: dict, critique: dict) -> str:
        """
        Merge analyst reasoning + adversary critique into a single CoT string
        for CoTShield to audit.

        This is the key integration point: CoTShield now sees the full
        analyst ↔ adversary dialogue, not just the analyst's clean summary.
        The HIDDEN_REASONING signal is strongest when:
          - Adversary finds unsupported technique claims
          - But the pattern verifier confirms those techniques ARE in the text
          → The analyst reached a correct score via unjustified reasoning
        """
        parts = []

        # Analyst reasoning (from structured response)
        for t in analysis.get("rhetorical_techniques", []):
            parts.append(
                f"[ANALYST] Technique '{t.get('technique')}': "
                f"{t.get('description', '')} "
                f"Evidence: '{t.get('example', '')}'"
            )

        em = analysis.get("emotional_manipulation", {})
        if em:
            def flag(val):
                return "present" if val else "absent"
            nf = analysis.get("narrative_framing", {})
            parts.append(
                f"[ANALYST] Emotional target: {em.get('primary_emotion')} "
                f"at {em.get('intensity')} intensity. {em.get('analysis', '')}"
            )
            if nf:
                parts.append(
                    f"[ANALYST] Narrative: {nf.get('core_narrative', '')}. "
                    f"Us-vs-them {flag(nf.get('us_vs_them'))}. "
                    f"Scapegoating {flag(nf.get('scapegoating'))}. "
                    f"False urgency {flag(nf.get('false_urgency'))}."
                )

        if analysis.get("summary"):
            parts.append(f"[ANALYST] Summary: {analysis['summary']}")

        parts.append("")

        # Adversarial challenges
        for challenge in critique.get("technique_challenges", []):
            verdict = challenge.get("verdict", "unknown")
            parts.append(
                f"[ADVERSARY] '{challenge.get('technique')}' is {verdict}: "
                f"{challenge.get('reason', '')}"
            )

        adj = critique.get("confidence_adjustment", 0)
        if adj != 0:
            direction = "lower" if adj < 0 else "raise"
            parts.append(
                f"[ADVERSARY] Recommends {direction} score by {abs(adj):.1f}. "
                f"Assessment: {critique.get('overall_assessment', 'unknown')}. "
                f"{critique.get('summary', '')}"
            )
        else:
            if critique.get("summary"):
                parts.append(
                    f"[ADVERSARY] Assessment: {critique.get('overall_assessment', 'unknown')}. "
                    f"{critique['summary']}"
                )

        if critique.get("missed_techniques"):
            parts.append(
                f"[ADVERSARY] Missed techniques: "
                f"{', '.join(critique['missed_techniques'])}"
            )

        return "\n".join(parts)
