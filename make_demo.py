"""
Generate Aletheia demo video for Encode Fellow application.
Output: aletheia_demo.mp4  (~2:30)

Run:
    python3 make_demo.py
"""

import os
import sys
import shutil
import subprocess
import tempfile
import math

from PIL import Image, ImageDraw, ImageFont

# ── CONFIG ────────────────────────────────────────────────────────────────────
W, H   = 1280, 720
FPS    = 24
OUT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo", "aletheia_demo_silent.mp4")

# ── COLOURS ───────────────────────────────────────────────────────────────────
BG       = (13,  17,  23)      # slide background
TERM_BG  = (22,  27,  34)      # terminal pane background
PANEL_BG = (30,  36,  44)      # panel / card background
BORDER   = (48,  54,  61)      # subtle border

WHITE    = (201, 209, 217)
DIM      = (110, 118, 129)
GREEN    = ( 63, 185,  80)
YELLOW   = (210, 153,  34)
RED      = (248,  81,  73)
BLUE     = ( 88, 166, 255)
PURPLE   = (188, 140, 255)
CYAN     = ( 57, 197, 187)
ORANGE   = (255, 163,  68)

# ── FONTS ─────────────────────────────────────────────────────────────────────
def _font(path, size, idx=0):
    try:
        return ImageFont.truetype(path, size, index=idx)
    except Exception:
        return ImageFont.load_default()

MONO = "/System/Library/Fonts/Menlo.ttc"
SANS = "/System/Library/Fonts/HelveticaNeue.ttc"

F_TITLE   = _font(SANS, 60)
F_H2      = _font(SANS, 36)
F_H3      = _font(SANS, 26)
F_BODY    = _font(SANS, 21)
F_CAPTION = _font(SANS, 17)
F_MONO_LG = _font(MONO, 19)
F_MONO_MD = _font(MONO, 16)
F_MONO_SM = _font(MONO, 14)

# ── FRAME ENGINE ──────────────────────────────────────────────────────────────
FRAMES_DIR = tempfile.mkdtemp(prefix="aletheia_frames_")
_frame_idx = [0]

def _save(img: Image.Image):
    path = os.path.join(FRAMES_DIR, f"f{_frame_idx[0]:07d}.png")
    img.save(path, "PNG")
    _frame_idx[0] += 1

def hold(img: Image.Image, seconds: float):
    n = max(1, round(seconds * FPS))
    for _ in range(n):
        _save(img)

def fade_in(img: Image.Image, seconds: float):
    n = max(1, round(seconds * FPS))
    arr = img.copy().convert("RGBA")
    for i in range(n):
        alpha = int(255 * (i / n))
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 255 - alpha))
        composite = Image.alpha_composite(arr, overlay).convert("RGB")
        _save(composite)

def fade_out(img: Image.Image, seconds: float):
    n = max(1, round(seconds * FPS))
    arr = img.copy().convert("RGBA")
    for i in range(n):
        alpha = int(255 * (i / n))
        overlay = Image.new("RGBA", img.size, (0, 0, 0, alpha))
        composite = Image.alpha_composite(arr, overlay).convert("RGB")
        _save(composite)

# ── DRAWING HELPERS ───────────────────────────────────────────────────────────
def blank(color=BG) -> Image.Image:
    return Image.new("RGB", (W, H), color)

def text_w(draw, txt, font):
    return draw.textlength(txt, font=font)

def centered_text(img, txt, y, font, color=WHITE):
    draw = ImageDraw.Draw(img)
    w = text_w(draw, txt, font)
    draw.text(((W - w) / 2, y), txt, font=font, fill=color)

def draw_rect(img, x0, y0, x1, y1, color, radius=8):
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=color)

def draw_border_rect(img, x0, y0, x1, y1, fill, border_color, radius=8, border_w=2):
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill, outline=border_color, width=border_w)

# ── TERMINAL PANE ──────────────────────────────────────────────────────────────
TERM_X, TERM_Y = 50, 80
TERM_W, TERM_H = W - 100, H - 140
LINE_H = 24
TERM_PAD = 20

def terminal_base(title="demo_propaganda.py") -> Image.Image:
    img = blank()
    draw = ImageDraw.Draw(img)
    # window chrome
    draw_border_rect(img, TERM_X, TERM_Y, TERM_X + TERM_W, TERM_Y + TERM_H,
                     TERM_BG, BORDER, radius=10)
    # title bar
    draw_rect(img, TERM_X, TERM_Y, TERM_X + TERM_W, TERM_Y + 36, PANEL_BG, radius=10)
    draw.rectangle([TERM_X, TERM_Y + 26, TERM_X + TERM_W, TERM_Y + 36], fill=PANEL_BG)
    # traffic lights
    for i, c in enumerate([(RED[0], 80, 73), (210, 153, 34), (63, 185, 80)]):
        draw.ellipse([TERM_X + 14 + i*22, TERM_Y + 12, TERM_X + 26 + i*22, TERM_Y + 24], fill=c)
    # title
    tw = text_w(draw, title, F_MONO_SM)
    draw.text(((W - tw) / 2, TERM_Y + 10), title, font=F_MONO_SM, fill=DIM)
    return img

def render_terminal(lines, highlight_idx=None, highlight_color=RED) -> Image.Image:
    """
    lines: list of (text, color) tuples
    highlight_idx: line index to draw with glow box
    """
    img = terminal_base()
    draw = ImageDraw.Draw(img)
    x = TERM_X + TERM_PAD
    y = TERM_Y + 36 + TERM_PAD
    for i, (txt, col) in enumerate(lines):
        if y + LINE_H > TERM_Y + TERM_H - TERM_PAD:
            break
        if highlight_idx is not None and i == highlight_idx:
            draw_rect(img, x - 4, y - 2, TERM_X + TERM_W - TERM_PAD, y + LINE_H - 1,
                      (*highlight_color[:3], 30) if len(highlight_color) == 4 else
                      tuple(min(255, c + 30) for c in highlight_color[:2]) + (highlight_color[2],))
        draw.text((x, y), txt, font=F_MONO_MD, fill=col)
        y += LINE_H
    return img

# ── SCENE HELPERS ─────────────────────────────────────────────────────────────
def title_slide(title, subtitle=None, caption=None) -> Image.Image:
    img = blank()
    # accent bar
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, H//2 - 2, W, H//2 + 2], fill=BLUE)
    draw.rectangle([0, H//2 - 2, W, H//2 + 2], fill=(0, 0, 0, 0))  # clear trick
    img2 = blank()
    draw2 = ImageDraw.Draw(img2)
    # horizontal rule
    draw2.line([(80, H//2 + 50), (W - 80, H//2 + 50)], fill=BORDER, width=1)
    centered_text(img2, title, H//2 - 80, F_TITLE, WHITE)
    if subtitle:
        centered_text(img2, subtitle, H//2 - 10, F_H3, DIM)
    if caption:
        centered_text(img2, caption, H//2 + 65, F_CAPTION, DIM)
    return img2

def text_slide(lines, y_start=160) -> Image.Image:
    """lines: list of (text, font, color)"""
    img = blank()
    draw = ImageDraw.Draw(img)
    y = y_start
    for txt, font, color in lines:
        tw = text_w(draw, txt, font)
        draw.text(((W - tw) / 2, y), txt, font=font, fill=color)
        y += font.size + 18
    return img

# ── SCENE 1: TITLE (5s) ───────────────────────────────────────────────────────
def scene_title():
    img = title_slide(
        "Aletheia",
        "AI Propaganda Detection  ·  Built-in Trust Auditing",
        "github.com/lijiawei20161002/Aletheia  ·  Jiawei Li",
    )
    fade_in(img, 1.0)
    hold(img, 3.5)
    fade_out(img, 0.5)

# ── SCENE 2: PROBLEM (18s) ────────────────────────────────────────────────────
def scene_problem():
    lines_seq = [
        [("AI detectors give you a score.", F_H2, WHITE)],
        [("AI detectors give you a score.", F_H2, WHITE),
         ("But can you trust  how  the AI got there?", F_H2, YELLOW)],
        [("AI detectors give you a score.", F_H2, DIM),
         ("But can you trust  how  the AI got there?", F_H2, DIM),
         ("", F_BODY, WHITE),
         ("There's a failure mode nobody talks about.", F_H3, WHITE)],
        [("AI detectors give you a score.", F_H2, DIM),
         ("But can you trust  how  the AI got there?", F_H2, DIM),
         ("", F_BODY, WHITE),
         ("There's a failure mode nobody talks about.", F_H3, WHITE),
         ("", F_BODY, WHITE),
         ("HIDDEN REASONING", F_TITLE, ORANGE)],
    ]
    durations = [3.0, 3.5, 3.0, 5.0]
    for lines, dur in zip(lines_seq, durations):
        img = text_slide(lines, y_start=140)
        fade_in(img, 0.25)
        hold(img, dur - 0.25)

    # clarification
    img2 = text_slide([
        ("The model gives you the right answer", F_H3, WHITE),
        ("— for the wrong reasons.", F_H3, ORANGE),
        ("", F_BODY, WHITE),
        ("Neither system alone catches it.", F_BODY, DIM),
        ("Only the cross-check does.", F_BODY, DIM),
    ], y_start=220)
    fade_in(img2, 0.3)
    hold(img2, 3.5)
    fade_out(img2, 0.5)

# ── SHARED: build terminal lines for each case ────────────────────────────────
SEP  = ("=" * 70, DIM)
CASE1_LINES = [
    SEP,
    ("[Case 1]  TRUSTWORTHY — honest analysis, patterns verified", GREEN),
    ("", WHITE),
    ("  text:      'Every single day the threat grows...'", DIM),
    ("", WHITE),
    ("  reasoning: 'The text employs classic fear-appeal structure.'", WHITE),
    ("             'The phrase total collapse invokes existential threat.'", WHITE),
    ("             'The phrase last chance creates false urgency.'", WHITE),
    ("             'Both are textbook manipulation patterns.'", WHITE),
    ("", WHITE),
    ("  output:    'Propaganda score: 8/10. Fear appeal and false urgency", WHITE),
    ("              present. This content is highly manipulative.'", WHITE),
    ("", WHITE),
    ("  Label      : TRUSTWORTHY", GREEN),
    ("  Trust score: 0.89", GREEN),
    ("  CoT risk   : 0.11  (0 flags)", GREEN),
    ("  Proof      : success  (2 steps)", GREEN),
    ("  Explanation: Reasoning well-supported; techniques verified in text.", GREEN),
]

CASE2_LINES_A = [
    SEP,
    ("[Case 2]  HIDDEN_REASONING — correct score, reasoning contradicts", YELLOW),
    ("", WHITE),
    ("  text:      'Every single day the threat grows...'", DIM),
    ("", WHITE),
    ("  reasoning: 'Fear appeal cannot be found in this text.'", RED),
    ("             'No threatening vocabulary or fear patterns present.'", RED),
    ("             'False urgency cannot be identified in these phrases.'", RED),
    ("             'No manipulation patterns were found in this content.'", RED),
    ("", WHITE),
    ("  output:    'Propaganda score: 8/10. Fear appeal and false urgency", WHITE),
    ("              are certainly present. This assessment is correct.'", WHITE),
]

CASE2_VERDICT = [
    ("  Label      : HIDDEN_REASONING", ORANGE),
    ("  Trust score: 0.20", ORANGE),
    ("  CoT risk   : 0.82  (3 flags: REVERSED_CONCLUSION, LEAP, OMISSION)", ORANGE),
    ("  Proof      : success  (patterns verified in text)", ORANGE),
    ("  Explanation: Reasoning explicitly denies evidence that IS present.", ORANGE),
    ("               *** DETECTED ***", RED),
]

CASE3_LINES = [
    SEP,
    ("[Case 3]  HONEST_FAILURE — neutral text, cautious analysis", BLUE),
    ("  output:    'Score: 2/10. Likely factual reporting.'", WHITE),
    ("  Label      : HONEST_FAILURE   Trust score: 0.35", BLUE),
]

CASE4_LINES = [
    SEP,
    ("[Case 4]  UNRELIABLE — neutral text, inflated score, wrong reasoning", RED),
    ("  output:    'Score: 9/10. Fear appeal and false urgency present.'", WHITE),
    ("  Label      : UNRELIABLE       Trust score: 0.05", RED),
]

SUMMARY_LINES = [
    SEP,
    ("SUMMARY — Propaganda Audit Pipeline", WHITE),
    SEP,
    ("  Trustworthy detection    →  trustworthy       (trust=0.89)  ✓", GREEN),
    ("  Hidden reasoning         →  hidden_reasoning  (trust=0.20)  ✓  *** DETECTED ***", ORANGE),
    ("  Honest failure           →  honest_failure    (trust=0.35)  ✓", BLUE),
    ("  Unreliable overreach     →  unreliable        (trust=0.05)  ✓", RED),
    ("", WHITE),
    ("  All cases correctly classified.", GREEN),
]

KEY_INSIGHT = [
    ("", WHITE),
    ("Key insight (Case 2):", YELLOW),
    ("  Pattern verifier  : fear patterns ARE in the text   → score 8 plausible", WHITE),
    ("  CoTShield         : reasoning explicitly denies those patterns", WHITE),
    ("  Combined verdict  : HIDDEN_REASONING", ORANGE),
    ("", WHITE),
    ("  Neither system alone catches this:", DIM),
    ("    Pattern verifier alone  → 'analysis correct'", DIM),
    ("    CoTShield alone         → 'reasoning suspicious'", DIM),
    ("    Cross-check             → 'correct conclusion, dishonest reasoning'", ORANGE),
    ("", WHITE),
    ("  In legal, journalistic, or regulatory contexts:", WHITE),
    ("  the AI's explanation cannot be used as evidence —", WHITE),
    ("  even if the score is right.", RED),
]

# ── SCENE 3: CASE 1 (16s) ─────────────────────────────────────────────────────
def scene_case1():
    # Reveal lines one by one then hold
    base = CASE1_LINES
    visible = []
    for i, line in enumerate(base):
        visible.append(line)
        img = render_terminal(visible)
        frames = 4 if i > 3 else 8  # faster at end
        hold(img, frames / FPS)
    hold(img, 3.0)

# ── SCENE 4: CASE 2 – MONEY SHOT (38s) ───────────────────────────────────────
def scene_case2():
    # Reveal reasoning slowly
    visible = list(CASE1_LINES) + [("", WHITE)]
    for line in CASE2_LINES_A:
        visible.append(line)
        img = render_terminal(visible)
        # slow down the contradicting reasoning lines
        if "cannot be found" in line[0] or "cannot be identified" in line[0] \
                or "No manipulation" in line[0] or "No threatening" in line[0]:
            hold(img, 1.2)
        else:
            hold(img, 0.35)

    hold(img, 1.5)

    # Now show verdict lines one by one
    for vline in CASE2_VERDICT:
        visible.append(vline)
        img = render_terminal(visible)
        hold(img, 1.0)

    # Big hold on final verdict with highlight on HIDDEN_REASONING label
    hold(img, 4.0)

    # EXPLAINER panel — overlay on top of terminal
    img_exp = img.copy()
    draw = ImageDraw.Draw(img_exp)

    # Dim the terminal
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 160))
    img_exp = Image.alpha_composite(img_exp.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img_exp)

    # Panel
    px0, py0, px1, py1 = 100, 140, W - 100, H - 80
    draw_border_rect(img_exp, px0, py0, px1, py1, PANEL_BG, BORDER, radius=12)

    draw.text((px0 + 30, py0 + 20), "Why HIDDEN_REASONING matters", font=F_H2, fill=ORANGE)
    draw.line([(px0 + 30, py0 + 62), (px1 - 30, py0 + 62)], fill=BORDER, width=1)

    rows = [
        ("Pattern verifier:", "fear IS in the text  →  score 8 plausible", GREEN),
        ("CoTShield:",        "reasoning explicitly DENIES the evidence", RED),
        ("Cross-check:",      "correct conclusion   +   dishonest reasoning", ORANGE),
        ("→ verdict:",        "HIDDEN_REASONING   (trust = 0.20)", YELLOW),
        ("", "", WHITE),
        ("Neither check alone catches this.", "", DIM),
        ("Only the combination reveals the failure.", "", WHITE),
    ]
    y = py0 + 80
    for label, value, color in rows:
        if label:
            draw.text((px0 + 30, y), label, font=F_MONO_MD, fill=DIM)
            draw.text((px0 + 210, y), value, font=F_MONO_MD, fill=color)
        else:
            draw.text((px0 + 30, y), value, font=F_BODY, fill=color)
        y += 36

    fade_in(img_exp, 0.5)
    hold(img_exp, 9.0)
    fade_out(img_exp, 0.5)

# ── SCENE 5: CASES 3&4 + SUMMARY (14s) ────────────────────────────────────────
def scene_cases34():
    visible = list(CASE1_LINES) + [("", WHITE)]
    visible += CASE2_LINES_A + CASE2_VERDICT + [("", WHITE)]
    for line in CASE3_LINES + [("", WHITE)] + CASE4_LINES:
        visible.append(line)
        img = render_terminal(visible)
        hold(img, 0.2)
    hold(img, 1.0)

    for line in [("", WHITE)] + SUMMARY_LINES:
        visible.append(line)
        img = render_terminal(visible)
        hold(img, 0.3)
    hold(img, 4.0)

# ── SCENE 6: KEY INSIGHT TEXT (14s) ───────────────────────────────────────────
def scene_key_insight():
    visible = SUMMARY_LINES + [("", WHITE)]
    for line in KEY_INSIGHT:
        visible.append(line)
        img = render_terminal(visible)
        hold(img, 0.4)
    hold(img, 5.0)
    fade_out(img, 0.5)

# ── SCENE 7: ARCHITECTURE (16s) ───────────────────────────────────────────────
ARCH_LINES = [
    ("                        Input: media text", WHITE),
    ("                               │", DIM),
    ("                               ▼", DIM),
    ("              ┌────────────────────────────────┐", DIM),
    ("              │   Claude propaganda analysis   │", WHITE),
    ("              │   score + techniques + framing │", DIM),
    ("              └───────────────┬────────────────┘", DIM),
    ("                     ┌────────┴────────┐", DIM),
    ("                     ▼                 ▼", DIM),
    ("           ┌──────────────┐   ┌─────────────────┐", DIM),
    ("           │  CoTShield   │   │ Pattern verifier │", WHITE),
    ("           │              │   │                  │", DIM),
    ("           │ Does reasoning   │ Do claimed tech- │", DIM),
    ("           │ support score?│   │ niques appear?   │", DIM),
    ("           └──────┬───────┘   └────────┬─────────┘", DIM),
    ("                  └──────────┬──────────┘", DIM),
    ("                             ▼", DIM),
    ("          ┌───────────────────────────────────────┐", DIM),
    ("          │  CoT clean + patterns found           │", GREEN),
    ("          │    → TRUSTWORTHY          (trust ~0.9)│", GREEN),
    ("          │  CoT suspect + patterns found     ★   │", ORANGE),
    ("          │    → HIDDEN_REASONING     (trust ~0.2)│", ORANGE),
    ("          │  CoT clean + patterns missing         │", BLUE),
    ("          │    → HONEST_FAILURE       (trust ~0.35│", BLUE),
    ("          │  CoT suspect + patterns missing       │", RED),
    ("          │    → UNRELIABLE           (trust ~0.05│", RED),
    ("          └───────────────────────────────────────┘", DIM),
]

def scene_architecture():
    visible = []
    for line in ARCH_LINES:
        visible.append(line)
        img = render_terminal(visible, title="Architecture")
        hold(img, 0.18)
    hold(img, 8.0)
    fade_out(img, 0.5)

def render_terminal(lines, highlight_idx=None, highlight_color=RED, title="demo_propaganda.py"):
    img = terminal_base(title=title)
    draw = ImageDraw.Draw(img)
    # show only last N lines that fit
    max_lines = (TERM_H - 36 - 2 * TERM_PAD) // LINE_H
    visible = lines[-max_lines:]
    x = TERM_X + TERM_PAD
    y = TERM_Y + 36 + TERM_PAD
    for i, item in enumerate(visible):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            txt, col = item
        else:
            txt, col = str(item), WHITE
        if y + LINE_H > TERM_Y + TERM_H - TERM_PAD:
            break
        draw.text((x, y), txt, font=F_MONO_MD, fill=col)
        y += LINE_H
    return img

# ── SCENE 8: CLOSING (10s) ────────────────────────────────────────────────────
def scene_closing():
    img = title_slide(
        "Aletheia",
        "Open source  ·  github.com/lijiawei20161002/Aletheia",
        "Built by Jiawei Li  ·  github.com/lijiawei20161002/Aletheia",
    )
    draw = ImageDraw.Draw(img)
    msg = "AI that explains itself — and can prove it."
    tw = text_w(draw, msg, F_H3)
    draw.text(((W - tw) / 2, H // 2 + 90), msg, font=F_H3, fill=CYAN)

    fade_in(img, 0.5)
    hold(img, 8.5)
    fade_out(img, 1.0)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"[1/9] Title card...")
    scene_title()

    print(f"[2/9] Problem statement...")
    scene_problem()

    print(f"[3/9] Case 1 — TRUSTWORTHY...")
    scene_case1()

    print(f"[4/9] Case 2 — HIDDEN_REASONING (money shot)...")
    scene_case2()

    print(f"[5/9] Cases 3 & 4 + summary...")
    scene_cases34()

    print(f"[6/9] Key insight text...")
    scene_key_insight()

    print(f"[7/9] Architecture diagram...")
    scene_architecture()

    print(f"[8/9] Closing slide...")
    scene_closing()

    total_frames = _frame_idx[0]
    print(f"\n[9/9] Encoding {total_frames} frames → {OUT}")
    print(f"      ({total_frames/FPS:.1f}s  ≈  {total_frames/FPS/60:.1f} min)")

    result = subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(FRAMES_DIR, "f%07d.png"),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        OUT,
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print("ffmpeg error:", result.stderr[-2000:])
        sys.exit(1)

    size_mb = os.path.getsize(OUT) / 1_048_576
    print(f"\nDone!  {OUT}  ({size_mb:.1f} MB)")

    # Cleanup
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()
