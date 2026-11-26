"""Build docs/Project_Guide.pdf: the whole project explained, technically and in simple words.

    python scripts/make_guide_figures.py && python docs/build_guide_pdf.py
"""
import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, Spacer, Table,
                                TableStyle)
from reportlab.platypus.doctemplate import BaseDocTemplate, Frame, PageTemplate
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG = os.path.join(ROOT, "docs", "figures")
OUT = os.path.join(ROOT, "docs", "Project_Guide.pdf")

# ------------------------------------------------------------------ fonts & styles
F = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", F + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", F + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf"))
pdfmetrics.registerFont(TTFont("DV-BI", "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf"))
pdfmetrics.registerFont(TTFont("DVM", F + "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

INK, MUTED = colors.HexColor("#1F2328"), colors.HexColor("#57606A")
BLUE, GREEN, PURPLE = colors.HexColor("#2F5D9E"), colors.HexColor("#2E7D4F"), colors.HexColor("#6A4FA3")
SIMPLE_BG, TECH_BG, NOTE_BG = colors.HexColor("#EAF5EE"), colors.HexColor("#EEF2FA"), colors.HexColor("#FFF6E0")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=24, leading=30, alignment=TA_CENTER, textColor=INK),
    "sub": ParagraphStyle("sub", fontName="DV", fontSize=12.5, leading=18, alignment=TA_CENTER, textColor=MUTED),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=17, leading=22, spaceBefore=4, spaceAfter=8, textColor=BLUE),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=12.5, leading=17, spaceBefore=10, spaceAfter=4, textColor=INK,
                        keepWithNext=1),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.6, leading=14.2, spaceAfter=5, textColor=INK),
    "bullet": ParagraphStyle("bullet", fontName="DV", fontSize=9.6, leading=14, leftIndent=14, bulletIndent=3,
                             spaceAfter=2, textColor=INK),
    "box": ParagraphStyle("box", fontName="DV", fontSize=9.4, leading=13.8, textColor=INK),
    "boxh": ParagraphStyle("boxh", fontName="DV-B", fontSize=9.4, leading=13, spaceAfter=2),
    "eq": ParagraphStyle("eq", fontName="DV", fontSize=10.2, leading=15, alignment=TA_CENTER, spaceBefore=3,
                         spaceAfter=6, textColor=INK),
    "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=8.5, leading=11.5, alignment=TA_CENTER, textColor=MUTED,
                          spaceAfter=8),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.6, leading=11.5, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="DV-B", fontSize=8.6, leading=11.5, textColor=INK),
    "code": ParagraphStyle("code", fontName="DVM", fontSize=8, leading=10.5, textColor=INK),
    "toc1": ParagraphStyle("toc1", fontName="DV", fontSize=10.5, leading=17, leftIndent=6),
}

story = []
_h1_count = [0]


def P(text, style="body"):
    story.append(Paragraph(text, S[style]))


def H1(text):
    _h1_count[0] += 1
    story.append(PageBreak())
    p = Paragraph(f"{_h1_count[0]}. {text}", S["h1"])
    p._toc = f"{_h1_count[0]}. {text}"
    story.append(p)


def H2(text):
    story.append(Paragraph(text, S["h2"]))


def BUL(items):
    for it in items:
        story.append(Paragraph(it, S["bullet"], bulletText="•"))


def _callout(title, body, bg, edge, title_color):
    paras = [Paragraph(title, ParagraphStyle("t", parent=S["boxh"], textColor=title_color))]
    for b in (body if isinstance(body, list) else [body]):
        paras.append(Paragraph(b, S["box"]))
    t = Table([[paras]], colWidths=[16.6 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("LINEBEFORE", (0, 0), (0, -1), 3, edge),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    story.append(t)
    story.append(Spacer(1, 7))


def SIMPLE(body):
    _callout("In simple words", body, SIMPLE_BG, GREEN, GREEN)


def TECH(body):
    _callout("Technical detail", body, TECH_BG, BLUE, BLUE)


def NOTE(body, title="Important"):
    _callout(title, body, NOTE_BG, colors.HexColor("#C9971C"), colors.HexColor("#8A6510"))


def EQ(text):
    P(text, "eq")


def CODE(text):
    t = Table([[Preformatted(text, S["code"])]], colWidths=[16.6 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F6F8FA")),
                           ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D0D7DE")),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 6),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.append(t)
    story.append(Spacer(1, 7))


def TABLE(rows, widths, header=True, bold_rows=()):
    data = [[Paragraph(str(c), S["cellb"] if (header and r == 0) or r in bold_rows else S["cell"])
             for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D7DE")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E7ECF3")))
    for r in bold_rows:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F3F0FA")))
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 8))


def FIGURE(name, caption, width=16.4):
    path = os.path.join(FIG, name)
    img = Image(path)
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width * cm, width * cm * ratio
    story.append(KeepTogether([img, Spacer(1, 3), Paragraph(caption, S["cap"])]))


def m(path):
    return json.load(open(os.path.join(ROOT, "results", path, "metrics.json")))


# ------------------------------------------------------------------ data for results
MAIN = m("main_stimulus4_subject")
RAND = m("main_stimulus4_random")
S6 = m("stimulus6_subject")
FVA = m("felt_va4_subject")
SYN = m("synthetic_random_all")
ABL = {k: m(f"ablation_affec_stimulus4_subject/{k}")["summary"] for k in
       ["eeg", "physio", "eye", "eeg+physio", "eeg+eye", "physio+eye", "eeg+physio+eye"]}


def pct(x):
    return f"{x * 100:.1f}%"


def pm(s, k="acc"):
    return f"{s[k][0] * 100:.1f} ± {s[k][1] * 100:.1f}%"


# ================================================================== CONTENT
# ---- cover
story.append(Spacer(1, 3.2 * cm))
P("Multi-Modal Emotion Recognition<br/>using Hybrid Deep Learning", "title")
P("CNN + LSTM/GRU + Transformer on EEG, Physiological and Eye-Tracking Signals", "sub")
story.append(Spacer(1, 0.8 * cm))
P("<b>The Complete Project Guide</b><br/>Everything from start to end — explained technically and in simple words", "sub")
story.append(Spacer(1, 1.2 * cm))
FIGURE("guide_architecture.png", "The whole system on one page: three body signals → three neural-network branches → "
       "attention-based fusion → emotion.", 16.4)
story.append(Spacer(1, 0.6 * cm))
P("Group 12 · Amrita Vishwa Vidyapeetham, Coimbatore<br/>Dodla Nithya Sai · M. Vishnu Vardhan · P. Sreekar Reddy · "
  "S. Beena Reddy", "sub")
story.append(Spacer(1, 0.4 * cm))
P("Code: github.com/Vishnu868/Multimodal-Emotion-Recognition-CNN-LSTM-Transformer", "sub")

# ---- how to read + TOC
story.append(PageBreak())
P("How to read this guide", "h1")
P("Every topic is explained twice:")
SIMPLE("A plain-language explanation with everyday analogies. Read only the green boxes and you will understand "
       "what the project does, why, and what it found.")
TECH("The precise version: equations, tensor shapes, hyper-parameters and design decisions — what you need to "
     "defend the work in a viva, reproduce it, or extend it.")
NOTE("Orange boxes flag things that are easy to get wrong or that an examiner is likely to ask about.", "Watch out")
P("Contents", "h2")
toc = TableOfContents()
toc.levelStyles = [S["toc1"]]
story.append(toc)

# ================================================================== 1. Big picture
H1("The Big Picture")
SIMPLE(["When you feel an emotion, your whole body reacts — not just your face. Your brain's electrical activity "
        "changes, your skin sweats a tiny bit more when you are excited or stressed, and your eyes move "
        "differently depending on what grabs your attention.",
        "This project builds a computer program that <b>watches three of these body signals at the same time</b> — "
        "brain waves (EEG), skin sweat and temperature (GSR), and where the eyes look (eye tracking) — and "
        "<b>guesses which emotion a person is experiencing</b>: neutral, happy, sad or angry.",
        "Think of it as a panel of three expert detectives, each looking at a different clue, who then talk to each "
        "other (that's the 'Transformer') before agreeing on a final answer."])
H2("What we did, step by step")
BUL(["<b>Found the data.</b> The original files were lost on a corrupted hard disk. From the project slides we "
     "identified the dataset — AFFEC — whose specifications match exactly (63-channel EEG at 256 Hz, GSR at "
     "~50 Hz, 16-channel eye tracking at 62 Hz, felt and perceived emotion labels) and downloaded it (≈12 GB).",
     "<b>Cleaned and cut the signals</b> into the same 5-second window for every trial (the moment the emotional "
     "video plays), split into 50 small time steps.",
     "<b>Turned each signal into a picture-like or sequence-like input</b> a neural network can read.",
     "<b>Built the hybrid network</b>: CNN (spatial patterns) + LSTM/GRU (patterns over time) per signal, and a "
     "Transformer that fuses them.",
     "<b>Tested it first on synthetic (fake) data</b> to prove the code works — 99.7% accuracy.",
     "<b>Trained and tested it on the real data</b>, always testing on people the model has never seen.",
     "<b>Measured which signals matter</b> by switching signals on and off (ablation study)."])
H2("The headline result")
TABLE([["What", "Result", "Chance level"],
       ["4 emotions, all three signals, tested on unseen people (5-fold)", pm(MAIN["summary"]), "25%"],
       ["Best signal pair: skin (physio) + eyes", pm(ABL["physio+eye"]), "25%"],
       ["6 emotions", pm(S6["summary"]), "16.7%"],
       ["Synthetic data (pipeline check)", pct(SYN["summary"]["acc"][0]), "25%"]], [9.2, 4.2, 3.2])
SIMPLE("If the computer just guessed randomly it would be right 1 time in 4 (25%). Our model is right about "
       "36 times in 100 for people it has never met before. That is clearly better than guessing (and the "
       "statistics prove it is not luck), but far from perfect — reading emotions from a few seconds of body "
       "signals of a stranger is genuinely hard, even for research systems.")
NOTE("We use the <b>real AFFEC dataset</b> for all main results. Synthetic data was only used once, to check the "
     "code works (objective 4 of the original slides).", "Real or synthetic data?")

# ================================================================== 2. Problem & objectives
H1("Problem Statement and Objectives")
P("<b>Problem.</b> Traditional emotion recognition uses a single signal — usually EEG or the face. A single signal is "
  "noisy and incomplete. Real emotions affect brain activity, body physiology and eye behaviour simultaneously. "
  "<i>How can we design a system that combines EEG, physiological and eye-tracking signals to recognise human "
  "emotions more accurately and robustly?</i>")
SIMPLE("If you only listen to someone's voice on the phone, you might misjudge their mood. If you can also see "
       "their face and their body language, you judge much better. Same idea here: more kinds of evidence → better "
       "guesses, and if one signal is noisy the others can compensate.")
H2("Objectives (from the original project slides) and how each was met")
TABLE([["#", "Objective", "How it was achieved"],
       ["1", "End-to-end deep learning model integrating EEG, physio and eye signals",
        "src/model.py — one network trained jointly from preprocessed signals to emotion"],
       ["2", "Extract spatial, temporal and cross-modal features with hybrid deep learning",
        "CNN = spatial, LSTM/GRU = temporal, Transformer = cross-modal"],
       ["3", "Transformer-based fusion to learn relationships among modalities",
        "2 Transformer layers, 4 attention heads over the 3 modality tokens"],
       ["4", "Evaluate on synthetic data and establish readiness for real data",
        "99.7% on synthetic; then full evaluation on real AFFEC data"],
       ["5", "Scalable, interpretable, reproducible architecture",
        "Attention weights returned; missing-modality masking; scripts reproduce every number"]],
      [0.8, 7.0, 8.8])

# ================================================================== 3. Background
H1("Background: The Signals")
H2("3.1 What is an emotion (for a computer)?")
SIMPLE("Psychologists often describe any emotion with two dials: <b>valence</b> (how pleasant: sad ← → happy) and "
       "<b>arousal</b> (how activated: calm ← → excited). Happy = pleasant + excited. Sad = unpleasant + calm. "
       "Angry = unpleasant + excited. Neutral/calm = in the middle or pleasant + calm.")
TECH("We use two kinds of labels. (1) <b>Categorical</b>: the emotion portrayed in the video (trial_type: neutral, "
     "happy, sad, angry, fear, disgust). (2) <b>Dimensional</b>: participants' self-ratings of valence and arousal "
     "on a 1–9 scale, both for what they <i>felt</i> and what they <i>perceived</i> in the actor. We map the felt "
     "ratings to four quadrants (split at 5) for the felt_va4 experiment.")
H2("3.2 EEG — electroencephalography (brain waves)")
SIMPLE("Your brain cells communicate with tiny electrical pulses. Electrodes on a cap pick up the combined activity "
       "of millions of them through the scalp. The signal looks like wiggly waves. Fast and slow wiggles "
       "(different 'frequency bands') are linked to different mental states — e.g. alpha waves (8–13 per second) "
       "are strong when relaxed; the balance between left and right frontal alpha is linked to positive vs "
       "negative emotion.")
TECH(["AFFEC: 63 channels, 10-10 montage, g.tec g.HIamp amplifier, 256 Hz sampling, reference = left earlobe (A1). "
      "The data are already filtered by the provider (0.5–100 Hz band-pass, 50 Hz notch).",
      "Frequency bands used: δ 1–4 Hz, θ 4–8 Hz, α 8–13 Hz, β 13–30 Hz, γ 30–45 Hz."])
H2("3.3 GSR / physiological signals")
SIMPLE("When you get excited, scared or stressed, your sweat glands open slightly — even if you don't notice. This "
       "makes your skin conduct electricity better. A small sensor on two fingers measures this "
       "('galvanic skin response' or 'electrodermal activity'). It tells us <b>how intense</b> an emotion is, but not "
       "whether it is good or bad. Skin temperature and hand movement (accelerometer) are recorded too.")
TECH("Shimmer3 GSR+ unit, ~52 Hz. Columns used: GSR_Conductance_cal (µS), Temperature_cal, "
     "Low_Noise_Accelerometer_X/Y/Z_cal. GSR responds 1–3 s after a stimulus and peaks a few seconds later — "
     "visible in our data (Fig. 5).")
H2("3.4 Eye tracking")
SIMPLE("A camera under the screen follows where your eyes point, 62 times per second. When people look at a face "
       "showing an emotion, where they look (eyes? mouth?) and how their gaze moves changes with the emotion — e.g. "
       "we tend to look at the mouth of a smiling face and at the eyes of a fearful one.")
TECH("Gazepoint GP3, 62 Hz, 16 channels: fixation point (FPOGX/Y, duration, ID, validity), left/right eye point of "
     "gaze, and the 'best' point of gaze BPOGX/BPOGY (average of both eyes when valid) — coordinates are fractions "
     "of the 1280×1024 screen. We use BPOGX/BPOGY/BPOGV.")

# ================================================================== 4. Deep learning background
H1("Background: The Deep-Learning Building Blocks")
H2("4.1 Neural network, training, loss")
SIMPLE("A neural network is a big adjustable function: numbers go in, a prediction comes out. At the start its "
       "millions of 'knobs' (weights) are random, so its guesses are random. We show it thousands of examples with "
       "the right answer; each time it is wrong we nudge every knob slightly in the direction that would have made it "
       "less wrong. After many rounds it gets better. 'How wrong' is measured by the <b>loss</b>.")
TECH(["Training minimises the cross-entropy loss with gradient descent (we use the AdamW variant):"])
EQ("L = −(1/N) Σ<sub>n</sub> Σ<sub>c</sub> y<sub>n,c</sub> · log p<sub>n,c</sub>"
   "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;W<sub>new</sub> = W<sub>old</sub> − η · ∂L/∂W")
H2("4.2 CNN — convolutional neural network (spatial patterns)")
SIMPLE("A CNN slides a small 'stencil' (e.g. 3×3) over a picture and asks at every position 'does this little "
       "pattern appear here?'. It's how computers recognise edges, then shapes, then objects. We give it a "
       "'picture' of the head (electrodes laid out as on the scalp) or a picture of where the eyes looked, so it can "
       "find patterns like 'left-front of the head is more active than right-front'.")
EQ("Y<sub>i,j,k</sub> = Σ<sub>m,n,c</sub> W<sub>m,n,c,k</sub> · X<sub>i+m, j+n, c</sub> + b<sub>k</sub>")
TECH("X = input map (for EEG: 9×9 grid with c = 5 frequency-band channels), W = learned 3×3 kernel of filter k, "
     "b<sub>k</sub> = bias. Followed by ReLU A = max(0, Y) and max-pooling P = max over a 2×2 window, which shrinks "
     "the map while keeping the strongest responses.")
H2("4.3 LSTM and GRU — recurrent networks (patterns over time)")
SIMPLE("Emotions unfold over seconds. An LSTM reads a sequence one step at a time while keeping a 'memory "
       "notebook'. At every step three little gates decide: what to <b>forget</b> from the notebook, what new "
       "information to <b>write</b>, and what to <b>show</b> as output. After reading all 50 steps, its final memory "
       "summarises the whole trial. A GRU is a slimmer LSTM with two gates instead of three.")
TECH(["LSTM at step t (x<sub>t</sub> = input, h = hidden state, C = cell state, σ = sigmoid, ⊙ = element-wise):"])
EQ("f<sub>t</sub> = σ(W<sub>f</sub>[h<sub>t−1</sub>, x<sub>t</sub>] + b<sub>f</sub>) &nbsp; "
   "i<sub>t</sub> = σ(W<sub>i</sub>[h<sub>t−1</sub>, x<sub>t</sub>] + b<sub>i</sub>) &nbsp; "
   "o<sub>t</sub> = σ(W<sub>o</sub>[h<sub>t−1</sub>, x<sub>t</sub>] + b<sub>o</sub>)")
EQ("C̃<sub>t</sub> = tanh(W<sub>C</sub>[h<sub>t−1</sub>, x<sub>t</sub>] + b<sub>C</sub>) &nbsp; "
   "C<sub>t</sub> = f<sub>t</sub> ⊙ C<sub>t−1</sub> + i<sub>t</sub> ⊙ C̃<sub>t</sub> &nbsp; "
   "h<sub>t</sub> = o<sub>t</sub> ⊙ tanh(C<sub>t</sub>)")
P("GRU (update gate z, reset gate r):")
EQ("z<sub>t</sub> = σ(W<sub>z</sub>g<sub>t</sub> + U<sub>z</sub>h<sub>t−1</sub> + b<sub>z</sub>) &nbsp; "
   "r<sub>t</sub> = σ(W<sub>r</sub>g<sub>t</sub> + U<sub>r</sub>h<sub>t−1</sub> + b<sub>r</sub>)")
EQ("h̃<sub>t</sub> = tanh(W<sub>h</sub>g<sub>t</sub> + U<sub>h</sub>(r<sub>t</sub> ⊙ h<sub>t−1</sub>) + b<sub>h</sub>)"
   " &nbsp; h<sub>t</sub> = (1 − z<sub>t</sub>) ⊙ h<sub>t−1</sub> + z<sub>t</sub> ⊙ h̃<sub>t</sub>")
H2("4.4 Transformer and attention (fusion)")
SIMPLE("After each branch has summarised its signal into a list of 128 numbers, the three summaries 'look at each "
       "other'. Attention lets each summary ask: 'which of the other signals is relevant to me right now?' and borrow "
       "information from them. Example: 'the brain looks stressed → let me check the skin: is it sweating? → yes → "
       "probably anger'. The model learns these questions and answers by itself.")
EQ("F = [f<sub>EEG</sub>; f<sub>phys</sub>; f<sub>eye</sub>] &nbsp;&nbsp; Q = FW<sub>Q</sub>, K = FW<sub>K</sub>, "
   "V = FW<sub>V</sub> &nbsp;&nbsp; Attention = softmax(QK<super>T</super> / √d<sub>k</sub>) V")
TECH("QK<super>T</super> = similarity of every token with every other token; dividing by √d<sub>k</sub> keeps the "
     "numbers in a stable range; softmax turns similarities into weights that sum to 1; multiplying by V mixes the "
     "tokens accordingly. Multi-head = 4 such attentions in parallel, each free to learn a different relationship.")
H2("4.5 Softmax classifier")
SIMPLE("The final layer gives a score to each emotion; softmax turns the scores into percentages that add to 100%. "
       "The highest percentage is the answer. E.g. [5%, 12%, 75%, 8%] → 'sad'.")
EQ("z = W<sub>c</sub>F′ + b<sub>c</sub> &nbsp;&nbsp;&nbsp; p<sub>i</sub> = e<super>z<sub>i</sub></super> / "
   "Σ<sub>j</sub> e<super>z<sub>j</sub></super>")

# ================================================================== 5. Dataset
H1("The Dataset: AFFEC")
SIMPLE(["Because the original files were lost, we had to find the dataset again. The slides described it very "
        "precisely, and only one public dataset matches every detail: <b>AFFEC</b> (Advancing Face-to-Face Emotion "
        "Communication) from the brAIn lab at the IT University of Copenhagen. It is free to use and hosted on "
        "Zenodo, a research data archive.",
        "In the experiment, 72 volunteers sat in front of a screen wearing an EEG cap, a skin sensor on their "
        "fingers and an eye tracker. They watched short videos of actors expressing emotions in a conversation and "
        "then rated how they themselves felt and what emotion they thought the actor showed."])
TABLE([["Slide description", "AFFEC", "Match"],
       ["Core metadata: demographics, trial times, felt & perceived labels", "participants.tsv, *_events.tsv, *_beh.tsv",
        "✓"],
       ["63-channel EEG, 256 Hz", "63 ch, 256 Hz (.edf)", "✓"],
       ["Skin conductance, temperature, accelerometer, 50 Hz", "Shimmer3, ~52 Hz", "✓"],
       ["16 channels of gaze and fixation points, 62 Hz", "Gazepoint, 16 ch, 62 Hz", "✓"]], [7.5, 7.0, 2.1])
H2("5.1 One trial, from start to end")
FIGURE("guide_trial_timeline.png", "Fig. 2. Timeline of one trial. We analyse the 5 seconds from the start of the "
       "emotional video and compare it with the 1 second just before (baseline).")
TECH(["Each participant did up to 4 runs × 28 trials (84 different videos, 6 emotions). The events file marks every "
      "phase: trial, first_fix, scenario, second_fix, video, last_frame_video, f_emotion_labelling, "
      "p_emotion_labelling, with onset (s) and duration.",
      "After removing the 6 participants flagged ex = 1 in participants.tsv: <b>66 participants, 5,306 trials</b> "
      "(≈900 per emotion). All three signals are present in 93.7% of trials."])
H2("5.2 Files and download")
TECH(["BIDS layout: sub-&lt;id&gt;/eeg/*_eeg.edf, sub-&lt;id&gt;/beh/*_recording-gaze_physio.tsv.gz, "
      "*_recording-gsr_physio.tsv.gz, *_beh.tsv, plus *_events.tsv. The .tsv.gz files have <i>no header row</i> — "
      "column names are in the matching .json 'sidecar'.",
      "Zenodo record 19408373 (v2) has core, EEG, gaze; the GSR files only exist in the older record 14794876 "
      "(v0.1) — the downloader takes GSR from there.",
      "scripts/download_affec.py downloads individual files out of the big zip archives using HTTP range "
      "requests, so you can fetch only some subjects (--subjects 10) and resume interrupted downloads."])
FIGURE("dataset_overview.png", "Fig. 3. Left: trials per stimulus emotion. Middle: how often each signal is "
       "available. Right: participants' felt valence vs arousal ratings — note how spread out they are.")

# ================================================================== 6. Preprocessing
H1("Preprocessing: From Raw Signals to Model Inputs")
FIGURE("guide_pipeline.png", "Fig. 4. The full data flow, file by file.")
SIMPLE("Raw recordings are long, continuous and messy, and the three devices record at different speeds. "
       "Preprocessing cuts out the important moment of every trial, lines the three signals up on the same clock, "
       "turns them into tidy fixed-size arrays and removes differences between people (e.g. some people simply "
       "have sweatier hands).")
H2("6.1 Time window and alignment")
TECH("All files share the BIDS 'onset' clock (seconds since the run started). For each trial we find the video "
     "onset t<sub>0</sub> and take [t<sub>0</sub>, t<sub>0</sub> + 5 s], divided into T = 50 frames of 0.1 s "
     "(the '50 timesteps' from the slides). Every modality is sampled at the same 50 frame centres, so the time "
     "axes line up exactly.")
H2("6.2 EEG → band-power maps")
SIMPLE("For every 0.1-second step and every electrode we measure 'how strong' each of the 5 brain-wave bands is. "
       "Then we place the electrodes on a 9×9 grid that looks like a top view of the head, so the CNN can see which "
       "areas are close to each other. Result: 50 small 'heat-pictures' of the head, each with 5 layers.")
TECH(["1) Read only the trial window ±5 s of padding (memory-safe; the first version crashed on a 250 MB file). "
      "2) Zero-phase 4th-order Butterworth band-pass into 5 bands. 3) In a 0.5 s window around each frame centre "
      "compute the <b>differential entropy</b>:"])
EQ("DE = ½ · log(2πe σ<super>2</super>)")
TECH(["For a band-limited Gaussian-like signal DE is the log of band power — a standard, robust EEG emotion "
      "feature (Duan et al., 2013). 4) Map 62 scalp electrodes onto the 9×9 grid (Fig. 6); the reference A1 is "
      "skipped; empty cells are 0. Output shape <b>(50, 5, 9, 9)</b>."])
FIGURE("guide_eeg_grid.png", "Fig. 6. The 9×9 electrode grid. Neighbouring cells are neighbouring electrodes, so a "
       "3×3 convolution looks at a small patch of scalp.", 8.5)
H2("6.3 Physio → 6 features per step")
TECH("Average within each 0.1 s frame: GSR conductance, GSR slope (first derivative — captures the rise of a "
     "response), skin temperature, accelerometer X/Y/Z. Output <b>(50, 6)</b>.")
H2("6.4 Eye → gaze heat-maps")
SIMPLE("For each 0.1-second step we draw a small blurry dot wherever the eyes looked, on a 16×16 grid covering the "
       "middle of the screen (where the actor's face is). The result is 50 tiny 'where did I look' pictures.")
TECH("Valid BPOG samples in each frame are Gaussian-splatted (σ = 1 cell) onto a 16×16 grid spanning screen "
     "coordinates 0.25–0.75 (≈90% of all gaze falls there; cropping doubled the resolution on the face), and "
     "normalised to sum 1. Output <b>(50, 1, 16, 16)</b>. A trial counts as having eye data if ≥ 50% of frames "
     "have valid gaze.")
H2("6.5 Baseline correction and normalisation")
SIMPLE("Some people always have higher brain-wave power or sweatier skin. We care about the <b>change</b> caused "
       "by the video, so we subtract each person's level just before the video. Then we rescale every feature per "
       "person so that it has average 0 and spread 1 — now everyone is on the same scale.")
EQ("x′<sub>t,f</sub> = (x<sub>t,f</sub> − μ<sub>f</sub>) / σ<sub>f</sub>")
TECH("Baseline = mean over 10 frames in the 1 s before video onset (EEG and physio). The z-score statistics "
     "μ<sub>f</sub>, σ<sub>f</sub> are computed per participant over all of that participant's trials — this uses "
     "no labels, so it is legitimate even for test participants (a form of unsupervised per-user calibration).")
H2("6.6 Missing data")
TECH("Each trial stores a 3-element availability mask (EEG, physio, eye). The main experiments keep only trials "
     "with all three (3,324 trials for the 4-class task), but the model itself can run with any subset — missing "
     "modalities are masked out of the attention.")
FIGURE("modality_examples.png", "Fig. 5. What the preprocessed data look like. Top: average alpha-band EEG map per "
       "emotion. Middle: average gaze heat-map. Bottom-left: skin conductance rises ~2.5 s after happy and angry "
       "videos (high arousal) but not after neutral or sad — exactly what physiology predicts.")

# ================================================================== 7. Model
H1("The Model in Detail")
SIMPLE("Three specialist branches, one per signal. Each branch first looks at 'space' (CNN: which head regions / "
       "which screen regions) and then at 'time' (LSTM/GRU: how it changes over the 5 seconds), and finally "
       "writes a 128-number summary. The Transformer lets the three summaries exchange information, and the "
       "classifier makes the final call.")
TABLE([["Stage", "Layers", "Output shape"],
       ["EEG input", "band DE maps", "(B, 50, 5, 9, 9)"],
       ["EEG CNN (per frame)", "Conv 5→32 3×3 + BN + ReLU → Conv 32→64 3×3 + BN + ReLU → MaxPool 2 → Linear 1024→128",
        "(B, 50, 128)"],
       ["EEG LSTM", "LSTM 128→128, keep last hidden state h<sub>T</sub>", "f<sub>EEG</sub> (B, 128)"],
       ["Physio LSTM", "2-layer LSTM 6→128 (dropout 0.3)", "f<sub>phys</sub> (B, 128)"],
       ["Eye CNN (per frame)", "scale to [0,1] → Conv 1→16 + ReLU + MaxPool → Conv 16→32 + ReLU → AdaptiveMaxPool 4 "
                               "→ flatten", "g<sub>t</sub> (B, 50, 512)"],
       ["Eye GRU", "GRU 512→128, last hidden state", "f<sub>eye</sub> (B, 128)"],
       ["Tokens", "stack + learned modality embeddings", "(B, 3, 128)"],
       ["Fusion", "2 × pre-norm Transformer layer: 4-head self-attention + feed-forward 128→512→128, "
                  "key-padding mask for missing signals", "(B, 3, 128)"],
       ["Classifier", "LayerNorm → flatten 384 → Linear → ReLU = F′ (256-D) → Linear 256→4", "(B, 4) logits"],
       ["Output", "softmax", "probabilities"]], [3.2, 10.0, 3.4])
TECH(["Total parameters: ≈1.23 million. Defaults: d_model 128, 4 heads, 2 fusion layers, dropout 0.3.",
      "Interpretability: the model can return the 3×3 attention matrix of every fusion layer "
      "(src/predict.py prints it), showing how much each modality attends to the others.",
      "Robustness: a mask removes a missing modality from attention and zeroes its token; during training "
      "'modality dropout' randomly hides one signal 10% of the time so the model learns not to depend on any "
      "single sensor."])
NOTE("The slides show F ∈ ℝ<super>256</super> after concatenation. In the implementation each modality vector is "
     "128-D and the three are kept as separate tokens for attention; the fused representation F′ that feeds the "
     "classifier is 256-D, as on the slides.", "Slide vs implementation")

# ================================================================== 8. Training & evaluation
H1("Training and Evaluation")
H2("8.1 Training recipe")
TABLE([["Setting", "Value", "Why"],
       ["Optimiser", "AdamW, lr 1e-3, weight decay 1e-2", "Robust default; weight decay fights over-fitting"],
       ["Schedule", "Cosine annealing over the epochs", "Large steps first, fine steps at the end"],
       ["Batch size / epochs", "64 / max 30, early stopping patience 8", "Stop when validation stops improving"],
       ["Loss", "Cross-entropy with class weights, label smoothing 0.05", "Fair to all classes; less over-confidence"],
       ["Regularisation", "Dropout 0.3, modality dropout 0.1, grad-clip 1.0", "Generalise to new people"],
       ["Model selection", "Best validation macro-F1", "Accuracy can hide a class being ignored"]],
      [3.6, 6.0, 7.0])
H2("8.2 How we test fairly")
SIMPLE(["The golden rule: <b>never test on data the model has studied</b>. We go further and test on <b>people</b> the "
        "model has never seen. That's like a teacher testing students on a completely new set of questions written "
        "by a different person — much harder than repeating homework questions, but it is the honest measure of "
        "whether the system would work for a new user.",
        "We also do this 5 times with different groups of test people (5-fold cross-validation) and average, so the "
        "result doesn't depend on one lucky split."])
TECH(["<b>Subject-independent</b> (main): GroupKFold by participant. Within the training participants, 15% are "
      "held out as validation for early stopping. No participant appears in two of train/val/test.",
      "<b>Subject-dependent</b> (comparison): stratified K-fold over trials — the same people appear in train and "
      "test (different trials).",
      "Metrics: accuracy; <b>macro-F1</b> (average F1 over classes — punishes ignoring a class); <b>balanced "
      "accuracy</b> (average recall per class — fair when classes are unequal); confusion matrix."])
H2("8.3 The synthetic check")
SIMPLE("Before using real data we generated fake signals where each emotion has a hidden pattern buried in a lot of "
       "noise. If the model could not find these, something would be broken. It found them (99.7%), so the code "
       "works end to end.")

# ================================================================== 9. Results
H1("Results")
H2("9.1 Main result: 4 emotions, unseen people, all three signals")
folds = [f"{f['acc'] * 100:.1f}%" for f in MAIN["folds"]]
TABLE([["Metric", "Mean ± std over 5 folds"],
       ["Accuracy", pm(MAIN["summary"])], ["Macro-F1", pm(MAIN["summary"], "f1")],
       ["Balanced accuracy", pm(MAIN["summary"], "balanced_acc")], ["Per-fold accuracy", ", ".join(folds)],
       ["Chance", "25%"]], [6, 10.6])
TECH("Pooled over all 3,324 test predictions the accuracy is 35.9%. A binomial test against p = 0.25 gives "
     "z = (0.359 − 0.25)/√(0.25·0.75/3324) ≈ 14.5 — far beyond any reasonable significance threshold.")
FIGURE("confusion_main.png", "Fig. 7. Confusion matrix (rows = true emotion, columns = prediction, row-normalised).",
       9.5)
SIMPLE("Read the table row by row: of all truly-happy trials, the model said 'happy' 36% of the time. Happy is the "
       "easiest to spot (when the model says 'happy' it is right 47% of the time). Angry, sad and neutral are often "
       "mixed up with each other — especially angry → sad.")
FIGURE("training_curves_fold0.png", "Fig. 8. Training curves (fold 0): training accuracy keeps rising while "
       "validation flattens — the gap is over-fitting, which early stopping limits.")
H2("9.2 Which signals matter? (ablation)")
FIGURE("guide_ablation.png", "Fig. 9. Accuracy for every combination of signals (same 3-fold split, unseen people).",
       13)
TABLE([["Signals", "Accuracy", "Macro-F1"]] +
      [[n, pm(ABL[k]), pm(ABL[k], "f1")] for n, k in
       [("EEG", "eeg"), ("Physio", "physio"), ("Eye", "eye"), ("EEG + Physio", "eeg+physio"),
        ("EEG + Eye", "eeg+eye"), ("Physio + Eye", "physio+eye"), ("EEG + Physio + Eye", "eeg+physio+eye")]],
      [6, 5.3, 5.3], bold_rows=(6,))
SIMPLE(["<b>Eyes are the best single clue</b> (35%), skin and brain alone are only a little above guessing (~29%).",
        "<b>Combining helps</b>: skin + eyes (36.1%) is the best of all, and adding skin to brain also helps.",
        "<b>The brain signal did not help for strangers</b>: adding EEG to skin + eyes did not improve the result. "
        "Brain waves differ a lot from person to person, and a 4-second clip is very short."])
H2("9.3 Other settings")
TABLE([["Setting", "Accuracy", "Macro-F1", "Chance"],
       ["4 emotions, subject-dependent (random split, 5-fold)", pm(RAND["summary"]), pm(RAND["summary"], "f1"), "25%"],
       ["6 emotions, subject-independent (3-fold)", pm(S6["summary"]), pm(S6["summary"], "f1"), "16.7%"],
       ["Felt valence × arousal quadrants, subject-independent",
        pm(FVA["summary"]) + f" (balanced {pct(FVA['summary']['balanced_acc'][0])})", pm(FVA["summary"], "f1"),
        "25%"]], [7.6, 4.0, 3.0, 2.0])
TECH(["Random vs subject split give the same accuracy (36.1% vs 35.9%): per-participant normalisation removes most "
      "person-specific offsets, so what remains hard is the single-trial signal itself.",
      "6 classes: 25.7% ≈ 1.5× chance; fear and disgust are hardest. Felt-emotion quadrants are unbalanced "
      "(49% are 'unpleasant + calm'), so balanced accuracy (33.3%) is the honest number."])

# ================================================================== 10. Discussion
H1("Discussion: What the Results Mean")
H2("10.1 Why isn't accuracy 90%?")
SIMPLE(["Papers reporting 90%+ usually (a) test on the <b>same people</b> they trained on, (b) use <b>long</b> "
        "emotional film clips of 1–4 minutes, or (c) classify only 2 classes. Our setting is harder on all three "
        "counts: brand-new people, 3–4-second clips, 4 emotions.",
        "Also, the label is the emotion the <b>actor portrayed</b>, which is not always what the viewer actually "
        "felt — the right-hand plot of Fig. 3 shows how differently people rated the same videos."])
H2("10.2 Why are the eyes so informative?")
TECH("Gaze partly reflects the stimulus itself: a smiling face draws the eyes to the mouth, a fearful face to the "
     "eyes. So eye tracking captures both the viewer's reaction and what is on screen. This is legitimate for "
     "recognising the portrayed emotion, but should be remembered when interpreting 'felt' emotion.")
H2("10.3 Why doesn't EEG help on new people?")
TECH("EEG-only reached > 55% training accuracy but ~28% on unseen participants — classic cross-subject over-fitting. "
     "Inter-individual EEG differences (anatomy, electrode impedance, cognitive style) are large relative to the "
     "emotion effect in a single 4-second trial. Remedies: subject-adaptation / domain-adversarial training, "
     "per-user calibration, longer windows, more data.")
H2("10.4 Limitations")
BUL(["Short stimulus windows limit slow signals (GSR peaks 2–5 s after onset).",
     "Labels = portrayed emotion; felt emotion is subjective and harder.",
     "CPU-only training with a limited hyper-parameter search.",
     "Facial video and pupil size (also in AFFEC) were not used."])
H2("10.5 Future work")
BUL(["Domain adaptation / few-shot calibration for new users to unlock EEG.",
     "Add pupil diameter and facial action units as extra branches.",
     "Cross-attention fusion (TACOformer-style) and per-emotion attention visualisation.",
     "Predict valence and arousal as continuous values (regression).", "GPU training with hyper-parameter search."])

# ================================================================== 11. Code
H1("The Code: File by File")
TABLE([["File", "What it does"],
       ["scripts/download_affec.py", "Downloads AFFEC (core, EEG, gaze, GSR) from Zenodo; partial & resumable"],
       ["scripts/preprocess_affec.py", "Raw BIDS → data/processed/affec_T50.npz (windows, features, baseline, "
                                       "z-score, labels, masks); runs subjects in parallel"],
       ["src/features.py", "Signal processing: band filters, differential entropy, 9×9 EEG grid, physio frames, "
                           "gaze heat-maps"],
       ["src/data.py", "Loads the .npz, builds labels (stimulus4/6, felt/perceived quadrants), synthetic data "
                       "generator, PyTorch Dataset with modality dropout"],
       ["src/model.py", "EEGBranch, PhysioBranch, EyeBranch, FusionLayer, MultimodalEmotionNet"],
       ["src/train.py", "Splits, training loop, early stopping, metrics, plots → outputs/&lt;run&gt;/"],
       ["src/predict.py", "Loads a checkpoint, prints probabilities and cross-modal attention"],
       ["scripts/run_ablation.py", "Trains every modality subset and writes a results table"],
       ["scripts/run_all_experiments.sh", "Reproduces every number in the report"],
       ["scripts/make_figures.py, make_guide_figures.py", "Figures for the report and this guide"],
       ["tests/test_pipeline.py", "8 automated tests: shapes, masking, features, channel mapping"],
       ["results/", "Metrics, plots and trained models from our runs"]], [5.6, 11.0])
H2("How to run everything")
CODE("pip install -r requirements.txt\n\n"
     "# quick check, no download needed\n"
     "python src/train.py --data synthetic --epochs 20 --split random\n\n"
     "# real data (~12 GB; add --subjects 10 for a small test)\n"
     "python scripts/download_affec.py --out data/raw/affec\n"
     "python scripts/preprocess_affec.py --raw data/raw/affec \\\n"
     "       --out data/processed/affec_T50.npz\n"
     "python src/train.py --data data/processed/affec_T50.npz \\\n"
     "       --labels stimulus4 --split subject --folds 5\n"
     "python scripts/run_ablation.py --data data/processed/affec_T50.npz --folds 3\n\n"
     "# use a trained model\n"
     "python src/predict.py --data data/processed/affec_T50.npz \\\n"
     "  --checkpoint results/models/fusion_eeg_physio_eye_stimulus4_subject_fold0.pt\n\n"
     "pytest -q            # run the tests")
H2("Example prediction output")
P("Real output of src/predict.py with the fold-0 model on three randomly drawn trials (illustrative only — "
  "random trials may include participants this fold trained on; the honest performance numbers are in Section 9):")
CODE("sample 1700: p = [0.45, 0.13, 0.26, 0.15] -> predicted 'neutral' (index 0), true 'neutral'\n"
     "  attention: eeg->[eeg 0.42, physio 0.27, eye 0.31]  physio->[eeg 0.45, physio 0.26, eye 0.28]\n"
     "sample 1571: p = [0.05, 0.86, 0.07, 0.02] -> predicted 'happy' (index 1), true 'happy'\n"
     "  attention: eeg->[eeg 0.29, physio 0.25, eye 0.45]  eye->[eeg 0.28, physio 0.26, eye 0.46]\n"
     "sample 2510: p = [0.18, 0.25, 0.22, 0.34] -> predicted 'angry' (index 3), true 'angry'\n"
     "  attention: physio->[eeg 0.53, physio 0.15, eye 0.32]  eye->[eeg 0.22, physio 0.61, eye 0.17]")
SIMPLE("p is the model's confidence for [neutral, happy, sad, angry]. For the 'happy' trial it is very sure (86%); "
       "for the 'angry' trial it is unsure (34% vs 25% for happy) but still right. The attention numbers show which "
       "signals each branch 'listened to' — e.g. for the happy trial every branch attended most to the eyes.")
TECH("Problems solved while building it: the GSR files exist only in an older dataset version; gaze/GSR files have "
     "no header row; one 250 MB EEG file exhausted memory (fixed by reading only per-trial windows); worker "
     "processes deadlocked under 'fork' (fixed with 'spawn'); reference electrode A1 and the extra F9/F10 needed "
     "explicit grid handling.")

# ================================================================== 12. Viva Q&A
H1("Likely Questions (Viva / Review) and Answers")
QA = [
    ("Which dataset did you use, and is it real?",
     "The real AFFEC dataset (IT University of Copenhagen, Zenodo), 66 participants after exclusions, 5,306 trials. "
     "Synthetic data was only used to validate the pipeline."),
    ("Why CNN + LSTM + Transformer and not just one?",
     "Each solves a different sub-problem: CNN = spatial layout (electrodes, gaze regions), LSTM/GRU = how the "
     "response evolves over time, Transformer = how the modalities relate. Using one architecture for all would "
     "ignore the structure of the data."),
    ("Why GRU for eyes but LSTM for EEG?",
     "The eye branch already has a large 512-D per-frame input from its CNN; the GRU has fewer parameters (2 gates) "
     "and trains faster with similar quality on short sequences."),
    ("What is differential entropy and why use it?",
     "½·log(2πeσ²) of the band-filtered signal — essentially log band power. It is the most widely used and robust "
     "EEG emotion feature (SEED benchmark)."),
    ("Why 50 timesteps?",
     "5 s window / 0.1 s per step. It matches the slides and gives enough temporal resolution for the GSR rise "
     "and gaze movements while keeping sequences short for the LSTM."),
    ("What is subject-independent evaluation and why does it matter?",
     "Test participants are never in training. It measures whether the system works for a new user — the realistic "
     "use case. Subject-dependent results are typically much more optimistic."),
    ("Your accuracy is 36%. Isn't that low?",
     "Chance is 25% and the result is highly significant (z ≈ 14.5). The task — 4 classes, 3–4 s clips, unseen "
     "people — is much harder than most published setups; we report it honestly instead of using a leaky split."),
    ("Did the fusion help?",
     "Yes: every combination beats its weakest member and the best models are multimodal (physio + eye 36.1% vs "
     "best single 35.1%, EEG 28.4%). EEG did not add information for unseen people."),
    ("How do you handle a missing sensor?",
     "A per-trial availability mask removes that modality's token from attention; training with modality dropout "
     "makes the model robust to it."),
    ("How do you avoid data leakage?",
     "Splits by participant; normalisation uses no labels; model selection only on validation participants; test "
     "participants are touched once at the end."),
    ("What is the role of softmax and cross-entropy?",
     "Softmax converts the 4 scores into probabilities; cross-entropy penalises low probability on the true class "
     "and is minimised by gradient descent."),
    ("How would you improve it?",
     "Subject adaptation for EEG, pupil and facial features, longer windows, continuous valence/arousal "
     "regression, GPU hyper-parameter search."),
]
for q, a in QA:
    story.append(KeepTogether([Paragraph(f"<b>Q: {q}</b>", S["body"]),
                               Paragraph(f"<font color='#2E7D4F'><b>A:</b></font> {a}", S["body"]), Spacer(1, 3)]))

# ================================================================== 13. Glossary
H1("Glossary")
GLOSS = [
    ("Ablation study", "Removing parts (here: signals) to see how much each contributes."),
    ("Arousal / valence", "How activated (calm→excited) / how pleasant (negative→positive) an emotion is."),
    ("Attention", "Mechanism letting a model weigh which inputs are relevant to each other."),
    ("Balanced accuracy", "Average of per-class recall; fair when classes have different sizes."),
    ("Baseline correction", "Subtracting the pre-stimulus level so only the change is analysed."),
    ("BIDS", "Brain Imaging Data Structure — a standard folder/file layout for neuro data."),
    ("Chance level", "Accuracy of random guessing (1 / number of classes for balanced data)."),
    ("Confusion matrix", "Table of true vs predicted classes."),
    ("Cross-validation (k-fold)", "Repeat train/test k times on different splits and average."),
    ("Differential entropy (DE)", "½·log(2πeσ²); log band power of an EEG band."),
    ("Dropout", "Randomly switching off neurons during training to reduce over-fitting."),
    ("Early stopping", "Stop training when validation performance stops improving."),
    ("EEG", "Electroencephalography: scalp recording of brain electrical activity."),
    ("Epoch", "One pass over the whole training set."),
    ("GSR / EDA", "Galvanic skin response / electrodermal activity: skin conductance from sweat glands."),
    ("Macro-F1", "Average over classes of the harmonic mean of precision and recall."),
    ("Modality", "A type of signal (EEG, physio, eye)."),
    ("Over-fitting", "Memorising training data instead of learning general patterns."),
    ("Softmax", "Turns scores into probabilities that sum to 1."),
    ("Subject-independent", "Evaluation where test people never appear in training."),
    ("Token", "One input vector to a Transformer (here: one per modality)."),
    ("z-score", "(x − mean) / standard deviation."),
]
TABLE([["Term", "Meaning"]] + [[a, b] for a, b in GLOSS], [4.6, 12.0])

# ================================================================== 14. References
H1("References")
for r in [
    "brAIn lab, IT University of Copenhagen. Advancing Face-to-Face Emotion Communication: A Multimodal Dataset "
    "(AFFEC). arXiv:2504.18969, 2025. Data: Zenodo records 19408373 and 14794876.",
    "W.-L. Zheng, B.-N. Dong, B.-L. Lu. Multimodal emotion recognition using EEG and eye tracking data. IEEE EMBC, "
    "2014.",
    "Iacono et al. CNN-based EEG–eye-tracking fusion with spatially arranged EEG maps. 2024.",
    "Li et al. TACOformer: token-channel compounded cross attention for multimodal emotion recognition. 2023.",
    "S. M. A. Abdullah, S. Y. A. Ameen, M. A. M. Sadeeq, S. Zeebaree. Multimodal Emotion Recognition using Deep "
    "Learning. JASTT 2(01):73–79, 2021.",
    "Ma et al. PHemoNet: a multimodal network for physiological signals. 2024.",
    "R.-N. Duan, J.-Y. Zhu, B.-L. Lu. Differential entropy feature for EEG-based emotion classification. IEEE NER, "
    "2013.",
    "S. Hochreiter, J. Schmidhuber. Long short-term memory. Neural Computation, 1997.",
    "K. Cho et al. Learning phrase representations using RNN encoder–decoder (GRU). EMNLP, 2014.",
    "A. Vaswani et al. Attention is all you need. NeurIPS, 2017.",
]:
    story.append(Paragraph(r, S["bullet"], bulletText="•"))


# ------------------------------------------------------------------ document template with TOC + page numbers
class Doc(BaseDocTemplate):
    def __init__(self, fn):
        super().__init__(fn, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm,
                         bottomMargin=2 * cm, title="Multi-Modal Emotion Recognition — Complete Project Guide",
                         author="Group 12, Amrita Vishwa Vidyapeetham")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("p", [frame], onPage=self.footer)])

    @staticmethod
    def footer(canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setFont("DV", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(2.2 * cm, 1.2 * cm, "Multi-Modal Emotion Recognition — Complete Project Guide")
        canvas.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f"{doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if hasattr(flowable, "_toc"):
            self.notify("TOCEntry", (0, flowable._toc, self.page))


if __name__ == "__main__":
    Doc(OUT).multiBuild(story)
    print("wrote", OUT)
