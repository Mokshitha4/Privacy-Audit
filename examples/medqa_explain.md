## What this run found

### ⚪ Exact Memorization: No signal
*Whether prompting the model with the start of a real training example causes it to output the rest, word for word.*

**No exact memorization detected in the tested samples.**
- None of the tested prompts produced a verbatim 50-token match with the real training continuation.
- Average text overlap with the real continuation (ROUGE-L): 0.17 of 1.00.
- Average meaning-level similarity to the real continuation (BERTScore-F1): 0.78 of 1.00.

### 🟡 Membership Inference: Weak signal
*Whether an attacker who can query the model can tell which examples were in its training set versus never seen.*

**A modest number of training examples could be identified as such by an attacker.**
- At a 5% false-alarm rate, 7% of real training examples were correctly flagged as 'in the training set' (chance alone would give about 5%).
- At a stricter 1% false-alarm rate: 0% correctly flagged (chance alone would give about 1%).
- ROC-AUC: 0.47 (0.50 = indistinguishable from guessing, 1.00 = perfectly distinguishable; shown for reference only, the 5%-false-alarm figure above is the one to trust).

### 🔴 Lightweight Membership Inference: Strong signal
*A faster membership-inference check that compares the model's confidence against a reference model's, on the same text.*

**A clear difference from the reference model was detected. This is a membership-inference risk.**
- Separation score: 0.66 (0.50 = indistinguishable from a reference model, 1.00 = fully distinguishable).
- At a strict 1% false-alarm rate: 2% of training examples correctly flagged as such.

---
**Reading these together:** These checks disagree: EZ_MIA shows strong signal while EM shows no signal. Don't rely on either one alone. Running multiple attack families and reading them together, as you just did, is exactly the point.

---
*This summary is generated locally from the numeric scores above using fixed rules. No data or scores were sent anywhere to produce it. It is not a legal, clinical, or regulatory determination, and does not replace a full privacy review. Different attacks measure different kinds of leakage and can legitimately disagree; that disagreement is itself informative, not a flaw in the tool.*
