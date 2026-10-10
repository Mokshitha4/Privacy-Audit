## What this run found

### ⚪ Exact Memorization: No signal
*Whether prompting the model with the start of a real training example causes it to output the rest, word for word.*

**No exact memorization detected in the tested samples.**
- None of the tested prompts produced a verbatim 50-token match with the real training continuation.
- Average text overlap with the real continuation (ROUGE-L): 0.00 of 1.00.
- Average meaning-level similarity to the real continuation (BERTScore-F1): 0.50 of 1.00.

### 🔴 Membership Inference: Strong signal
*Whether an attacker who can query the model can tell which examples were in its training set versus never seen.*

**Training examples can be reliably picked out from held-out data. This is a clear membership-inference risk.**
- At a 5% false-alarm rate, 100% of real training examples were correctly flagged as 'in the training set' (chance alone would give about 5%).
- At a stricter 1% false-alarm rate: 100% correctly flagged (chance alone would give about 1%).
- ROC-AUC: 1.00 (0.50 = indistinguishable from guessing, 1.00 = perfectly distinguishable; shown for reference only, the 5%-false-alarm figure above is the one to trust).

### ⚪ Lightweight Membership Inference: No signal
*A faster membership-inference check that compares the model's confidence against a reference model's, on the same text.*

**No meaningful difference from the reference model. No signal detected by this check.**
- Separation score: 0.50 (0.50 = indistinguishable from a reference model, 1.00 = fully distinguishable).
- At a strict 1% false-alarm rate: 1% of training examples correctly flagged as such.

---
**Reading these together:** These checks disagree: MIA shows strong signal while EM shows no signal. Don't rely on either one alone. Running multiple attack families and reading them together, as you just did, is exactly the point.

---
*This summary is generated locally from the numeric scores above using fixed rules. No data or scores were sent anywhere to produce it. It is not a legal, clinical, or regulatory determination, and does not replace a full privacy review. Different attacks measure different kinds of leakage and can legitimately disagree; that disagreement is itself informative, not a flaw in the tool.*
