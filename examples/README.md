# Example artifacts

Concrete samples of what `privaudit` consumes and produces. There are two sets: a real run on a
fine-tuned model, and a synthetic pipeline demo.

## 1. run: Qwen3-0.6B fine-tuned on MedQA sample data

| File | What it is |
|---|---|
| [`medqa_job.json`](medqa_job.json) | The input: all three attacks on a LoRA-fine-tuned Qwen3-0.6B, with `Qwen/Qwen3-0.6B` as the EZ-MIA reference model. |
| [`medqa_report.json`](medqa_report.json) | The output of `privaudit run`. |
| [`medqa_explain.md`](medqa_explain.md) | The plain-language explanation from `privaudit explain` (rule-based, local). |

Setup: EM on 200 samples (50-token prefix, 500-token continuation); MIA and EZ-MIA on 100 member
and 100 non-member sequences each. Members come from the 8,142-row training split, non-members from
the 1,018-row validation split. Run on CPU.

| Attack | Headline result | Verdict from `explain` |
|---|---|---|
| EM | `mem@50` = 0.00, ROUGE-L 0.17 | No signal |
| MIA | TPR@5%FPR = 0.07, AUC 0.47 | Weak signal |
| EZ-MIA | score 0.66, TPR@1%FPR = 0.02 | Strong signal |

The attacks disagree, which is the paper's central point: exact regurgitation is absent, yet the
EZ-MIA check separates members from non-members.

Read the MIA row with care. An AUC of 0.47 is at or below chance, and 7% at a 5% false-alarm rate
on only 100 members is within sampling noise, so "weak signal" here most likely means "no real
signal". These are small samples, a single run and a single seed; use larger caps before drawing
conclusions about this model.

To reproduce, you need the model checkpoint and `train_json.jsonl` / `val_json.jsonl` from
`Example_MedQA/`. That folder is gitignored, so it is not in this repository. Then, from the repo
root:

```bash
privaudit run --config examples/medqa_job.json --output examples/medqa_report.json
privaudit explain --report examples/medqa_report.json
```

The report holds scores only. `return_raw_generations` is `false`, so no model output text or training records are included.
