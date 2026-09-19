# R4 (first half) — is the failure column-specific? Outcome: **no, and the fleet result stands**

Pre-registered in `results/remodel/FINDINGS.md` §R4 before the run: *the failure is not column-specific* — on the
units scoreable in both columns the paid arm's closer-than-Chain-Ladder rate should be **not better than
38.8% + 2.3pp**, with a *fired falsifier* being a result in its own right (it would mean the entry's fleet
finding must be reported per column, not as one number).

## The result

`scripts/fleet_eval.py --column CumPaidLoss`, same units, same anchors, same held-out policy, same arms;
compared by `scripts/column_comparison.py`, paired on `(triangle, anchor)`:

```
incurred units 464 | paid units 1299 | intersection 414

metric (n=414)                        incurred      paid
median |model error| %                   160.1      33.4
median |Chain Ladder error| %            121.0      25.0
closer than Chain Ladder                 38.2%     36.2%     (noise floor 2.3pp)

paired difference (paid - incurred) in closer-than-CL: -1.9%  (se 3.3%)
```

**Expectation met.** Paid sits inside the noise floor of incurred, and the recomputation **reproduces the
published number** — 38.2% on the 414-unit intersection against the recorded 38.8% over all 464, the difference
being the intersection. The failure is a property of the *method and the regime*, not of the column chosen, and
the entry's fleet finding stands as one number.

## Two secondary findings worth keeping

1. **The incurred column is simply the harder one, for both methods.** Both the model *and* Chain Ladder are
   far more accurate on paid (model median |error| **33.4%** against **160.1%**; Chain Ladder **25.0%** against
   **121.0%**). So the large absolute errors in the published fleet result are substantially the incurred
   column's own noise — case-reserve strengthening, the thing `MunichAdjustment` and `CaseOutstanding` exist to
   correct — and the model's **relative** deficit is unchanged by the swap. That is a cleaner explanation of the
   error magnitudes than anything the entry currently gives, and it costs no new evidence to state.
2. **The paid column supports ~2.8× the evidence base**: 1299 scoreable units against incurred's 464, on the
   same 775 containers. The entry's fleet result rests on the smaller set; a robustness section could rest on
   the larger one.

## The defect this run exposed, and it is the reason the analysis was rewritten

The first version of the comparison read the stored `error_chainladder_pct` field and reported:

> **FALSIFIER FIRED**: the rates differ by 36.2%, beyond the 2.3pp noise floor.

That was false, and it would have been a headline about the entry's central finding. `clrd-IncurLoss.jsonl`
carries **`error_chainladder_pct = 0.0` on all 464 rows** — the same null defect this repository found and fixed
once before, in the *summary* (`scripts/fleet_eval.py:198`, *"everything is recomputed from primitives"*) but
never rewritten in that stored file. Reading it gave incurred a closer-than-CL rate of **0.0%**, which is
impossible for a run whose recorded rate is 38.8%, and which the cross-check against the published number
therefore caught.

The comparison now recomputes both errors from the primitives (`actual`, `direct`, `chainladder`), refuses to
run if those are absent, and **reports the stale-field condition when it finds it** rather than silently
recomputing around it. The stored field is left as it is: it is a record, and the canonical number lives in
`results/fleet/FINDINGS.md`. A warning about it is recorded there.

## Provenance

- Paid run: `results/fleet/clrd-CumPaidLoss.jsonl` (1299 units), log `results/fleet/run-CumPaidLoss.log`.
- Incurred: `results/fleet/clrd-IncurLoss.jsonl` (464 units) — primitives only; its stored error fields are
  degenerate (see above).
- Comparison: `scripts/column_comparison.py`, paired, errors recomputed, guarded against adjudicating on a
  sample without power (below 100 units or with both rates at zero it exits 2 and says so).
