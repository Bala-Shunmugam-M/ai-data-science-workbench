"""
src.trust
=========

PURPOSE
-------
The audit-grade evidence layer. The pipeline up to ``evaluate`` answers "how
accurate is the champion?"; this package answers the three questions a governed
model is actually asked before anyone relies on it:

    fairness            Does accuracy hold up across subgroups, or is the
                        headline number an average that hides a group it fails?
    robust importance   Which inputs does the model actually depend on, measured
                        without trusting the model's own internals?
    calibration         When it says 0.8, does that happen 80% of the time?

:mod:`src.trust.model_card` then assembles those three plus the artifacts the
earlier stages already wrote into one document.

PIPELINE POSITION
-----------------
    evaluate -> explain -> [trust] -> report

GOVERNANCE
----------
This stage reads the test split a second time. That is deliberate and it is not
a breach of the untouched-test doctrine: the champion is already frozen by
:mod:`src.model_evaluation.evaluator` before this stage runs, and nothing here
feeds back into model selection or tuning. Reading test again produces a
*report* on a decision already made, not the decision. Test is the right split
precisely because it is the only unbiased estimate of deployed behaviour -
measuring fairness on data the model was tuned against would flatter it.

WHAT THIS PACKAGE WILL NOT DO
-----------------------------
It does not decide whether a disparity is acceptable, and it does not label any
column a "protected attribute". It measures gaps between groups and reports the
group sizes behind them. Whether a gap is lawful, expected, or disqualifying is
a human judgement about a specific domain, and nothing here has the context to
make it.
"""
