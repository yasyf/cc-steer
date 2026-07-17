"""Production retrain machinery: build the training pool, freeze the eval, train, promote.

The offline export lays down the dataset; this subpackage turns it into a promoted
model. :mod:`~cc_steer.retrain.data` reads the exported watcher train view and
curates a deterministic training pool (near-dup collapse, negative balancing,
corrective oversampling). :mod:`~cc_steer.retrain.sentinel` renders the frozen eval's
fire signal as athome eval rows, and :func:`athome.train.retrain` trains a LoRA on
Thinking Machines' managed Tinker API under a hard spend cap and scores checkpoints.
:mod:`~cc_steer.retrain.evalset` freezes the promotion eval and stores incumbent
probabilities. :mod:`~cc_steer.retrain.promotion` runs the free-metric promotion
bars and journals every verdict. The two component lanes tie it together:
:mod:`~cc_steer.retrain.lexical` retrains the stage-1 gate, and
:mod:`~cc_steer.retrain.watcher` trains, gates, converts, and promotes the stage-2
LoRA watcher.
"""

from __future__ import annotations
