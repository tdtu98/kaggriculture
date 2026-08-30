"""Behaviour-cloning line for Kaggriculture (PLAN_BC Chapters 2-5).

Phase 1 is the data pipeline and nothing else:

    vocab.py         every enumerated constant, in one place
    masks.py         legality, as data; the running turn simulation
    decode.py        replay JSON -> verified (state, action) pairs
    features.py      delivered observation -> fixed-shape token arrays
    dataset.py       shard writer / numpy-only loader
    build_shards.py  the CLI, and the verification report

Nothing here imports torch, and nothing here is on the submission path.
`main.py` is owned by the compiler line for the whole program (PLAN_BC D22 iii).
"""
