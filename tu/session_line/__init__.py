"""The Claude-session executor line: the agents PLAN_v4 §1 synthesises, kept verbatim.

Load them through `_load.load`, never by importing the files directly — they hard-code sandbox
paths. See README.md in this directory for what each one is and what it measured.
"""
from ._load import SUBSTITUTIONS, agent, load  # noqa: F401
