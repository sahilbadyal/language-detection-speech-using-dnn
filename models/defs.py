#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common definitions for Lang Detection
"""

from util import one_hot

LBLS = [
    "EN",
    "HI",
    "O",
    ]
NONE = "O"
LMAP = {k: one_hot(3,i) for i, k in enumerate(LBLS)}
NUM = "NNNUMMM"
UNK = "UUUNKKK"

EMBED_SIZE = 50
