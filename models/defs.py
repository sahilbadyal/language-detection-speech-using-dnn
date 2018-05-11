#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common definitions for Lang Detection
"""

from util import one_hot

n_classes = 3
LBLS = [
    "Belize Kriol English",
    "Hindi",
    "Other",
    ]
NONE = "O"
LMAP = {i: one_hot(n_classes,i) for i, k in enumerate(LBLS)}
LID = {k: i for i, k in enumerate(LBLS)}
