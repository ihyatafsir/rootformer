#!/usr/bin/env python3
"""Correct two AWZAN_TEMPLATES labels now that the patterns exist in the blueprint.

The analyser recognised both shapes already -- the labels were wrong only because
FAAWAIL and FAAIL were absent from the wazn inventory. The source comments even name the
patterns correctly ("fawa'il", "fa'a'il") while the tuple says mafa'il.
"""
from pathlib import Path
import sys

P = Path(sys.argv[1] if len(sys.argv) > 1
         else "/workspace/hf_v19_2_release/models/morphemic_tokenizer_v12_arabic.py")
s = P.read_text(encoding="utf-8")
n = 0

# fawa'il: hawadith -> hadath
old_a = """(re.compile(r'^([^\\W\\d_])و[ا]ء?([^\\W\\d_])([^\\W\\d_])$'), 'مَفَاعِل', 'fawa_il'),"""
new_a = """(re.compile(r'^([^\\W\\d_])و[ا]ء?([^\\W\\d_])([^\\W\\d_])$'), 'فَوَاعِل', 'fawa_il'),"""
if old_a in s:
    s = s.replace(old_a, new_a); n += 1

# fa'a'il: haqa'iq -> haqq
old_b = """(re.compile(r'^([^\\W\\d_])([^\\W\\d_])ائ([^\\W\\d_])$'), 'مَفَاعِل', None),"""
new_b = """(re.compile(r'^([^\\W\\d_])([^\\W\\d_])ائ([^\\W\\d_])$'), 'فَعَائِل', None),"""
if old_b in s:
    s = s.replace(old_b, new_b); n += 1

if n == 0:
    print("PATCH DID NOT APPLY -- patterns not found")
    sys.exit(1)
P.write_text(s, encoding="utf-8")
print(f"patched {n} template label(s)")
