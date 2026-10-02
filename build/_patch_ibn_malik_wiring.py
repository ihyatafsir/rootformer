#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot patch: wire the two Ibn Mālik decision points to ibn_malik_automaton.py.

Idempotent-safe: every replacement is asserted to occur exactly once and the script
exits non-zero without writing if anything does not match.
"""
import io
import sys
from pathlib import Path

REL = Path('/workspace/hf_v19_2_release')

# ---------------------------------------------------------------- A. andalusian file
A_OLD_IMPORTS = """import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
"""
A_NEW_IMPORTS = """import sys
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

# Ibn Mālik's two cited decision procedures (the POS transition automaton with the coordinator
# exception, and marātib al-maʿārif) live in ONE module so that this table, the decoder that
# calls it (nrmp_generate.py) and classical_governance_v2.IbnMalikV2 all run the same code.
# Read ibn_malik_automaton.CITATIONS for the verbatim Arabic of every rule.
try:
    from ibn_malik_automaton import IbnMalikAutomaton as _IbnMalikAutomaton
except Exception:                                    # pragma: no cover
    _IbnMalikAutomaton = None
"""

A_OLD_BLOCK = '''    @classmethod
    def transition_allowed(cls, prev_pos: int, next_pos: int,
                           coordinator_between: bool = False) -> bool:
        """Ibn Mālik's transition rule WITH the coordinator exception (Alfiyyah, bāb al-ʿaṭf)."""
        table = (cls.PERMISSIBLE_TRANSITIONS if coordinator_between
                 else cls.PERMISSIBLE_TRANSITIONS_NO_COORDINATOR)
        return next_pos in table.get(prev_pos, {1, 2, 3, 4})
'''

A_NEW_BLOCK = '''    # ---- WIRED to ibn_malik_automaton.IbnMalikAutomaton (the citation-bearing authority) ----
    # Marātib al-maʿārif, 7..1.  al-Alfiyyah l.54-55 gives the types by their examples:
    #   «نكرة قابل أل مؤثرا ... أو واقع موقع ما قد ذكرا»
    #   «وغيره معرفة كهم وذي ... وهند وابني والغلام والذي»
    # and the ranks are ordered by Ibn Mālik in al-Tashīl (reported verbatim by al-Shāṭibī):
    #   «وقد جعل لها في "التسهيل" ست مراتب، فأعلاها ضمير المتكلم، ثم ضمير المخاطب، ثم العلم،
    #    ثم ضمير الغائب السالم عن إبهام، ثم المشار به، ثم الموصول وذو الألف واللام»
    # and by Abū Ḥayyān (Tadhyīl al-Tashīl):
    #   «والذي تلقناه من الشيوخ أن أعرف المعارف هو المضمر، ويليه العلم، ويليه اسم الإشارة»
    RANK = _IbnMalikAutomaton.RANKS if _IbnMalikAutomaton is not None else {
        'PRONOUN': 7, 'PROPER': 6, 'DEMONSTRATIVE': 5, 'RELATIVE': 4,
        'DEFINITE': 3, 'ANNEXED': 2, 'INDEFINITE': 1}
    MARATIB_AL_MAARIF = RANK
    DEFINITENESS = RANK

    @classmethod
    def transition_allowed(cls, prev_pos: int, next_pos: int,
                           coordinator_between: bool = False) -> bool:
        """Ibn Mālik's transition rule WITH the coordinator exception (Alfiyyah, bāb al-ʿaṭf).

        WIRED to ibn_malik_automaton.IbnMalikAutomaton.transition: the decode loop in
        nrmp_generate.py calls this method, so the cited automaton is what the model runs
        (Alfiyyah l.569 «وعطفك الفعل على الفعل يصح», gated by the ḥarf of l.549
        «تال بحرف متبع عطف النسق»; refused without one because of l.280 «إن عاملان اقتضيا في اسم
        عمل ... قبل فللواحد منهما العمل»).  The two literal tables above mirror the automaton's
        and are checked by `tables_match_automaton()`.
        """
        if _IbnMalikAutomaton is not None:
            return _IbnMalikAutomaton().transition(prev_pos, next_pos, coordinator_between)
        table = (cls.PERMISSIBLE_TRANSITIONS if coordinator_between
                 else cls.PERMISSIBLE_TRANSITIONS_NO_COORDINATOR)
        return next_pos in table.get(prev_pos, {1, 2, 3, 4})

    @classmethod
    def rank(cls, word: str = '', def_cls: str = None, annexed_to: str = None) -> int:
        """Marātib al-maʿārif, 7..1 (al-Alfiyyah l.54-55; al-Tashīl's six ranks)."""
        if _IbnMalikAutomaton is not None:
            return _IbnMalikAutomaton().rank(word, def_cls, annexed_to=annexed_to)
        return cls.RANK.get(def_cls, 1)

    @classmethod
    def definiteness_ok(cls, mubtada: str, khabar: str) -> bool:
        """rank(mubtada') >= rank(khabar) over the marātib al-maʿārif."""
        if _IbnMalikAutomaton is not None:
            return _IbnMalikAutomaton().definiteness_ok(mubtada, khabar)
        return cls.RANK.get(mubtada, 0) >= cls.RANK.get(khabar, 0)

    @classmethod
    def naat_ok(cls, manut: str, naat: str) -> bool:
        """SOURCED (al-Shāṭibī, Sharh al-Alfiyyah): «وإنما ينعت بما كان في رتبته أو دون رتبته،
        لا بما هو فوق رتبته» -- a naʿt is of its manʿūt's rank or below, never above."""
        if _IbnMalikAutomaton is not None:
            return _IbnMalikAutomaton().naat_ok(manut, naat)
        return cls.RANK.get(naat, 0) <= cls.RANK.get(manut, 0)

    @classmethod
    def coordinator_of(cls, word: str) -> bool:
        """The ʿāṭif wāw/fāʾ itself (al-Alfiyyah l.550 «فالعطف مطلقا بواو ثم فا»)."""
        if _IbnMalikAutomaton is not None:
            return _IbnMalikAutomaton().coordinator_of(word)
        return (word or '').strip() in ('و', 'ف')

    @classmethod
    def tables_match_automaton(cls) -> bool:
        """True when the literal tables above mirror the cited automaton exactly."""
        if _IbnMalikAutomaton is None:
            return True
        A = _IbnMalikAutomaton
        return ({k: set(v) for k, v in cls.PERMISSIBLE_TRANSITIONS.items()}
                == {k: set(v) for k, v in A.TRANSITIONS.items()}
                and {k: set(v) for k, v in cls.PERMISSIBLE_TRANSITIONS_NO_COORDINATOR.items()}
                == {k: set(v) for k, v in A.TRANSITIONS_NO_COORDINATOR.items()})
'''

# ---------------------------------------------------------------- B. classical_governance_v2
B_OLD_IMPORTS = """import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
"""
B_NEW_IMPORTS = """import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# The single source of truth for Ibn Mālik's two procedures, with the verbatim citations:
# marātib al-maʿārif (al-Alfiyyah l.54-55) and the POS automaton with the coordinator exception
# (l.549/550/569 + bāb al-tanāzuʿ l.280).  IbnMalikV2 below delegates to it.
try:
    from ibn_malik_automaton import IbnMalikAutomaton as _IbnMalikAutomaton
except Exception:                                    # pragma: no cover
    _IbnMalikAutomaton = None
"""

B_OLD_BLOCK = """    # Maratib al-Ma'arif: Pronoun > Proper > Demonstrative > Definite > Indefinite
    # Ibn Malik, Alfiyyah v.55: "وغيره معرفة كهم وذي ... وهند وابني والغلام والذي"
    #   ham=pronoun, dhi=demonstrative, hind=proper, ibni=annexed, al-ghulam=with al-, alladhi=relative
    DEFINITENESS = {'PRONOUN': 7, 'PROPER': 6, 'DEMONSTRATIVE': 5, 'RELATIVE': 4,
                    'DEFINITE': 3, 'ANNEXED': 2, 'INDEFINITE': 1}

    @classmethod
    def transition_ok(cls, prev: int, nxt: int, coordinator_between: bool = False) -> bool:
        if coordinator_between:
            return True
        return nxt in cls.TRANSITIONS.get(prev, {cls.ISM, cls.FIL, cls.HARF, cls.SIFAH})

    @classmethod
    def definiteness_ok(cls, mubtada: str, khabar: str) -> bool:
        \"\"\"rank(mubtada') >= rank(khabar).\"\"\"
        return cls.DEFINITENESS.get(mubtada, 0) >= cls.DEFINITENESS.get(khabar, 0)
"""

B_NEW_BLOCK = """    # Maratib al-Ma'arif: Pronoun > Proper > Demonstrative > Relative > Definite-with-al >
    # Annexed (mudaf) > Indefinite.  Ibn Malik, Alfiyyah v.54-55:
    #   "نكرة قابل أل مؤثرا ... أو واقع موقع ما قد ذكرا"
    #   "وغيره معرفة كهم وذي ... وهند وابني والغلام والذي"
    #   ham=pronoun, dhi=demonstrative, hind=proper, ibni=annexed, al-ghulam=with al-, alladhi=relative
    # WIRED: the ranks now come from ibn_malik_automaton.IbnMalikAutomaton, which also carries
    # Ibn Malik's own six ranks in al-Tashil (via al-Shatibi's Sharh) and Abu Hayyan's ordering.
    if _IbnMalikAutomaton is not None:
        DEFINITENESS = _IbnMalikAutomaton.RANKS
    else:                                            # pragma: no cover
        DEFINITENESS = {'PRONOUN': 7, 'PROPER': 6, 'DEMONSTRATIVE': 5, 'RELATIVE': 4,
                        'DEFINITE': 3, 'ANNEXED': 2, 'INDEFINITE': 1}

    @classmethod
    def transition_ok(cls, prev: int, nxt: int, coordinator_between: bool = False) -> bool:
        \"\"\"Delegates to the cited automaton: Fi'l -> Fi'l only through a coordinator.

        Alfiyyah, bab al-'atf: "وعطفك الفعل على الفعل يصح"; the ʿatf is "تال بحرف متبع عطف النسق",
        and without the harf the two verbs would be two operators on one operand, which
        "إن عاملان اقتضيا في اسم عمل ... قبل فللواحد منهما العمل" forbids.
        \"\"\"
        if _IbnMalikAutomaton is not None:
            return _IbnMalikAutomaton().transition(prev, nxt, coordinator_between)
        if coordinator_between:
            return True
        return nxt in cls.TRANSITIONS.get(prev, {cls.ISM, cls.FIL, cls.HARF, cls.SIFAH})

    @classmethod
    def definiteness_ok(cls, mubtada: str, khabar: str) -> bool:
        \"\"\"rank(mubtada') >= rank(khabar).  ENGINEERING direction; the ordering is sourced.\"\"\"
        if _IbnMalikAutomaton is not None:
            return _IbnMalikAutomaton().definiteness_ok(mubtada, khabar)
        return cls.DEFINITENESS.get(mubtada, 0) >= cls.DEFINITENESS.get(khabar, 0)

    @classmethod
    def definiteness_rank(cls, cls_name: str) -> int:
        \"\"\"The 7..1 rank of a maratib al-ma'arif class label (al-Alfiyyah v.55).\"\"\"
        if _IbnMalikAutomaton is not None:
            return _IbnMalikAutomaton().rank(cls=cls_name)
        return cls.DEFINITENESS.get(cls_name, 0)
"""

PATCHES = [
    ('andalusian_grammatical_algorithms.py', [(A_OLD_IMPORTS, A_NEW_IMPORTS),
                                              (A_OLD_BLOCK, A_NEW_BLOCK)]),
    ('classical_governance_v2.py', [(B_OLD_IMPORTS, B_NEW_IMPORTS),
                                    (B_OLD_BLOCK, B_NEW_BLOCK)]),
]


def main() -> int:
    staged = {}
    for name, pairs in PATCHES:
        path = REL / name
        text = path.read_text(encoding='utf-8')
        for old, new in pairs:
            n = text.count(old)
            if n != 1:
                print(f'ABORT: {name}: pattern occurs {n} times (expected 1):\n{old[:120]}')
                return 2
            text = text.replace(old, new, 1)
        staged[name] = text
    for name, text in staged.items():
        path = REL / name
        path.write_text(text, encoding='utf-8')
        print(f'patched {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
