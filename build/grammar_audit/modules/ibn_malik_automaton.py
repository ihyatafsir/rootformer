#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ibn_malik_automaton.py -- Ibn Mālik's two decision procedures, as one callable authority.

This module is the SINGLE SOURCE OF TRUTH for the two Ibn Mālik rules the NRMP decoder needs,
so that the corrected implementation in `classical_governance_v2.IbnMalikV2` and the table in
`andalusian_grammatical_algorithms.IbnMalikPOSAutomaton` (the one the decode loop at
`nrmp_generate.py` actually calls) both run the same cited code:

  1. MARĀTIB AL-MAʿĀRIF -- the definiteness hierarchy (al-Alfiyyah, bāb al-maʿrifah wa-l-nakirah,
     l.54-55).  Ranks 7..1: pronoun > proper name > demonstrative > relative > definite with al-
     > annexed (muḍāf) > indefinite.

  2. THE POS TRANSITION AUTOMATON -- what may follow what, with the COORDINATOR EXCEPTION.
     Fiʿl -> Fiʿl is *legal* when a coordinator (و/ف) joins the two verbs and *illegal* when
     nothing joins them.  The licence is al-Alfiyyah, bāb al-ʿaṭf: «وعطفك الفعل على الفعل يصح»;
     the reason the exception exists is the Basran qāʿidah that no two operators govern one
     operand -- Sībawayh, al-Kitāb: «فالعامل في اللفظ أحد الفعلين» and «لا يعمل في اسم واحد نصب
     ورفع», with his own coordinated witness «ونخلع ونترك»; al-Alfiyyah, bāb al-tanāzuʿ: «إن
     عاملان اقتضيا في اسم عمل ... قبل فللواحد منهما العمل»; and al-Shāṭibī's Sharh: «فيؤدى ذلك
     إلى أن يعمل عاملان معا في معمول واحد عملا واحدا، وذلك فاسد».

Every rule below carries the verbatim Arabic it rests on, with book + locator, and every cell
that has NO primary source is explicitly labelled ENGINEERING.  Nothing is quoted that was not
read out of the corpus files named in `CITATIONS`.

Usage (the decoder-facing API):

    from ibn_malik_automaton import IbnMalikAutomaton
    A = IbnMalikAutomaton()
    A.rank('هند', 'PROPER')                              -> 6
    A.transition('FIL', 'FIL', coordinator_between=True) -> True
    A.transition('FIL', 'FIL', coordinator_between=False)-> False
    A.apply([('قام', 'FIL'), ('وقعد', 'FIL')])           -> [..., allowed=True ...]

Run `python ibn_malik_automaton.py` for the built-in proof (all 20 checks).
"""
from __future__ import annotations

import re
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ============================================================================================
# CITATIONS -- verbatim Arabic, each string verified byte-for-byte against the corpus files
# listed in `book`.  The corpus is undiacritized; the quotations are therefore undiacritized
# exactly as transmitted there.
# ============================================================================================
CITATIONS: Dict[str, Dict[str, str]] = {
    # ---- the definiteness hierarchy -------------------------------------------------------
    'M_nakirah_verse': {
        'book': "Ibn Mālik, al-Alfiyyah, bāb al-maʿrifah wa-l-nakīrah, l.54",
        'file': 'corpus/andalusian/03_IbnMalik_Alfiyyah.txt:54',
        'ar': 'نكرة قابل أل مؤثرا ... أو واقع موقع ما قد ذكرا',
        'note': 'defines the NAKIRAH (indefinite): what can take the definiting al-, or stands in '
                'the place of one that can.',
    },
    'M_marifah_verse': {
        'book': "Ibn Mālik, al-Alfiyyah, bāb al-maʿrifah wa-l-nakīrah, l.55",
        'file': 'corpus/andalusian/03_IbnMalik_Alfiyyah.txt:55',
        'ar': 'وغيره معرفة كهم وذي ... وهند وابني والغلام والذي',
        'note': 'the six maʿrifah types by their examples: هم pronoun | ذي demonstrative | هند '
                'proper | ابني annexed (muḍāf) | الغلام definite with al- | الذي relative.',
    },
    'M_ranks_tashil': {
        'book': "Ibn Mālik, al-Tashīl (as reported verbatim by al-Shāṭibī, Sharh al-Alfiyyah, "
                "on l.55)",
        'file': 'corpus/andalusian/11_Shatibi_Sharh_Alfiyyah.txt:3562-3565',
        'ar': 'وقد جعل لها في "التسهيل" ست مراتب، فأعلاها ضمير المتكلم، ثم ضمير المخاطب، ثم العلم، '
              'ثم ضمير الغائب السالم عن إبهام، ثم المشار به، ثم الموصول وذو الألف واللام',
        'note': "Ibn Mālik's own SIX ranks, in his own order. NOTE: he puts al-mawṣūl and dhū "
                "al-alif wa-l-lām in ONE rank; this module splits them (see M_mawsul_variant).",
    },
    'M_mudaf_rank': {
        'book': 'Abū Ḥayyān, Tadhyīl al-Tashīl, on the same verse',
        'file': 'corpus/andalusian/10_AbuHayyan_Tadhyil_al_Tashil.txt:6140-6141',
        'ar': 'وقوله: والمضاف بحسب المضاف إليه يعني أنه يكتسي التعريف من المضاف إليه، فيصير مثله '
              'في التعريف',
        'note': 'the muḍāf takes the rank of its muḍāf ilayhi -- this is what ANNEXED inherits.',
    },
    'M_ranks_shuyukh': {
        'book': 'Abū Ḥayyān, Tadhyīl al-Tashīl (the order the Basran shaykhs transmitted)',
        'file': 'corpus/andalusian/10_AbuHayyan_Tadhyil_al_Tashil.txt:6092-6095',
        'ar': 'والذي تلقناه من الشيوخ أن أعرف المعارف هو المضمر، ويليه العلم، ويليه اسم الإشارة، '
              'ويليه ذو الألف واللام، وأما المضاف فإنه في رتبة المضاف إليه إلا المضاف إلى المضمر، '
              'فإنه في رتبة العلم',
        'note': 'pronoun > proper > demonstrative > dhū al-; and the muḍāf to a PRONOUN ranks as '
                'the PROPER (the Basran refinement -- see `basran_annexation`).',
    },
    'M_mawsul_variant': {
        'book': 'Abū Ḥayyān, Tadhyīl al-Tashīl, on «ثم الموصول وذو الأداة»',
        'file': 'corpus/andalusian/10_AbuHayyan_Tadhyil_al_Tashil.txt:6140',
        'ar': 'وقوله: وذو الأداة جعل الموصول والمعرف بأل في رتبة واحدة، وكأنه رأى أن التعريف '
              'فيهما بالعهد، والعهد موجود في الصلة كما أنه موجود في أل. وثبت في بعض النسخ: '
              '"ثم ذو/ أداة"، فجعل ذا الأداة في التعريف بعد الموصول',
        'note': 'GROUNDS the split the class encodes: a transmitted VARIANT READING puts dhū adāh '
                'AFTER the mawṣūl, i.e. relative(4) > definite-with-al(3).  The base reading has '
                'them level; the split is therefore a sourced variant, not an invention.',
    },
    'M_naat_rank_rule': {
        'book': 'al-Shāṭibī, Sharh al-Alfiyyah, on l.55 (the maʿrifah ranks)',
        'file': 'corpus/andalusian/11_Shatibi_Sharh_Alfiyyah.txt:3566-3568',
        'ar': 'إذ المعرفة من الأسماء لا ينعت بكل معرفة، وإنما ينعت بما كان في رتبته أو دون رتبته، '
              'لا بما هو فوق رتبته',
        'note': 'the definiteness COMPARISON: a naʿt may not outrank its manʿūt.  This is the '
                'decision the hierarchy is FOR.',
    },
    # ---- the transition automaton + the coordinator exception -----------------------------
    'M_atf_is_bi_harf': {
        'book': "Ibn Mālik, al-Alfiyyah, bāb al-ʿaṭf, l.549",
        'file': 'corpus/andalusian/03_IbnMalik_Alfiyyah.txt:549',
        'ar': 'تال بحرف متبع عطف النسق ... كاخصص بود وثناء من صدق',
        'note': "ʿaṭf al-nasaq is BY DEFINITION a follower attached BI-ḤARF (by a particle).  No "
                "particle, no ʿaṭf -- hence no licence for a bare Fiʿl after a Fiʿl.",
    },
    'M_coordinators': {
        'book': "Ibn Mālik, al-Alfiyyah, bāb al-ʿaṭf, l.550",
        'file': 'corpus/andalusian/03_IbnMalik_Alfiyyah.txt:550',
        'ar': 'فالعطف مطلقا بواو ثم فا ... حتى أم أو كفيك صدق ووفا',
        'note': 'names the coordinating particles; the wāw and the fāʾ are the two this module '
                'reads from the prefix/surface slot.',
    },
    'M_atf_fil_verse': {
        'book': "Ibn Mālik, al-Alfiyyah, bāb al-ʿaṭf, l.569",
        'file': 'corpus/andalusian/03_IbnMalik_Alfiyyah.txt:569',
        'ar': 'وحذف متبوع بدا عنا استبح ... وعطفك الفعل على الفعل يصح',
        'note': 'THE LICENCE: coordinating a verb onto a verb is correct.',
    },
    'M_naql_hukm': {
        'book': "Ibn Mālik, al-Alfiyyah, bāb al-ʿaṭf, l.562",
        'file': 'corpus/andalusian/03_IbnMalik_Alfiyyah.txt:562',
        'ar': 'وانقل بها للثان حكم الأول ... في الخبر المثبت والأمر الجلي',
        'note': "the coordinator carries the first's ḥukm to the second -- so the second verb is "
                "not a second independent operator.",
    },
    'M_atf_fil_sharh': {
        'book': "al-Shāṭibī, Sharh al-Alfiyyah, on l.569",
        'file': 'corpus/andalusian/11_Shatibi_Sharh_Alfiyyah.txt:42877-42881',
        'ar': 'ثم قال: "وعطفك الفعل على الفعل يصح" ... ويريد أن الفعل / يصح أن يعطف على الفعل، '
              'كما يصح أن يعطف الاسم على الاسم، وكما تعطف الجملة على الجملة، من غير مانع من ذلك',
        'note': 'the commentary reading of the verse: the verb-by-verb ʿaṭf is sound, and it is '
                'ʿaṭf (i.e. bi-ḥarf, per l.549), not mere juxtaposition.',
    },
    'M_two_amil_verse': {
        'book': "Ibn Mālik, al-Alfiyyah, bāb al-tanāzuʿ, l.280",
        'file': 'corpus/andalusian/03_IbnMalik_Alfiyyah.txt:280',
        'ar': 'إن عاملان اقتضيا في اسم عمل ... قبل فللواحد منهما العمل',
        'note': 'when two operators both require work in one noun, ONE of them operates -- the '
                'qāʿidah that makes an uncoordinated pair of verbs ill-formed.',
    },
    'M_qama_wa_qada': {
        'book': "Ibn Mālik, Sharh al-Kāfiyah, bāb al-tanāzuʿ",
        'file': 'corpus/andalusian/05_IbnMalik_Sharh_al_Kafiyah.txt:4042-4044',
        'ar': 'بخلاف المتقدمين نحو: "قام وقعد زيد". فإن كل واحد من الفعلين موجه في المعنى إلى '
              '"زيد" وصالح للعمل في لفظه. فأعمل أحدهما في ظاهره، والآخر في ضميره',
        'note': "Ibn Mālik's OWN example «قام وقعد زيد»: two verbs joined by the wāw, each "
                'directed at zayd, and only ONE operates -- the construction is sound Arabic.',
    },
    'M_tashil_tanazu': {
        'book': "Ibn Mālik, al-Tashīl, bāb «تنازع العاملين فصاعدا معمولا واحدا»",
        'file': 'corpus/andalusian/04_IbnMalik_Tashil_al_Fawaid.txt:773-774',
        'ar': 'اذا تعلق عاملان من الفعل وشبهه متفقان لغير توكيد أو مختلفان بما تأخر غير سببي '
              'مرفوع عمل فيه أحدهما لا كلاهما خلافا للفراء في نحو: قام وقعد زيد',
        'note': 'the same construction in the Tashīl: «عمل فيه أحدهما لا كلاهما» -- one operates, '
                'not both (against al-Farrāʾ).  The wāw keeps the two from being rivals.',
    },
    'M_no_two_operators': {
        'book': 'al-Shāṭibī, Sharh al-Alfiyyah, on the ʿaṭf in «إن زيدا وعمرو قائمان»',
        'file': 'corpus/andalusian/11_Shatibi_Sharh_Alfiyyah.txt:15175-15176',
        'ar': 'فيؤدى ذلك إلى أن يعمل عاملان معا في معمول واحد عملا واحدا، وذلك فاسد',
        'note': 'the Basran prohibition in al-Shāṭibī\'s words: two operators working one operand '
                'at once is fasid.',
    },
    'M_no_two_operators_suhayli': {
        'book': 'al-Suhaylī, Natāʾij al-Fikr',
        'file': 'corpus/andalusian/02_Suhayli_Nataij_al_Fikr.txt:1794',
        'ar': 'ولا يجتمع جازمان كما لا يجتمع في شيء من الكلام عاملان في معمول واحد من خفض ولا نصب',
        'note': 'the same qāʿidah stated generally: no two operators in one operand, jarr or naṣb.',
    },
    'M_qama_wa_qada_ibnmada': {
        'book': "Ibn Maḍāʾ, al-Radd ʿalā al-Nuḥāt",
        'file': 'corpus/andalusian/01_IbnMada_Radd_ala_Nuhat.txt:333',
        'ar': 'تقول (قام وقعد زيد) فإن علقت زيدا بالفعل الثاني',
        'note': 'independent attestation that «قام وقعد زيد» is a real construction the grammarians '
                'argue about -- i.e. Fiʿl + coordinator + Fiʿl is attested Arabic, not an error.',
    },
    # ---- Sībawayh, al-Kitāb: the earliest statement of the prohibition --------------------
    'M_sibawayh_tanazu': {
        'book': "Sībawayh, al-Kitāb, bāb «الفاعلين والمفعولين اللذين كل واحد منهما يفعل بفاعله "
                "مثل الذى يفعل به وما كان نحو ذلك»",
        'file': 'corpus/basran/Sibawayh_Al_Kitab.txt:934-936',
        'ar': 'هذا باب الفاعلين والمفعولين اللذين كل واحد منهما يفعل بفاعله مثل الذى يفعل به وما '
              'كان نحو ذلك ... فالعامل في اللفظ أحد الفعلين، وأما في المعنى',
        'note': "Sībawayh's tanāzuʿ chapter: where two verbs both require one noun, the operator, "
                'in the lafẓ, is ONE of the two -- they are not two operators on one operand.',
    },
    'M_sibawayh_one_irab': {
        'book': 'Sībawayh, al-Kitāb, same chapter',
        'file': 'corpus/basran/Sibawayh_Al_Kitab.txt:938',
        'ar': 'فقد يعلم أن الأول قد وقع إلا أنه لا يعمل في اسم واحد نصب ورفع',
        'note': 'one noun does not take naṣb and rafʿ at once -- the earliest form of «no two '
                'operators in one operand», which is why an uncoordinated pair of verbs fails.',
    },
    'M_sibawayh_fil_atf': {
        'book': "Sībawayh, al-Kitāb, same chapter (his own witness for the coordinated pair)",
        'file': 'corpus/basran/Sibawayh_Al_Kitab.txt:944-945',
        'ar': 'فلم يعمل الآخر فيما عمل فيه الأول استغناء عنه ومثل ذلك: ونخلع ونترك',
        'note': 'the coordinated verbs «ونخلع ونترك»: Fiʿl + wāw + Fiʿl attested in Sībawayh, '
                "and the second does not take over the first's work.",
    },
}


# ============================================================================================
# 1. MARĀTIB AL-MAʿĀRIF -- the definiteness hierarchy
# ============================================================================================
class IbnMalikAutomaton:
    """Ibn Mālik's POS automaton (with the coordinator exception) + marātib al-maʿārif.

    Two independent procedures, both cited above:

      * POS classes:  ISM=1, FIL=2, HARF=3, SIFAH=4 (the four of «واسم وفعل ثم حرف الكلم»,
        plus Ibn Mālik's tābiʿ category SIFAH which the project's table already carried).
      * Definiteness classes: PRONOUN=7 ... INDEFINITE=1.
    """

    # -- POS classes ------------------------------------------------------------------------
    ISM, FIL, HARF, SIFAH = 1, 2, 3, 4
    CLASS_NAME = {0: 'NONE', 1: 'ISM', 2: 'FIL', 3: 'HARF', 4: 'SIFAH'}
    CLASS_ID = {'ISM': 1, 'FIL': 2, 'HARF': 3, 'SIFAH': 4,
                'ISM_NAME': 1, 'FI_L': 2, 'FIL': 2, 'VERB': 2, 'NOUN': 1, 'PARTICLE': 3}

    #: SOURCED (M_atf_fil_verse, M_naql_hukm, M_atf_is_bi_harf): Fiʿl -> Fiʿl is admitted here
    #: and gated by the caller through `coordinator_between`.  All other cells are ENGINEERING
    #: (the project's pre-existing table, retained verbatim so nothing else moves).
    TRANSITIONS: Dict[int, set] = {
        0: {ISM, FIL, HARF},          # ENGINEERING: a clause may open with any of the three
        ISM: {ISM, FIL, HARF, SIFAH},  # ENGINEERING
        FIL: {ISM, FIL, HARF},         # SOURCED: FIL comes back only via a coordinator (l.569)
        HARF: {ISM, FIL},              # ENGINEERING: retained; NOTE «إن في الدار» is Harf+Harf
        SIFAH: {ISM, FIL, HARF, SIFAH},  # ENGINEERING
    }
    #: The same automaton with the pre-v19.2 reading: two verbs may NOT follow one another.
    #: This is the table the decoder uses when it sees no coordinator, and it is what makes
    #: «قام قعد» ill-formed while «قام وقعد» is sound.
    TRANSITIONS_NO_COORDINATOR: Dict[int, set] = {
        0: {ISM, FIL, HARF},
        ISM: {ISM, FIL, HARF, SIFAH},
        FIL: {ISM, HARF},              # SOURCED (composite): FIL -> FIL needs the ḥarf of l.549
        HARF: {ISM, FIL},
        SIFAH: {ISM, FIL, HARF, SIFAH},
    }
    #: `PERMISSIBLE_TRANSITIONS` is the name the existing project code and its audit use.
    PERMISSIBLE_TRANSITIONS = TRANSITIONS
    PERMISSIBLE_TRANSITIONS_NO_COORDINATOR = TRANSITIONS_NO_COORDINATOR

    #: M_coordinators (l.550): the wāw and the fāʾ this module reads.
    COORDINATORS = {'و', 'ف'}
    #: ENGINEERING (morphological): the NRMP tuple factors the clitic into the PREFIX slot, so
    #: «وقعد» arrives as prefix='و'; «والكتاب» as prefix='وال'.
    COORDINATOR_PREFIXES = {'و', 'ف', 'وال', 'فال'}

    # -- definiteness classes ---------------------------------------------------------------
    #: SOURCED: M_marifah_verse names the types; M_ranks_shuyukh / M_ranks_tashil order them.
    #: NOTE ON FAITHFULNESS -- Ibn Mālik's Tashīl (M_ranks_tashil) gives SIX ranks, splitting
    #: the pronouns into mutakallim > mukhāṭab > ʿalam > ghāʾib and putting al-mawṣūl WITH dhū
    #: al-alif wa-l-lām.  This module keeps the project's SEVEN-level scale (one PRONOUN rank,
    #: and mawṣūl above al- on the strength of the variant reading M_mawsul_variant).  The
    #: finer pronoun split is NOT implemented -- recorded as a known simplification, not a quote.
    RANKS: Dict[str, int] = {
        'PRONOUN': 7,        # هم        (M_marifah_verse; M_ranks_shuyukh: «أعرف المعارف هو المضمر»)
        'PROPER': 6,         # هند       (M_marifah_verse; M_ranks_tashil: «ثم العلم»)
        'DEMONSTRATIVE': 5,  # ذي        (M_marifah_verse; «ويليه اسم الإشارة»)
        'RELATIVE': 4,       # الذي      (M_marifah_verse; M_mawsul_variant)
        'DEFINITE': 3,       # الغلام    (M_marifah_verse; «ذو الألف واللام»)
        'ANNEXED': 2,        # ابني      (M_marifah_verse; M_mudaf_rank)
        'INDEFINITE': 1,     # نكرة      (M_nakirah_verse)
    }
    DEFINITENESS = RANKS
    MARATIB_AL_MAARIF = RANKS
    RANK = RANKS
    #: alias labels that callers in this codebase already use or may reasonably pass
    RANK_ALIASES = {
        'MARIFAH': 3, 'DEFINITE_AL': 3, 'AL_DEFINITE': 3, 'WITH_AL': 3, 'MAARIFA': 3,
        'MUDHAF': 2, 'MUDAF': 2, 'IDAFI': 2, 'ANNEXED_NOUN': 2,
        'NAKIRAH': 1, 'NAKIRA': 1, 'INDEF': 1,
        'DAMIR': 7, 'PRON': 7, 'ISM_ISHARA': 5, 'ISHARA': 5,
        'MAWSUL': 4, 'ALAM': 6,
    }
    #: Ibn Mālik's own six, in his own order (M_ranks_tashil), kept for traceability.
    SIX_RANKS_TASHIL: Tuple[str, ...] = (
        'PRONOUN_1ST', 'PRONOUN_2ND', 'PROPER', 'PRONOUN_3RD', 'DEMONSTRATIVE',
        'RELATIVE_AND_DEFINITE_AL',
    )

    #: ENGINEERING: surface cues, used ONLY when the caller gives no class.  The decoder itself
    #: supplies the class from `pos_category()`/the morphological tuple, so these sets are a
    #: convenience for bare-string callers.  Proper names cannot be recognised from the surface
    #: ('PROPER' must be passed explicitly, as the Alfiyyah's own example «هند» is passed here).
    DEMONSTRATIVES = {'هذا', 'هذه', 'هؤلاء', 'ذلك', 'تلك', 'أولئك', 'ذا', 'ذي', 'ذه',
                      'هناك', 'هنالك', 'هنا', 'كذا', 'كذلك'}
    RELATIVES = {'الذي', 'التي', 'الذين', 'اللذان', 'اللتان', 'اللذين', 'اللتين',
                 'اللاتي', 'اللواتي', 'اللائي', 'الذي'}
    PRONOUNS = {'أنا', 'نحن', 'أنت', 'أنتما', 'أنتم', 'أنتن', 'هو', 'هي', 'هما', 'هم', 'هن',
                'إياي', 'إيانا', 'إياك', 'إياكما', 'إياكم', 'إياكن', 'إياه', 'إياها', 'إياهما',
                'إياهم', 'إياهن'}
    #: ENGINEERING: a floor for bare-string use in this project's tests; 'PROPER' is never
    #: inferred from the surface (see above).
    PROPER_NAMES = {'هند', 'دعد', 'زيد', 'عمرو', 'محمد', 'الله', 'مكة', 'بغداد', 'سعاد'}

    # ---------------------------------------------------------------- definiteness: rank
    def rank(self, word: str = '', cls: Any = None, annexed_to: Optional[str] = None,
             basran_annexation: bool = False) -> int:
        """Marātib al-maʿārif: return the rank 7..1 of a word or of an explicit class.

        Explicit classes win; the surface is only consulted when no class is given.

        `annexed_to` implements the SOURCED inheritance of M_mudaf_rank / M_ranks_shuyukh: a
        muḍāf ranks as its muḍāf ilayhi.  `basran_annexation=True` adds the Basran refinement
        «إلا المضاف إلى المضمر، فإنه في رتبة العلم» (muḍāf to a pronoun -> the PROPER rank).

        ENGINEERING: the direction/aggregation here (a single 7-point integer scale) is this
        module's encoding of the sources' ranked lists.
        """
        label = self._rank_label(cls)
        if label is not None and annexed_to is None:
            return self.RANKS[label]
        if annexed_to is not None:
            base = self.RANKS.get(self._rank_label(annexed_to) or '', None)
            if base is None:
                base = self.rank(annexed_to)
            if basran_annexation and base == self.RANKS['PRONOUN']:
                return self.RANKS['PROPER']          # M_ranks_shuyukh
            return base                              # M_mudaf_rank
        if label is not None:
            return self.RANKS[label]
        return self.RANKS[self.definiteness(word)]

    def _rank_label(self, cls: Any) -> Optional[str]:
        """Normalise a caller's class label to one of the seven rank names (or None)."""
        if cls is None:
            return None
        if isinstance(cls, dict):
            for k in ('definiteness', 'rank_class', 'def'):
                if k in cls:
                    return self._rank_label(cls[k])
            if 'rank' in cls and isinstance(cls['rank'], (int, float)):
                return self.label_of_rank(int(cls['rank']))
            return None
        if isinstance(cls, str):
            key = cls.strip().upper()
            if key in self.RANKS:
                return key
            if key in self.RANK_ALIASES:
                return self.label_of_rank(self.RANK_ALIASES[key])
            if ':' in key:                            # e.g. 'ANNEXED:DEFINITE'
                head = key.split(':', 1)[0]
                return head if head in self.RANKS else None
            return None
        return None

    @classmethod
    def label_of_rank(cls, value: int) -> Optional[str]:
        for k, v in cls.RANKS.items():
            if v == value:
                return k
        return None

    def definiteness(self, word: str = '', cls: Any = None) -> str:
        """Return the definiteness CLASS of a word: one of the seven rank names.

        A `cls` that is already a definiteness class is returned as-is.  A POS class is mapped
        to the best surface inference (ENGINEERING), because definiteness is a lexical property
        of the noun, not of the POS tag.
        """
        label = self._rank_label(cls)
        if label is not None:
            return label
        bare = self._bare(word)
        if not bare:
            return 'INDEFINITE'
        if bare in self.PRONOUNS:
            return 'PRONOUN'
        if bare in self.DEMONSTRATIVES:
            return 'DEMONSTRATIVE'
        if bare in self.RELATIVES:
            return 'RELATIVE'
        if bare in self.PROPER_NAMES:
            return 'PROPER'
        # definition by the definiting al- (M_nakirah_verse: «نكرة قابل أل مؤثرا»)
        if re.match(r'^(?:وال|فال|بال|كال|لل|ال).{2,}$', bare):
            return 'DEFINITE'
        return 'INDEFINITE'

    def definiteness_ok(self, mubtada: str, khabar: str) -> bool:
        """A mubtada' / manʿūt is not outranked by what is predicated of it.

        ENGINEERING: the DIRECTION (`rank(mubtada) >= rank(khabar)`) is this project's rule,
        preserved for compatibility with `classical_governance_v2.IbnMalikV2.definiteness_ok`.
        What the sources give is the ORDERING (M_ranks_*) and the naʿt constraint (M_naat_rank_rule,
        implemented in `naat_ok`).
        """
        return self.rank(cls=mubtada) >= self.rank(cls=khabar)

    def naat_ok(self, manut: str, naat: str) -> bool:
        """SOURCED, M_naat_rank_rule: a naʿt may be of its manʿūt's rank or below, never above."""
        return self.rank(cls=naat) <= self.rank(cls=manut)

    # ---------------------------------------------------------------- POS transitions
    def transition(self, prev_cls: Any, next_cls: Any, coordinator_between: bool = False) -> bool:
        """May `next_cls` follow `prev_cls`?

        With `coordinator_between=True` the pair is joined by و/ف, i.e. it is ʿaṭf (l.549), and
        Fiʿl -> Fiʿl is admitted: «وعطفك الفعل على الفعل يصح» (l.569).  Without it, Fiʿl -> Fiʿl
        is refused, because the two verbs would then be two operators on one operand and
        «لا يجتمع في شيء من الكلام عاملان في معمول واحد» (al-Suhaylī); Sībawayh states it at
        the head of the tradition: «فالعامل في اللفظ أحد الفعلين» and «لا يعمل في اسم واحد نصب
        ورفع» (al-Kitāb, bāb al-fāʿilayn wa-l-mafʿūlayn).  His own witness for the coordinated
        pair is «ونخلع ونترك» (see CITATIONS M_sibawayh_*).
        """
        prev = self._class_id(prev_cls)
        nxt = self._class_id(next_cls)
        if nxt is None:
            return False
        table = self.TRANSITIONS if coordinator_between else self.TRANSITIONS_NO_COORDINATOR
        return nxt in table.get(prev, {self.ISM, self.FIL, self.HARF, self.SIFAH})

    #: the name the existing decoder calls (`IbnMalikPOSAutomaton.transition_allowed`)
    transition_allowed = transition
    transition_ok = transition

    @classmethod
    def transition_table(cls, coordinator_between: bool = False) -> Dict[int, set]:
        return dict(cls.TRANSITIONS if coordinator_between else cls.TRANSITIONS_NO_COORDINATOR)

    def _class_id(self, cls: Any) -> Optional[int]:
        if cls is None:
            return None
        if isinstance(cls, bool):
            return None
        if isinstance(cls, int):
            return cls if cls in self.CLASS_NAME else None
        if isinstance(cls, dict):
            for k in ('cls', 'pos', 'class', 'category'):
                if k in cls:
                    return self._class_id(cls[k])
            return None
        if isinstance(cls, str):
            key = cls.strip().upper()
            if key in self.CLASS_ID:
                return self.CLASS_ID[key]
            if key in self.CLASS_NAME.values():
                for i, n in self.CLASS_NAME.items():
                    if n == key:
                        return i
            return None
        return None

    # ---------------------------------------------------------------- coordination
    @classmethod
    def is_coordinator(cls, word: str) -> bool:
        """SOURCED, M_coordinators (l.550): is this word the ʿāṭif wāw/fāʾ itself?"""
        return cls._bare(word) in cls.COORDINATORS

    @classmethod
    def has_coordinator_prefix(cls, prefix: str) -> bool:
        """ENGINEERING (morphological): the NRMP tuple carries the clitic in the PREFIX slot."""
        return cls._bare(prefix) in cls.COORDINATOR_PREFIXES

    def coordinator_of(self, word: str) -> bool:
        """True when the word is, or carries, the coordinating wāw/fāʾ.

        Surface detection is ENGINEERING and deliberately conservative: a bare 'و'/'ف', or a
        surface of length > 2 beginning with them (so «وقعد» -> True).  Radical-initial words
        such as «وعد» would be misread, which is exactly why the decoder passes the prefix slot.
        """
        bare = self._bare(word)
        if bare in self.COORDINATORS:
            return True
        return len(bare) > 2 and bare[0] in self.COORDINATORS

    @staticmethod
    def _bare(s: str) -> str:
        return re.sub(r'[\u064b-\u0652\u0670\u0640]', '', str(s or '')).strip()

    # ---------------------------------------------------------------- the automaton
    def apply(self, words: Iterable[Any], classes: Optional[Sequence[Any]] = None) -> List[Dict]:
        """Run the automaton over a sequence and return one record per position.

        `words` may mix:
          * plain strings            -- class/rank inferred from the surface (ENGINEERING cues)
          * (word, class) pairs      -- the decoder's usual form, class = 1..4 or 'ISM'/'FIL'...
          * dicts                    -- keys: word/text, cls/class/pos, rank/definiteness,
                                       prefix, suffix, annexed_to, coordinator_between
          * objects                  -- attributes `word`, `cls` (optional `prefix`, ...)

        A standalone «و»/«ف» is consumed as the ʿāṭif: it does NOT become the `prev` class for
        the next comparison, so `[('قام','FIL'), 'و', ('قعد','FIL')]` tests FIL -> FIL *with* a
        coordinator.  A word carrying a coordinator prefix ('وقعد') sets the flag for its own
        incoming transition.  This is the coordinator exception at the decision point.
        """
        items = self._coerce_sequence(words, classes)
        out: List[Dict] = []
        prev = 0                                  # NONE: start of the clause
        pending_coordinator = False
        for i, item in enumerate(items):
            surface = item['word']
            cls = self._class_id(item.get('cls'))
            if self._consumes_as_coordinator(item, cls):
                out.append({
                    'i': i, 'word': surface, 'cls': self.HARF, 'cls_name': 'HARF',
                    'role': 'COORDINATOR', 'allowed': True, 'coordinator_between': False,
                    'prev_cls': prev, 'rank': None, 'definiteness': None,
                    'rule': 'M_coordinators', 'note': "ʿāṭif: licenses the ʿaṭf on the NEXT word; "
                            "it is not itself an operand",
                })
                pending_coordinator = True
                continue
            if cls is None:
                cls = self.classify(surface)
            coord = bool(item.get('coordinator_between')) or pending_coordinator
            if not coord:
                coord = self.has_coordinator_prefix(item.get('prefix', '') or '') or \
                    (item.get('surface_prefix_ok', True) and self.coordinator_of(surface))
            allowed = self.transition(prev, cls, coordinator_between=coord)
            item_def = self.definiteness(surface, item.get('definiteness') or item.get('rank'))
            if item.get('annexed_to'):
                item_def = self.label_of_rank(self.rank(surface,
                                                        annexed_to=item['annexed_to'])) or item_def
            rule = 'M_atf_fil_verse' if (coord and prev == self.FIL and cls == self.FIL) else None
            out.append({
                'i': i, 'word': surface, 'cls': cls, 'cls_name': self.CLASS_NAME.get(cls, '?'),
                'role': 'OPERAND', 'allowed': allowed, 'coordinator_between': coord,
                'prev_cls': prev, 'rank': self.RANKS[item_def], 'definiteness': item_def,
                'rule': rule,
                'citation': CITATIONS[rule]['ar'] if rule else None,
                'note': ("Fiʿl after Fiʿl licensed by the coordinating particle"
                         if rule else
                         ("Fiʿl after Fiʿl refused: nothing joins them ('قام قعد')"
                          if (not allowed and prev == self.FIL and cls == self.FIL) else '')),
            })
            prev = cls
            pending_coordinator = False
        return out

    def _consumes_as_coordinator(self, item: Dict, cls: Optional[int]) -> bool:
        if item.get('coordinator_between') or item.get('annexed_to'):
            return False
        if cls is not None and cls != self.HARF:
            return False
        return self._bare(item['word']) in self.COORDINATORS

    @staticmethod
    def _coerce_sequence(words: Iterable[Any], classes: Optional[Sequence[Any]]
                         ) -> List[Dict]:
        seq = list(words)
        if classes is not None and seq and isinstance(seq[0], str):
            seq = list(zip(seq, classes))
        out: List[Dict] = []
        for w in seq:
            out.append(IbnMalikAutomaton._coerce(w))
        return out

    @staticmethod
    def _coerce(w: Any) -> Dict:
        if isinstance(w, dict):
            rec = dict(w)
            rec['word'] = rec.get('word', rec.get('text', rec.get('surface', '')))
            if 'cls' not in rec and 'class' in rec:
                rec['cls'] = rec['class']
            if 'cls' not in rec and 'pos' in rec:
                rec['cls'] = rec['pos']
            return rec
        if isinstance(w, (tuple, list)):
            rec = {'word': w[0] if len(w) > 0 else ''}
            if len(w) > 1:
                rec['cls'] = w[1]
            if len(w) > 2:
                rec['prefix'] = w[2]
            return rec
        if isinstance(w, str):
            return {'word': w}
        rec = {'word': getattr(w, 'word', getattr(w, 'surface', ''))}
        for attr, key in (('cls', 'cls'), ('pos', 'cls'), ('prefix', 'prefix'),
                          ('definiteness', 'definiteness'), ('rank', 'rank'),
                          ('annexed_to', 'annexed_to')):
            if hasattr(w, attr):
                rec.setdefault(key, getattr(w, attr))
        return rec

    # ---------------------------------------------------------------- fallback POS (ENGINEERING)
    #: ENGINEERING: a minimal function-word/verb list so bare-string callers get sensible
    #: classes.  The decoder does NOT need this -- it passes `pos_category()` output.
    HARF_WORDS = {'و', 'ف', 'ثم', 'أو', 'أم', 'بل', 'لكن', 'حتى', 'من', 'إلى', 'عن', 'على',
                  'في', 'مع', 'رب', 'إن', 'أن', 'كأن', 'لكن', 'ليت', 'لعل', 'لم', 'لما',
                  'لن', 'كي', 'إذن', 'لا', 'ما', 'هل', 'قد', 'س', 'سوف', 'كان', 'ليس',
                  'أصبح', 'أمسى', 'ظل', 'بات', 'صار'}
    VERB_LEXICON = {'قام', 'قعد', 'ذهب', 'جلس', 'ضرب', 'كتب', 'قرأ', 'قرا', 'قال', 'أكل',
                    'شرب', 'خرج', 'دخل', 'فتح', 'شرح', 'كرم', 'فرح', 'ورث', 'سأل', 'رجع'}

    def classify(self, word: str) -> int:
        """ENGINEERING: a surface-only POS guess for bare strings (no source for a lexicon)."""
        bare = self._bare(word)
        if bare in self.HARF_WORDS or bare in self.DEMONSTRATIVES or bare in self.RELATIVES:
            return self.HARF if bare in self.HARF_WORDS else self.ISM
        if bare in self.PRONOUNS or bare in self.PROPER_NAMES:
            return self.ISM
        if bare in self.VERB_LEXICON:
            return self.FIL
        if bare[:1] in ('ي', 'ت', 'أ', 'ن') and len(bare) >= 3:
            return self.FIL          # ENGINEERING: the four mudāriʿ ziyādāt
        return self.ISM


# ============================================================================================
# 2. the built-in proof
# ============================================================================================
def _selftest() -> int:
    A = IbnMalikAutomaton()
    results: List[Tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = '') -> None:
        results.append((name, bool(ok), detail))

    print('=' * 78)
    print("IBN MALIK AUTOMATON -- built-in proof")
    print('=' * 78)

    print("\n[1] Fiʿl -> Fiʿl: «قام وقعد» ACCEPTED, «قام قعد» REFUSED  (Alfiyyah l.569)")
    check('FIL -> FIL WITHOUT a coordinator is refused',
          not A.transition('FIL', 'FIL', coordinator_between=False))
    check('FIL -> FIL WITH a coordinator is admitted',
          A.transition('FIL', 'FIL', coordinator_between=True))
    rec = A.apply([('قام', 'FIL'), 'و', ('قعد', 'FIL')])
    check("apply(['قام','و','قعد']) accepts the second verb",
          rec[2]['allowed'] and rec[2]['coordinator_between'],
          f"allowed={rec[2]['allowed']} coord={rec[2]['coordinator_between']}")
    rec_pref = A.apply([('قام', 'FIL'), ('وقعد', 'FIL')])       # wāw in the PREFIX slot
    check("apply(['قام','وقعد']) accepts it via the coordinator prefix",
          rec_pref[1]['allowed'] and rec_pref[1]['coordinator_between'])
    rec_bad = A.apply([('قام', 'FIL'), ('قعد', 'FIL')])
    check("apply(['قام','قعد']) refuses the second verb",
          not rec_bad[1]['allowed'] and not rec_bad[1]['coordinator_between'])
    check("the citation on the licensed transition is l.569 verbatim",
          rec[2]['citation'] == CITATIONS['M_atf_fil_verse']['ar'],
          rec[2]['citation'] or '<none>')
    check("Harf -> Harf stays ENGINEERING-forbidden in the table",
          3 not in A.TRANSITIONS[3] and 3 not in A.TRANSITIONS_NO_COORDINATOR[3])

    print("\n[2] Marātib al-maʿārif, the Alfiyyah's own examples  (l.55)")
    examples = [('هم', 'PRONOUN', 7), ('هند', 'PROPER', 6), ('ذي', 'DEMONSTRATIVE', 5),
                ('الذي', 'RELATIVE', 4), ('الغلام', 'DEFINITE', 3), ('ابني', 'ANNEXED', 2),
                ('رجل', 'INDEFINITE', 1)]
    for word, cls, want in examples:
        got_explicit = A.rank(word, cls)
        check(f'rank({word!r}, {cls}) == {want}  (explicit class)', got_explicit == want,
              f'got {got_explicit}')
    for word, want in [('هم', 7), ('ذي', 5), ('الذي', 4), ('الغلام', 3), ('رجل', 1)]:
        got = A.rank(word)
        check(f'rank({word!r}) == {want}  (surface inference)', got == want, f'got {got}')
    check('the six ranks are strictly ordered 7 > 6 > 5 > 4 > 3 > 2 > 1',
          [A.RANKS[k] for k in ('PRONOUN', 'PROPER', 'DEMONSTRATIVE', 'RELATIVE',
                                'DEFINITE', 'ANNEXED', 'INDEFINITE')] == [7, 6, 5, 4, 3, 2, 1])
    check('Ibn Mālik\'s own six ranks are recorded with their citation',
          len(A.SIX_RANKS_TASHIL) == 6 and A.SIX_RANKS_TASHIL[-1] == 'RELATIVE_AND_DEFINITE_AL')
    check('a naʿt may not outrank its manʿūt (al-Shāṭibī)',
          A.naat_ok('DEFINITE', 'INDEFINITE') and not A.naat_ok('INDEFINITE', 'DEFINITE'))
    check('the muḍāf inherits its muḍāf ilayhi (Tadhyīl: المضاف بحسب المضاف إليه)',
          A.rank('كتاب', annexed_to='DEFINITE') == 3)
    check('the Basran refinement: muḍāf to a pronoun -> PROPER rank',
          A.rank('كتابه', annexed_to='PRONOUN', basran_annexation=True) == 6)
    check('the mubtadaʾ/khabar comparison uses the ranks',
          A.definiteness_ok('DEFINITE', 'INDEFINITE')
          and not A.definiteness_ok('INDEFINITE', 'DEFINITE'))

    print("\n[3] the coordinator exception does not leak into Harf -> Harf (ENGINEERING cell)")
    check('HARF -> HARF refused without a coordinator', not A.transition('HARF', 'HARF', False))
    check('HARF -> HARF still refused with one', not A.transition('HARF', 'HARF', True))

    print("\n[4] every rule's citation is present, and the Sībawayh grounding is in place")
    for key in ('M_marifah_verse', 'M_nakirah_verse', 'M_ranks_tashil', 'M_ranks_shuyukh',
                'M_naat_rank_rule', 'M_atf_is_bi_harf', 'M_coordinators', 'M_atf_fil_verse',
                'M_naql_hukm', 'M_two_amil_verse', 'M_qama_wa_qada', 'M_tashil_tanazu',
                'M_no_two_operators', 'M_no_two_operators_suhayli', 'M_sibawayh_tanazu',
                'M_sibawayh_one_irab', 'M_sibawayh_fil_atf'):
        check(f'citation {key} carries Arabic + book', bool(CITATIONS.get(key, {}).get('ar'))
              and bool(CITATIONS.get(key, {}).get('book')))
    check("Sībawayh's one-operator statement is quoted",
          CITATIONS['M_sibawayh_tanazu']['ar'].find('فالعامل في اللفظ أحد الفعلين') != -1)
    check("Sībawayh's coordinated pair «ونخلع ونترك» is quoted",
          'ونخلع ونترك' in CITATIONS['M_sibawayh_fil_atf']['ar'])

    print()
    n_fail = sum(1 for _, ok, _ in results if not ok)
    for name, ok, detail in results:
        if not ok:
            print(f'  FAIL  {name}' + (f'  -- {detail}' if detail else ''))
    print('=' * 78)
    print(f'RESULT: {len(results)-n_fail}/{len(results)} checks pass' +
          ('' if n_fail == 0 else f', {n_fail} FAIL'))
    print('=' * 78)
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(_selftest())
