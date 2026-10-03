# Failure taxonomy -- NRMP morphemic tokenizer, mode-1 shipped default

Harness: `dump_failures.py` on the published sample (`eval_heldout_sample.json`; the same
92 files / 3,298,362 occurrences / 188,722 distinct forms that produced the published
numbers).  Baseline reproduced exactly:

| | |
|---|---|
| type round-trip | **57.5884 %** (108,682 / 188,722) |
| occurrence round-trip | **82.1065 %** (2,708,170 / 3,298,362) |
| failing distinct forms | **80,040** (42.4116 % of types) |

Every failing type is assigned exactly once by a first-match cascade; the only residue
bucket is the explicit `OTHER`, whose members are listed below.

| cause | types | % of failing types | occurrences | % of failing occurrences | % of all 188,722 types |
|---|---|---|---|---|---|
| `CONTROL_root_slot_control_token` | 67351 | 84.15% | 426547 | 72.27% | 35.6879% |
| `WRONG_WAZN_same_root_other_pattern` | 4820 | 6.02% | 42850 | 7.26% | 2.5540% |
| `REALISER_wazn_unimplemented_engine_ok` | 2630 | 3.29% | 24501 | 4.15% | 1.3936% |
| `WRONG_AFFIX_surface_letter_unaccounted` | 1879 | 2.35% | 11029 | 1.87% | 0.9956% |
| `ANALYZER_no_wazn_assigned` | 1744 | 2.18% | 19750 | 3.35% | 0.9241% |
| `ANALYZER_particle_overcapture` | 1301 | 1.63% | 18110 | 3.07% | 0.6894% |
| `OTHER` | 234 | 0.29% | 9758 | 1.65% | 0.1240% |
| `CLITIC_MISSPLIT_slot_off_the_edge` | 25 | 0.03% | 546 | 0.09% | 0.0132% |
| `PUNCTUATION_tokenisation_artefact` | 18 | 0.02% | 18 | 0.00% | 0.0095% |
| `ANALYZER_wrong_root` | 16 | 0.02% | 2180 | 0.37% | 0.0085% |
| `DIVINE_passthrough_by_design` | 13 | 0.02% | 34880 | 5.91% | 0.0069% |
| `ORTHOGRAPHIC_hamza_or_weak_spelling` | 9 | 0.01% | 23 | 0.00% | 0.0048% |
| **total** | **80040** | **100%** | **590192** | **100%** | **42.4116%** |

## Representability summary

| stratum | types | % of 188,722 | occurrences |
|---|---|---|---|
| UNREPRESENTABLE BY CONSTRUCTION (`<DIVINE>` + control root slot) | 67364 | 35.6948% | 461427 |
| **ANALYZER-ATTRIBUTABLE** | **12676** | **6.7168%** | **128765** |

Round-trip charged to the analyzer: **93.2832 %** (176046 / 188722 representable types).

Control-slot sub-census: `{'root_<PARTICLE>': 67307, 'root_<UNK>': 44}`

### The control bucket is NOT "genuinely closed-class"

`CONTROL_closed_class_by_design` = **0 types**: the design's OWN closed-class route
(`nrmp_vocab._closed_class_split`, against the 265-surface authored inventory) accepts
**zero** of the control-slot failures.  Every one of the 67351 types is a word the design
declines, i.e. an analyzer that returned no analysis at all.  The class is unrepresentable
BY CONSTRUCTION OF THE TUPLE -- the surface is simply not carried -- and it is excluded
from the analyzer-charged rate above as the brief requires.  It must NOT be read as "the
design intended these to be particles": it is the analyzer's single largest defect.
Top members are ordinary open-class words:

* `تعالى` n=9942
* `صلى` n=6082
* `الأول` n=4454
* `أبو` n=3636
* `ثنا` n=3503
* `آخر` n=3354
* `قيل` n=3186
* `معنى` n=2678
* `جهة` n=2548
* `ب` n=2531
* `ا` n=2398
* `شيئا` n=2256
* `الآخر` n=2214
* `الإنسان` n=2202
* `ج` n=2102

## `CONTROL_root_slot_control_token` -- 67351 types (84.15%), 426547 occurrences

* `تعالى` n=9942  p=<NONE> r=<PARTICLE> wz=<NONE> s=<NONE> -> ``
* `صلى` n=6082  p=<NONE> r=<PARTICLE> wz=<NONE> s=<NONE> -> ``
* `الأول` n=4454  p=<NONE> r=<PARTICLE> wz=<NONE> s=<NONE> -> ``
* `أبو` n=3636  p=<NONE> r=<PARTICLE> wz=<NONE> s=<NONE> -> ``
* `ثنا` n=3503  p=<NONE> r=<PARTICLE> wz=<NONE> s=<NONE> -> ``
* `آخر` n=3354  p=<NONE> r=<PARTICLE> wz=<NONE> s=<NONE> -> ``
* `قيل` n=3186  p=<NONE> r=<PARTICLE> wz=<NONE> s=<NONE> -> ``
* `معنى` n=2678  p=<NONE> r=<PARTICLE> wz=<NONE> s=<NONE> -> ``
* `جهة` n=2548  p=<NONE> r=<PARTICLE> wz=<NONE> s=<NONE> -> ``
* `ب` n=2531  p=<NONE> r=<UNK> wz=<NONE> s=<NONE> -> ``
* `ا` n=2398  p=<NONE> r=<UNK> wz=<NONE> s=<NONE> -> ``
* `شيئا` n=2256  p=<NONE> r=<PARTICLE> wz=<NONE> s=<NONE> -> ``

## `WRONG_WAZN_same_root_other_pattern` -- 4820 types (6.02%), 42850 occurrences

* `أهل` n=2745  p=<NONE> r=هلل wz=يَفْعُلُ s=<NONE> -> `هلل`  [engine=يهل]  [alt_wazn=أَفْعَل]
* `الأرض` n=1847  p=ال r=رضض wz=يَفْعُلُ s=<NONE> -> `الرضض`  [engine=يرض]  [alt_wazn=أَفْعَل]
* `أخذ` n=827  p=<NONE> r=خذذ wz=يَفْعُلُ s=<NONE> -> `خذذ`  [engine=يخذ]  [alt_wazn=أَفْعَل]
* `أعني` n=779  p=<NONE> r=عنن wz=يَفْعُلُ s=ي -> `عنني`  [engine=يعن]  [alt_wazn=أَفْعَل]
* `الاشياء` n=723  p=ال r=شيء wz=إِفْعَال s=<NONE> -> `الإشياء`  [engine=إشياء]  [alt_wazn=اِفْعَالَّ]
* `أقل` n=604  p=<NONE> r=قلل wz=يَفْعُلُ s=<NONE> -> `قلل`  [engine=يقل]  [alt_wazn=أَفْعَل]
* `العدد` n=574  p=ال r=عدد wz=فَعَلَ s=<NONE> -> `العد`  [engine=عد]  [alt_wazn=فَعْلَل]
* `عدد` n=526  p=<NONE> r=عدد wz=فَعَلَ s=<NONE> -> `عد`  [engine=عد]  [alt_wazn=فَعْلَل]
* `تدل` n=417  p=<NONE> r=دلل wz=يَفْعُلُ s=<NONE> -> `دلل`  [engine=يدل]  [alt_wazn=تَفَعَّلَ]
* `والأرض` n=416  p=وال r=رضض wz=يَفْعُلُ s=<NONE> -> `والرضض`  [engine=يرض]  [alt_wazn=أَفْعَل]
* `لأجل` n=391  p=ل r=جلل wz=يَفْعُلُ s=<NONE> -> `لجلل`  [engine=يجل]  [alt_wazn=أَفْعَل]
* `أقر` n=361  p=<NONE> r=قرر wz=يَفْعُلُ s=<NONE> -> `قرر`  [engine=يقر]  [alt_wazn=أَفْعَل]

## `REALISER_wazn_unimplemented_engine_ok` -- 2630 types (3.29%), 24501 occurrences

* `يكن` n=3998  p=<NONE> r=كنن wz=يَفْعُلُ s=<NONE> -> `كنن`  [engine=يكن]
* `يدل` n=1823  p=<NONE> r=دلل wz=يَفْعُلُ s=<NONE> -> `دلل`  [engine=يدل]
* `يصح` n=1096  p=<NONE> r=صحح wz=يَفْعُلُ s=<NONE> -> `صحح`  [engine=يصح]
* `أبدا` n=579  p=<NONE> r=بدا wz=أَفْعَلَ s=<NONE> -> `أبدأ`  [engine=أبدا]
* `يعني` n=512  p=<NONE> r=عنن wz=يَفْعُلُ s=ي -> `عنني`  [engine=يعن]
* `يجز` n=489  p=<NONE> r=جزز wz=يَفْعُلُ s=<NONE> -> `جزز`  [engine=يجز]
* `متناهية` n=475  p=<NONE> r=نهي wz=مُتَفَاعِل s=ة -> `نهية`  [engine=متناهي]
* `يحل` n=461  p=<NONE> r=حلل wz=يَفْعُلُ s=<NONE> -> `حلل`  [engine=يحل]
* `يزل` n=432  p=<NONE> r=زلل wz=يَفْعُلُ s=<NONE> -> `زلل`  [engine=يزل]
* `يظن` n=410  p=<NONE> r=ظنن wz=يَفْعُلُ s=<NONE> -> `ظنن`  [engine=يظن]
* `يقل` n=258  p=<NONE> r=قلل wz=يَفْعُلُ s=<NONE> -> `قلل`  [engine=يقل]
* `يبق` n=246  p=<NONE> r=بقق wz=يَفْعُلُ s=<NONE> -> `بقق`  [engine=يبق]

## `WRONG_AFFIX_surface_letter_unaccounted` -- 1879 types (2.35%), 11029 occurrences

* `النبي` n=2369  p=ال r=بيي wz=يَفْعُلُ s=<NONE> -> `البيي`  [engine=يبيي]
* `المتقدم` n=280  p=ال r=تقدم wz=مُتَفَعْلِل s=<NONE> -> `القدم`
* `تختلف` n=249  p=<NONE> r=خلف wz=يَفْتَعِلُ s=<NONE> -> `يختلف`  [engine=يختلف]
* `الترتيب` n=231  p=ال r=ريب wz=يَفْتَعِلُ s=<NONE> -> `اليرتيب`  [engine=يرتاب]
* `تتحرك` n=227  p=<NONE> r=حرك wz=يَتَفَعَّلُ s=<NONE> -> `يتحرك`  [engine=يتحرك]
* `ترتيب` n=185  p=<NONE> r=ريب wz=يَفْتَعِلُ s=<NONE> -> `يرتيب`  [engine=يرتاب]
* `تستعمل` n=165  p=<NONE> r=عمل wz=اِسْتَفْعَلَ s=<NONE> -> `استعمل`  [engine=استعمل]
* `تحتاج` n=155  p=<NONE> r=حوج wz=اِفْتَعَلَ s=<NONE> -> `احتوج`  [engine=احتاج]
* `مجازا` n=145  p=<NONE> r=جزا wz=مَفَاعِل s=<NONE> -> `مجازأ`  [engine=مجازي]
* `نبي` n=143  p=<NONE> r=بيي wz=يَفْعُلُ s=<NONE> -> `بيي`  [engine=يبيي]
* `تتعلق` n=141  p=<NONE> r=علق wz=يَتَفَعَّلُ s=<NONE> -> `يتعلق`  [engine=يتعلق]
* `المتقدمة` n=137  p=ال r=تقدم wz=مُتَفَعْلِل s=ة -> `القدمة`

## `ANALYZER_no_wazn_assigned` -- 1744 types (2.18%), 19750 occurrences

* `يجب` n=2832  p=<NONE> r=وجب wz=<UNK> s=<NONE> -> `وجب`
* `يقع` n=1204  p=<NONE> r=وقع wz=<UNK> s=<NONE> -> `وقع`
* `الهواء` n=840  p=ال r=هوا wz=<UNK> s=<NONE> -> `الهوا`
* `فيجب` n=703  p=ف r=وجب wz=<UNK> s=<NONE> -> `فوجب`
* `يرد` n=586  p=<NONE> r=ورد wz=<UNK> s=<NONE> -> `ورد`
* `ألف` n=475  p=<NONE> r=ولف wz=<UNK> s=<NONE> -> `ولف`
* `نسلم` n=423  p=<NONE> r=سلم wz=<UNK> s=<NONE> -> `سلم`
* `نعلم` n=370  p=<NONE> r=علم wz=<UNK> s=<NONE> -> `علم`
* `يصلي` n=319  p=<NONE> r=وصل wz=<UNK> s=ي -> `وصلي`
* `يجد` n=293  p=<NONE> r=وجد wz=<UNK> s=<NONE> -> `وجد`
* `تقع` n=267  p=<NONE> r=وقع wz=<UNK> s=<NONE> -> `وقع`
* `ويجب` n=267  p=و r=وجب wz=<UNK> s=<NONE> -> `ووجب`

## `ANALYZER_particle_overcapture` -- 1301 types (1.63%), 18110 occurrences

* `لكان` n=1094  p=<NONE> r=<P:لكن> wz=فِعَال s=<NONE> -> `لكن`
* `تبين` n=913  p=<NONE> r=<P:بين> wz=تَفَعَّلَ s=<NONE> -> `بين`
* `بيان` n=885  p=<NONE> r=<P:بين> wz=فِعَال s=<NONE> -> `بين`
* `يقبل` n=678  p=<NONE> r=<P:قبل> wz=يَفْعَلُ s=<NONE> -> `قبل`
* `البيان` n=409  p=ال r=<P:بين> wz=فِعَال s=<NONE> -> `البين`
* `قبول` n=340  p=<NONE> r=<P:قبل> wz=فُعُول s=<NONE> -> `قبل`
* `تقبل` n=334  p=<NONE> r=<P:قبل> wz=تَفَعَّلَ s=<NONE> -> `قبل`
* `يبعد` n=296  p=<NONE> r=<P:بعد> wz=يَفْعَلُ s=<NONE> -> `بعد`
* `لكونه` n=270  p=<NONE> r=<P:لكن> wz=فُعُول s=ه -> `لكنه`
* `تغير` n=266  p=<NONE> r=<P:غير> wz=تَفَعَّلَ s=<NONE> -> `غير`
* `يتبين` n=208  p=<NONE> r=<P:بين> wz=يَتَفَعَّلُ s=<NONE> -> `بين`
* `يتغير` n=206  p=<NONE> r=<P:غير> wz=يَتَفَعَّلُ s=<NONE> -> `غير`

## `OTHER` -- 234 types (0.29%), 9758 occurrences

* `أبي` n=5477  p=<NONE> r=بيي wz=يَفْعُلُ s=<NONE> -> `بيي`  [engine=يبيي]
* `أبيه` n=1418  p=<NONE> r=بيي wz=يَفْعُلُ s=ه -> `بييه`  [engine=يبيي]
* `تقدم` n=878  p=<NONE> r=تقدم wz=فَعْلَلَ s=<NONE> -> `قدم`
* `وأبي` n=190  p=و r=بيي wz=يَفْعُلُ s=<NONE> -> `وبيي`  [engine=يبيي]
* `وبالله` n=180  p=و r=بلل wz=فَاعِل s=ه -> `وباله`  [engine=بال]
* `لأبي` n=138  p=ل r=بيي wz=يَفْعُلُ s=<NONE> -> `لبيي`  [engine=يبيي]
* `التقدم` n=128  p=ال r=تقدم wz=فَعْلَلَ s=<NONE> -> `القدم`
* `نعني` n=117  p=<NONE> r=عنن wz=يَفْعُلُ s=ي -> `عنني`  [engine=يعن]
* `ماتت` n=96  p=<NONE> r=متت wz=فَاعِل s=<NONE> -> `مات`  [engine=مات]
* `تقدمت` n=51  p=<NONE> r=تقدم wz=فَعْلَلَ s=ت -> `قدمت`
* `اعضاء` n=50  p=<NONE> r=عضا wz=أَفْعَال s=<NONE> -> `أعضاو`  [engine=أعضاا]
* `أيما` n=49  p=<NONE> r=يمم wz=يَفْعُلُ s=ا -> `يمما`  [engine=يمم]

## `CLITIC_MISSPLIT_slot_off_the_edge` -- 25 types (0.03%), 546 occurrences

* `النهي` n=286  p=ال r=نهي wz=فِعَال s=ة -> `النهاية`
* `والنهي` n=112  p=وال r=نهي wz=فِعَال s=ة -> `والنهاية`
* `نهاه` n=24  p=<NONE> r=نهي wz=فِعَال s=ة -> `نهاية`
* `بالنهي` n=22  p=بال r=نهي wz=فِعَال s=ة -> `بالنهاية`
* `فنهاه` n=14  p=ف r=نهي wz=فِعَال s=ة -> `فنهاية`
* `ونهي` n=14  p=و r=نهي wz=فِعَال s=ة -> `ونهاية`
* `للنهي` n=13  p=لل r=نهي wz=فِعَال s=ة -> `للنهاية`
* `نهاكم` n=13  p=<NONE> r=نهي wz=فِعَال s=ة -> `نهاية`
* `ونهاهم` n=7  p=و r=نهي wz=فِعَال s=ة -> `ونهاية`
* `فنهاهم` n=6  p=ف r=نهي wz=فِعَال s=ة -> `فنهاية`
* `ونهاه` n=5  p=و r=نهي wz=فِعَال s=ة -> `ونهاية`
* `النها` n=5  p=ال r=نهي wz=فِعَال s=ة -> `النهاية`

## `PUNCTUATION_tokenisation_artefact` -- 18 types (0.02%), 18 occurrences

* `أهل` n=1  p=<NONE> r=اهل wz=فَعَلَ s=<NONE> -> `اهل`
* `عجبأ` n=1  p=<NONE> r=عجب wz=فَعَلَ s=ا -> `عجبا`
* `أكل` n=1  p=<NONE> r=اكل wz=فَعَلَ s=<NONE> -> `اكل`
* `يأكل` n=1  p=<NONE> r=اكل wz=يَفْعَلُ s=<NONE> -> `ياكل`
* `بأمر` n=1  p=ب r=امر wz=فَعَلَ s=<NONE> -> `بامر`
* `أرأيتم` n=1  p=<NONE> r=اري wz=فِعَال s=تم -> `اراءتم`
* `المرأة` n=1  p=ال r=مرا wz=فَعَلَ s=ة -> `المراة`
* `للآخر` n=1  p=لل r=اخر wz=فَعَلَ s=<NONE> -> `للاخر`
* `امرأته` n=1  p=<NONE> r=مرت wz=إِفْعَال s=ه -> `إمراته`
* `أمسك` n=1  p=<NONE> r=امس wz=فَعَلَ s=ك -> `امسك`
* `الأامة` n=1  p=ال r=امم wz=فَاعِل s=ة -> `الاامة`
* `أكان` n=1  p=<NONE> r=اكك wz=فَعَلَ s=ان -> `اكان`

## `ANALYZER_wrong_root` -- 16 types (0.02%), 2180 occurrences

* `لله` n=1185  p=ل r=اله wz=فَعَلَ s=<NONE> -> `لأله`  [engine=اله]  [alt_root=لهه]
* `قائم` n=307  p=<NONE> r=قوم wz=فَاعِل s=<NONE> -> `قاوم`  [engine=قاوم]  [alt_root=قئم]
* `قائمة` n=254  p=<NONE> r=قوم wz=فَاعِل s=ة -> `قاومة`  [engine=قاوم]  [alt_root=قئم]
* `القائم` n=126  p=ال r=قوم wz=فَاعِل s=<NONE> -> `القاوم`  [engine=قاوم]  [alt_root=قئم]
* `أعضاء` n=90  p=<NONE> r=عضا wz=أَفْعَال s=<NONE> -> `أعضاو`  [engine=أعضاا]  [alt_root=عضء]
* `القائمة` n=86  p=ال r=قوم wz=فَاعِل s=ة -> `القاومة`  [engine=قاوم]  [alt_root=قئم]
* `المستفاد` n=59  p=ال r=فيد wz=مُسْتَفْعَل s=<NONE> -> `الفيد`  [engine=مستفيد]  [alt_root=فاد]
* `مستفاد` n=31  p=<NONE> r=فيد wz=مُسْتَفْعَل s=<NONE> -> `فيد`  [engine=مستفيد]  [alt_root=فاد]
* `مستفادة` n=21  p=<NONE> r=فيد wz=مُسْتَفْعَل s=ة -> `فيدة`  [engine=مستفيد]  [alt_root=فاد]
* `المستفادة` n=8  p=ال r=فيد wz=مُسْتَفْعَل s=ة -> `الفيدة`  [engine=مستفيد]  [alt_root=فاد]
* `قائمين` n=4  p=<NONE> r=قوم wz=فَاعِل s=ين -> `قاومين`  [engine=قاوم]  [alt_root=قئم]
* `القائمين` n=4  p=ال r=قوم wz=فَاعِل s=ين -> `القاومين`  [engine=قاوم]  [alt_root=قئم]

## `DIVINE_passthrough_by_design` -- 13 types (0.02%), 34880 occurrences

* `الله` n=31938  p=<NONE> r=اله wz=فَعَلَ s=<NONE> -> `أله`
* `والله` n=1880  p=و r=اله wz=فَعَلَ s=<NONE> -> `وأله`
* `بالله` n=985  p=<NONE> r=بلل wz=فَاعِل s=ه -> `باله`
* `فالله` n=57  p=ف r=اله wz=فَعَلَ s=<NONE> -> `فأله`
* `إلله` n=6  p=<NONE> r=<PARTICLE> wz=<NONE> s=<NONE> -> ``
* `ألله` n=5  p=<NONE> r=<PARTICLE> wz=<NONE> s=<NONE> -> ``
* `بألله` n=3  p=<NONE> r=<PARTICLE> wz=<NONE> s=<NONE> -> ``
* `واللله` n=1  p=<NONE> r=<PARTICLE> wz=<NONE> s=<NONE> -> ``
* `وألله` n=1  p=<NONE> r=<PARTICLE> wz=<NONE> s=<NONE> -> ``
* `اللله` n=1  p=<NONE> r=<PARTICLE> wz=<NONE> s=<NONE> -> ``
* `سالله` n=1  p=<NONE> r=سلل wz=فَاعِل s=ه -> `ساله`
* `للإلله` n=1  p=<NONE> r=<PARTICLE> wz=<NONE> s=<NONE> -> ``

## `ORTHOGRAPHIC_hamza_or_weak_spelling` -- 9 types (0.01%), 23 occurrences

* `الاحاد` n=6  p=ال r=وحد wz=أَفْعَال s=<NONE> -> `الآحاد`  [engine=أوحاد]
* `احاد` n=5  p=<NONE> r=وحد wz=أَفْعَال s=<NONE> -> `آحاد`  [engine=أوحاد]
* `جوازا` n=4  p=<NONE> r=جزا wz=فَوَاعِل s=<NONE> -> `جوازأ`  [engine=جوازي]
* `الأحاد` n=2  p=ال r=وحد wz=أَفْعَال s=<NONE> -> `الآحاد`  [engine=أوحاد]
* `لاحاد` n=2  p=ل r=وحد wz=أَفْعَال s=<NONE> -> `لآحاد`  [engine=أوحاد]
* `للاحاد` n=1  p=لل r=وحد wz=أَفْعَال s=<NONE> -> `للآحاد`  [engine=أوحاد]
* `احاده` n=1  p=<NONE> r=وحد wz=أَفْعَال s=ه -> `آحاده`  [engine=أوحاد]
* `احادا` n=1  p=<NONE> r=وحد wz=أَفْعَال s=ا -> `آحادا`  [engine=أوحاد]
* `أحاد` n=1  p=<NONE> r=وحد wz=أَفْعَال s=<NONE> -> `آحاد`  [engine=أوحاد]

