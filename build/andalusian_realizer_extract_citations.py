#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract VERBATIM citation spans from the local corpus into citations.json.

Every span is located by (start_pattern, end_pattern) over diacritic-stripped text and
then sliced out of the RAW file, so the Arabic is copied byte-for-byte from the source.
OpenITI edition markup is then removed transparently (page markers, ms markers, line
continuation tildes, comment hashes) and the raw line number is recorded so any reader
can re-check the quotation in the file.
"""
import json
import re
import sys

DIAC = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]')
ALEF = str.maketrans({'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ى': 'ي', 'ؤ': 'و', 'ئ': 'ي', 'ة': 'ه'})

CORPUS = '/home/grem3/Documents/deepseek-harness/default-workspace/rootformer/corpus'

# (key, file, book, start_pattern, end_pattern, note)
SPANS = [
    # ---------------- Sibawayh, al-Kitab : the verb's government ----------------
    ('sib_verb_government', 'basran/Sibawayh_Al_Kitab.txt', 'سيبويه، الكتاب',
     'هذا باب الفاعل الذي يتعداه فعله الي مفعول وذلك قولك ضرب عبد الله زيدا',
     'وانتصب زيد لانه مفعول تعدي اليه فعل الفاعل', ''),
    ('sib_iktifa_principle', 'basran/Sibawayh_Al_Kitab.txt', 'سيبويه، الكتاب',
     'اعلم انهم مما يحذفون الكلم وان كان اصله في الكلام غير ذلك',
     'استغنوا عنها بترك', ''),
    ('sib_clitic_is_fail', 'basran/Sibawayh_Al_Kitab.txt', 'سيبويه، الكتاب',
     'فلذلك احتجت فيه الي فاعل ومفعول',
     'لان المضمر في ضارب هو الفاعل', ''),
    ('sib_clitic_occupies_verb', 'basran/Sibawayh_Al_Kitab.txt', 'سيبويه، الكتاب',
     'وانما قلت زيدا اضربه',
     'فلا يستغني عن الاضمار ان لم يظهر', ''),
    ('sib_pronoun_host_dependence', 'basran/Sibawayh_Al_Kitab.txt', 'سيبويه، الكتاب',
     'واعلم ان حذف النون والتنوين لازم مع علامه المضمر غير المنفصل',
     'حتي يكون متصلا بفعل قبله او باسم فيه ضمير', ''),
    ('sib_pronoun_after_prep', 'basran/Sibawayh_Al_Kitab.txt', 'سيبويه، الكتاب',
     'فان قلت زيد مررت به فهو من النصب ابعد من ذلك',
     'ولم يوصل اليه الفعل في اللفظ', ''),
    ('sib_jazm_naqis', 'basran/Sibawayh_Al_Kitab.txt', 'سيبويه، الكتاب',
     'واعلم ان الاخر اذا كان يسكن في الرفع حذف في الجزم',
     'تقول هو يرمي ويغزو ويخشي', ''),
    ('sib_jazm_ajwaf', 'basran/Sibawayh_Al_Kitab.txt', 'سيبويه، الكتاب',
     'ومثل ذلك لم يبع ولم يقل',
     'لاجريت مجري لم يخف', ''),
    ('sib_jazm_clitic_protects', 'basran/Sibawayh_Al_Kitab.txt', 'سيبويه، الكتاب',
     'باب ما تلحقه الهاء في الوقف لتحرك اخر الحرف',
     'ولم يقضه ولم يرضه', ''),
    ('sib_energetic_nun', 'basran/Sibawayh_Al_Kitab.txt', 'سيبويه، الكتاب',
     'باب احوال الحروف التي قبل النون الخفيفه والثقيله',
     'اعلمن ذلك واكرمن زيدا', ''),
    # ---------------- Ibn 'Usfur : i'rab marks and the diptote ----------------
    ('usfur_jazm_marks', 'andalusian/14_IbnUsfur_Sharh_Jumal_al_Zajjaji.txt',
     'ابن عصفور، شرح جمل الزجاجي',
     'قوله وللجزم علامتان السكون والحذف',
     'تقول لم يقم ولم يقعد', ''),
    ('usfur_diptote_two_causes', 'andalusian/14_IbnUsfur_Sharh_Jumal_al_Zajjaji.txt',
     'ابن عصفور، شرح جمل الزجاجي',
     'والاسم الذي لا ينصرف هو كل اسم اجتمعت فيه علتان فرعيتان',
     'هو ما كان من الجموع علي وزن مفاعل او مفاعيل', ''),
    ('usfur_diptote_jarr_fatha', 'andalusian/14_IbnUsfur_Sharh_Jumal_al_Zajjaji.txt',
     'ابن عصفور، شرح جمل الزجاجي',
     'فالفتحه تكون علامه الخفض في كل اسم وجدت فيه علتان فرعيتان',
     'او عله تقوم مقام علتين', ''),
    ('usfur_mamdud_is_alif_tanith', 'andalusian/13_IbnUsfur_Mumtic_fi_al_Tasrif.txt',
     'ابن عصفور، الممتع في التصريف',
     'الدليل علي ذلك ان الهمزه لا تخلو',
     'او بدلا من الف التانيث', ''),
    # ---------------- Ibn Malik : Alfiyyah + Lamiyyat al-Af'al ----------------
    ('malik_alif_tanith', 'andalusian/03_IbnMalik_Alfiyyah.txt', 'ابن مالك، الألفية',
     'فالف التانيث مطلقا منع صرف الذي حواه كيفما وقع',
     'فالف التانيث مطلقا منع صرف الذي حواه كيفما وقع', ''),
    ('malik_sighat_muntaha', 'andalusian/03_IbnMalik_Alfiyyah.txt', 'ابن مالك، الألفية',
     'وكن لجمع مشبه مفاعلا او المفاعيل بمنع كافلا',
     'وكن لجمع مشبه مفاعلا او المفاعيل بمنع كافلا', ''),
    ('malik_jarr_fatha', 'andalusian/03_IbnMalik_Alfiyyah.txt', 'ابن مالك، الألفية',
     'وجر بالفتحه ما لا ينصرف ما لم يضف او يك بعد ال ردف',
     'وجر بالفتحه ما لا ينصرف ما لم يضف او يك بعد ال ردف', ''),
    ('malik_mutall_jazm', 'andalusian/03_IbnMalik_Alfiyyah.txt', 'ابن مالك، الألفية',
     'فالالف انو فيه غير الجزم وابد نصب ما كيدعو يرمي والرفع فيهما انو واحذف جازما',
     'ثلاثهن تقض حكما لازما', ''),
    ('malik_amr_hamzat_wasl', 'andalusian/06_IbnMalik_Lamiyyat_al_Afal.txt',
     'ابن مالك، لامية الأفعال',
     'من افعل الامر افعل واعزه لسواه كالمضارع ذي الجزم الذي اختزلا',
     'والهمز قبل لزوم الضم ضم', ''),
    # ---------------- al-Shatibi : waw functions (al-Maqasid al-Shafiyya) ----------------
    ('shatibi_waw_atf', 'andalusian/11_Shatibi_Sharh_Alfiyyah.txt',
     'الشاطبي، المقاصد الشافية (شرح الألفية)',
     'لان واو العطف شركت بينهما في العامل',
     'لان واو العطف شركت بينهما في العامل', ''),
    ('shatibi_waw_hal', 'andalusian/11_Shatibi_Sharh_Alfiyyah.txt',
     'الشاطبي، المقاصد الشافية (شرح الألفية)',
     'فقوله ولها كتاب معلوم جمله حاليه مصدره بواو الحال',
     'بواو الحال', ''),
    ('shatibi_waw_istinaf', 'andalusian/11_Shatibi_Sharh_Alfiyyah.txt',
     'الشاطبي، المقاصد الشافية (شرح الألفية)',
     'وقد تقرر ان هذا في الفاء والواو ولمعني فيهما',
     'وإما للاستئناف', ''),
    ('shatibi_waw_qasam', 'andalusian/11_Shatibi_Sharh_Alfiyyah.txt',
     'الشاطبي، المقاصد الشافية (شرح الألفية)',
     'والواو القسميه والعاطفه والفاء العاطفه',
     'والواو القسميه والعاطفه', ''),
    ('shatibi_waw_qasam_vs_atf', 'andalusian/11_Shatibi_Sharh_Alfiyyah.txt',
     'الشاطبي، المقاصد الشافية (شرح الألفية)',
     'كواو القسم لا يجوز ان يجمع بينها وبين الباء',
     'فالواو فيه عاطفه', ''),
    ('shatibi_waw_maiyya', 'andalusian/11_Shatibi_Sharh_Alfiyyah.txt',
     'الشاطبي، المقاصد الشافية (شرح الألفية)',
     'فالوصف الاول تحرز به من الواو التي تكون لمطلق الجمع',
     'فانها لا تعين مفهوم مع', ''),
    ('shatibi_sighat_muntaha', 'andalusian/11_Shatibi_Sharh_Alfiyyah.txt',
     'الشاطبي، المقاصد الشافية (شرح الألفية)',
     'بل هي غايه منتهي الجموع',
     'كونه صيغه منتهي الجموع', ''),
    ('shatibi_maratib', 'andalusian/03_IbnMalik_Alfiyyah.txt', 'ابن مالك، الألفية',
     'وغيره معرفه كهم وذي وهند وابني والغلام والذي',
     'وغيره معرفه كهم وذي وهند وابني والغلام والذي', ''),
    ('malik_falan_wasf', 'andalusian/03_IbnMalik_Alfiyyah.txt', 'ابن مالك، الألفية',
     'وزائدا فعلان في وصف سلم',
     'من ان يري بتاء تانيث ختم', ''),
    ('malik_tanith_wasf', 'andalusian/03_IbnMalik_Alfiyyah.txt', 'ابن مالك، الألفية',
     'ووصف اصلي ووزن افعلا',
     'ممنوع تانيث بتا كاشهلا', ''),
    # ---------------- the parent agent's five leads, verified in the corpus -------------
    ('malik_tashil_an_istinaf', 'andalusian/04_IbnMalik_Tashil_al_Fawaid.txt',
     'ابن مالك، تسهيل الفوائد',
     'وتضمر ان الناصبه ايضا لزوما بعد واو الجمع الواقعه في مواضع الفاء',
     'او قصد الاستئناف بطل اضمار', ''),
    ('malik_tashil_maa_test', 'andalusian/04_IbnMalik_Tashil_al_Fawaid.txt',
     'ابن مالك، تسهيل الفوائد',
     'ويميز واو الجمع تقدير مع موضعها',
     'او حال مكانها', ''),
    ('malik_tashil_maa_fits', 'andalusian/04_IbnMalik_Tashil_al_Fawaid.txt',
     'ابن مالك، تسهيل الفوائد',
     'فاذا لم يلق الفعل بتالي الواو',
     'والا تعين الاضمار', ''),
    ('abuhayyan_maa_fits', 'andalusian/10_AbuHayyan_Tadhyil_al_Tashil.txt',
     'أبو حيان، تذييل التسهيل',
     'فان لم يلق الفعل بتالي الواو جاز النصب علي المعيه',
     'والا تعين الاضمار', ''),
    ('shatibi_maa_test', 'andalusian/11_Shatibi_Sharh_Alfiyyah.txt',
     'الشاطبي، المقاصد الشافية (شرح الألفية)',
     'ولهذا شرط الناظم في الواو ان تكون معينه لمعني مع',
     'لا العاطفه', ''),
    ('ibn_mada_waw_iqtiran', 'andalusian/01_IbnMada_Radd_ala_Nuhat.txt',
     'ابن مضاء، الرد على النحاة',
     'وانما اراد ان قيام الاب اقترن بقعود عمرو',
     'في حكم الجمله الواحده', ''),
    ('ibn_sayyidih_atf_not_qasam', 'andalusian/07_IbnSayyidih_Al_Mukhassas.txt',
     'ابن سيده، المخصص',
     'والواو فيها واو العطف لا واو القسم',
     'واو العطف لا واو القسم', ''),
    ('abuhayyan_istidrak', 'andalusian/09_AbuHayyan_Irtishaf_al_Darab.txt',
     'أبو حيان، ارتشاف الضرب',
     'والاستدراك هو لخبر توهم انه موافق لما قبله في الحكم',
     'ولتوكيد الاول', ''),
    ('mubarrad_waw_implies_an', 'basran/Al_Mubarrad_Al_Muqtadab.txt',
     'المبرد، المقتضب',
     'فمعني الواو الجمع بين الشيئين',
     'كما كان في الفاء', ''),
]


ARABIC_ONLY = re.compile(r'[^\u0621-\u064A]')


def build_map(t):
    """Letter-only stream of the raw text + index map back to raw offsets.

    Matching is done on the Arabic *letter sequence* alone, so a quotation may be
    located even when the edition breaks it with line-continuation tildes, page
    markers (PageV01P032) or manuscript sigla (ms346).  The slice handed back to the
    caller is always taken from the RAW file, so nothing is ever paraphrased."""
    out, idx = [], []
    for i, ch in enumerate(t):
        if DIAC.match(ch):
            continue
        c = ALEF.get(ord(ch), ch)
        if not ('\u0621' <= c <= '\u064A'):
            continue
        out.append(c)
        idx.append(i)
    return ''.join(out), idx


def norm(s):
    return ARABIC_ONLY.sub('', DIAC.sub('', s).translate(ALEF))


def clean(raw):
    """Remove OpenITI/Shamela edition markup only; never touch lexical characters."""
    s = raw
    s = re.sub(r'PageV\d+P\d+', ' ', s)
    s = re.sub(r'\bms\d+\b', ' ', s)
    s = re.sub(r'\bAUTO\b', ' ', s)          # OpenITI structural marker
    s = re.sub(r'\s+([؛،:.!؟])', r'\1', s)   # space before punctuation
    s = s.replace('%~%', ' ')
    s = s.replace('~~', ' ')
    s = s.replace('###', ' ').replace('|||', ' ').replace('||', ' ').replace('###', ' ')
    s = re.sub(r'^\s*[#|]+\s*', '', s)
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = re.sub(r'\s*[#|]+\s*', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def main():
    out = {}
    for key, rel, book, start, end, note in SPANS:
        raw = open(f'{CORPUS}/{rel}', encoding='utf-8').read()
        s, idx = build_map(raw)
        i = s.find(norm(start))
        if i < 0:
            print(f'MISS start {key}: {start[:50]}', file=sys.stderr)
            continue
        j = s.find(norm(end), i)
        if j < 0:
            print(f'MISS end   {key}: {end[:50]}', file=sys.stderr)
            continue
        j += len(norm(end))
        ra = idx[i]
        rb = idx[j - 1] + 1
        raw_span = raw[ra:rb]
        arabic = clean(raw_span)
        line = raw[:ra].count('\n') + 1
        out[key] = {'book': book, 'file': rel, 'line': line,
                    'arabic': arabic, 'note': note,
                    'raw_chars': rb - ra}
        print(f'[{key}] {rel}:{line}  ({len(arabic)} chars clean)')
        print(f'    {arabic[:160]}...' if len(arabic) > 160 else f'    {arabic}')
    json.dump(out, open('/home/grem3/Documents/deepseek-harness/default-workspace/'
                        'scratch_pod/andalusian/citations.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(f'\nwrote {len(out)} citations')


if __name__ == '__main__':
    main()
