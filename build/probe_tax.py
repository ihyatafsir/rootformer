import os, sys, json
os.environ.setdefault('ROOTFORMER_VALIDATED_SEG','1')
RELEASE='/workspace/hf_v19_2_release'
for p in (RELEASE, RELEASE+'/models'):
    if p not in sys.path: sys.path.insert(0,p)
import nrmp_vocab as nv
CLS=next(c for k,c in vars(nv).items() if isinstance(c,type) and 'MorphemicVocab' in k)
V=CLS(RELEASE+'/data/rootformer_v12_arabic_blueprint.json')
v=V
words=['فى','تعالى','صلى','وسلم','وأما','الأول','الشيء','عليه','إليه','مما','عما','لكان','تبين','له','به','ان','إلا','و','ب','ا','أيضا','لهم','إنما','كذلك','يكن','تقدم','الله','والله','لله','بالله','أبي','أهل','الأرض','العدد','عدد','متناهية','أبدا','يجب','بذاته','الاشياء','مما','فيما','بما','لما','مالم','ولديه','لديه','أولو','أولى','الذين','الذي','هؤلاء','هذا']
for w in words:
    p,r,wz,s=v.encode_word(w)
    dec=v.decode_word(p,r,wz,s)
    print('%-10s -> %-14s  P=%-6s R=%-8s W=%-14s S=%s'%(w,dec,v.id2prefix.get(p),v.id2root.get(r),v.id2wazn.get(wz),v.id2suffix.get(s)))
print()
print('COMMON_PARTICLES n=',len(v.COMMON_PARTICLES))
bt=v.base_tok
print('has CLOSED_PARTICLES', hasattr(bt,'CLOSED_PARTICLES'), len(getattr(bt,'CLOSED_PARTICLES',[])))
print('token_to_id size', len(bt.token_to_id), 'roots_set', len(bt.roots_set))
import inspect
src=inspect.getsource(bt.realize_root_and_wazn)
print('realize src lines', src.count(chr(10)))
import ast
tree=ast.parse(src)
impl=set()
for node in ast.walk(tree):
    if isinstance(node,ast.Compare) and isinstance(node.left,ast.Name) and node.left.id=='wazn':
        for op,comp in zip(node.ops,node.comparators):
            if isinstance(op,ast.Eq) and isinstance(comp,ast.Constant) and isinstance(comp.value,str): impl.add(comp.value)
            if isinstance(op,ast.In) and isinstance(comp,(ast.List,ast.Tuple)):
                for e in comp.elts:
                    if isinstance(e,ast.Constant) and isinstance(e.value,str): impl.add(e.value)
print('implemented wazn count', len(impl))
print(sorted(impl))
allw=set(bt.awzan_set)
print('awzan_set', len(allw), 'unimplemented', len(allw-impl))
print(sorted(allw-impl))
