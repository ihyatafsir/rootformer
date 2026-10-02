#!/usr/bin/env python3
"""make_ewc_trainer.py -- produce nrmt_train_ewc.py from the UNTOUCHED nrmt_train.py.

Rationale: the training arms currently running use nrmt_train.py.  Editing it in place would
change the artefact another agent's run depends on.  So the original is left byte-identical
(md5 recorded and re-checked) and the EWC variant is generated here by exact-anchor insertion,
which also yields a small, reviewable diff and a mechanically checkable inertness proof.

The patch adds exactly four things:
  1. an asset-root fallback, because the generated file lives OUTSIDE the release tree
     (/workspace/ghazali_forget) while its data/blueprint live inside it;
  2. four CLI flags, all inert by default (--ewc-lambda defaults to 0.0);
  3. an _EWCPenalty object built from the pretrained anchor + a diagonal retained-Fisher file,
     constructed only when --ewc-lambda != 0;
  4. two lines in the training loop that add the penalty to the loss.

Nothing else is changed.  PATCHES is the single source of truth: patch() applies it forward and
unpatch() applies it backward, so tests/test_ewc_unit.py T7 can prove that the generated file
minus the inserted blocks is byte-identical to the original -- which is a strictly stronger
statement than any finite smoke run.
"""
import hashlib
import sys

SRC = '/workspace/hf_v19_2_release/nrmt_train.py'
DST = '/workspace/ghazali_forget/nrmt_train_ewc.py'
RELEASE = '/workspace/hf_v19_2_release'

ROOT_ANCHOR = """ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'models'))
"""
ROOT_PATCH = """ROOT_DIR = Path(__file__).resolve().parent
# [ewc patch] this generated copy lives outside the release tree; resolve its assets against the
# release directory unless it happens to be sitting beside them (then the original logic holds).
_REL_ROOT = ROOT_DIR if (ROOT_DIR / 'nrmp_vocab.py').exists() else Path('%s')
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(_REL_ROOT))
sys.path.insert(0, str(_REL_ROOT / 'models'))
ROOT_DIR = _REL_ROOT
""" % RELEASE

ARGS_ANCHOR = "    ap.add_argument('--save', default='/workspace/nrmt_head.pt')\n"
ARGS_PATCH = """    ap.add_argument('--save', default='/workspace/nrmt_head.pt')
    # ---- [ewc patch] al-tarjih: retained-structure (EWC-form) penalty ----------------------
    # ALL default to OFF.  --ewc-lambda 0.0 leaves the historical code path byte-for-byte the
    # one that produced A2 19.53 % / FIX 6.62 % / Run A's 0.1x collapse (proved by smoke equiv).
    ap.add_argument('--ewc-lambda', type=float, default=0.0,
                    help='weight of the retained-structure penalty '
                         'lambda * sum_i F_i (theta_i - theta0_i)^2.  0.0 = OFF (historical).')
    ap.add_argument('--ewc-fisher', default='',
                    help='.pt file: {trunk_param_name: diagonal Fisher}.  Required when '
                         '--ewc-lambda != 0.  Measured on the RETAINED distribution -- see '
                         'ewc_fisher.py / ewc_fisher_head.py.')
    ap.add_argument('--ewc-scope', default='',
                    help='comma-separated trunk layer indices to penalise.  Default = the '
                         'unfrozen set (--unfreeze-last).')
    ap.add_argument('--ewc-reduction', choices=('sum', 'mean-per-tensor'), default='sum',
                    help='how the per-tensor penalties are combined.  `sum` is the EWC form '
                         'lambda * sum_i F_i d_i^2 as specified; `mean-per-tensor` is a diagnostic.')
"""

CLASS_ANCHOR = "\ndef main():\n"
CLASS_PATCH = '''

class _EWCPenalty:
    """lambda * sum_i F_i (theta_i - theta0_i)^2, with theta0 the PRETRAINED trunk.

    Every quantity is a measured tensor: F_i comes from ewc_fisher.py / ewc_fisher_head.py and
    theta0 is the value loaded from --checkpoint.  The penalty is exactly 0.0 at construction
    (theta == theta0) and its gradient is exactly 2*lambda*F_i*(theta_i - theta0_i).
    """

    def __init__(self, named_params, fisher, reduction='sum'):
        self.reduction = reduction
        self.names, self.params, self.F, self.theta0 = [], [], [], []
        missing = []
        for n, p in named_params:
            if n not in fisher:
                missing.append(n)
                continue
            self.names.append(n)
            self.params.append(p)
            self.F.append(fisher[n].to(device=p.device, dtype=torch.float32).contiguous())
            # theta0 = the PRETRAINED weights: cloned here, before a single optimiser step.
            self.theta0.append(p.detach().clone())
        if missing:
            raise SystemExit(
                f'--ewc-fisher is missing {len(missing)} penalised tensors, e.g. {missing[:3]}; '
                f'it must cover the whole --ewc-scope')
        if not self.params:
            raise SystemExit('--ewc-fisher covered none of the penalised tensors')
        self.n_params = sum(p.numel() for p in self.params)

    @torch.no_grad()
    def value(self):
        tot = 0.0
        for p, f, t0 in zip(self.params, self.F, self.theta0):
            d = p.detach().float() - t0.float()
            tot += float((f * d * d).sum())
        return tot

    def penalty(self):
        tot = None
        for p, f, t0 in zip(self.params, self.F, self.theta0):
            d = p.float() - t0.float()
            term = (f * d * d).sum()
            tot = term if tot is None else tot + term
        if self.reduction == 'mean-per-tensor':
            tot = tot / len(self.params)
        return tot

    def stats(self):
        f = torch.cat([x.reshape(-1) for x in self.F])
        return {'n_tensors': len(self.params), 'n_params': self.n_params,
                'fisher_mean': float(f.mean()), 'fisher_median': float(f.median()),
                'fisher_max': float(f.max()), 'fisher_min': float(f.min()),
                'fisher_zeros': int((f == 0).sum()), 'penalty_at_init': self.value()}
'''

SETUP_ANCHOR = "    if args.lambda_orbit != 0.0:\n"
SETUP_PATCH = """    # ---- [ewc patch] al-tarjih: build the penalty (inert unless --ewc-lambda != 0) ----------
    ewc = None
    if args.ewc_lambda != 0.0:
        if not args.ewc_fisher:
            raise SystemExit('--ewc-lambda != 0 requires --ewc-fisher (see ewc_fisher.py)')
        if not trunk_params:
            raise SystemExit('--ewc-lambda != 0 but no trunk parameters are trainable '
                             '(pass --unfreeze-last N)')
        _scope = ([int(x) for x in args.ewc_scope.split(',') if x.strip()] if args.ewc_scope
                  else unfrozen_layers)
        _scope_t = tuple(f'backbone.layers.{i}.' for i in _scope)
        _fisher_raw = torch.load(args.ewc_fisher, map_location='cpu')
        _named = [(n, p) for n, p in model.named_parameters()
                  if p.requires_grad and n.startswith(_scope_t)]
        ewc = _EWCPenalty(_named, _fisher_raw, args.ewc_reduction)
        _st = ewc.stats()
        print(f'[*] EWC [al-tarjih]: lambda={args.ewc_lambda:g} scope={_scope} '
              f'fisher={Path(args.ewc_fisher).name} reduction={args.ewc_reduction}', flush=True)
        print(f'[*] EWC stats: {_st}', flush=True)
        print(f'[*] EWC self-check: penalty(theta0) == 0 -> {_st["penalty_at_init"] == 0.0} '
              f'(value {_st["penalty_at_init"]!r})', flush=True)
        if _st['penalty_at_init'] != 0.0:
            raise SystemExit('EWC penalty is not zero at the pretrained anchor')

    if args.lambda_orbit != 0.0:
"""

LOSS_ANCHOR = """        if not torch.isfinite(loss):
            nan_steps += 1
"""
LOSS_PATCH = """        if ewc is not None:
            # [ewc patch] AL-TARJIH, applied: the retained structure's second-order restoring
            # term.  Added BEFORE the isfinite check so a non-finite penalty is caught.
            _pen = ewc.penalty()
            loss = loss + args.ewc_lambda * _pen
            info['ewc_pen'] = float(_pen.detach())
            info['ewc_pen_weighted'] = float(args.ewc_lambda * _pen.detach())
        if not torch.isfinite(loss):
            nan_steps += 1
"""

PATCHES = [(ROOT_ANCHOR, ROOT_PATCH, 'root'),
           (ARGS_ANCHOR, ARGS_PATCH, 'args'),
           (CLASS_ANCHOR, CLASS_PATCH + '\ndef main():\n', 'class'),
           (SETUP_ANCHOR, SETUP_PATCH, 'setup'),
           (LOSS_ANCHOR, LOSS_PATCH, 'loss')]


def patch(src):
    out = src
    for anchor, repl, what in PATCHES:
        n = out.count(anchor)
        if n != 1:
            raise SystemExit(f'anchor {what!r} matched {n} times (expected 1) -- refusing to patch')
        out = out.replace(anchor, repl)
    return out


def unpatch(dst):
    out = dst
    for anchor, repl, what in PATCHES:
        if out.count(repl) != 1:
            raise SystemExit(f'inserted block {what!r} not found exactly once -- cannot un-patch')
        out = out.replace(repl, anchor)
    return out


def main():
    src = open(SRC, encoding='utf-8').read()
    dst = patch(src)
    open(DST, 'w', encoding='utf-8').write(dst)
    print(f'original {SRC}\n  md5 {hashlib.md5(src.encode()).hexdigest()}  lines {src.count(chr(10))}')
    print(f'patched  {DST}\n  md5 {hashlib.md5(dst.encode()).hexdigest()}  lines {dst.count(chr(10))}')
    print(f'inserted {dst.count(chr(10)) - src.count(chr(10))} lines')
    print(f'un-patch round trip identical: {unpatch(dst) == src}')


if __name__ == '__main__':
    main()
