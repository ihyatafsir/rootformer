#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""patch_module.py -- generate root_cross_attn_width.py from the SHIPPED, UNTOUCHED
/workspace/root_attn/root_cross_attn.py (md5 b53ae702bde947ac66f933191f904c36).

ONE addition: an optional internal attention width `d_attn` (WIDTH LEVER).

    q_proj : (d_model -> d_attn)      was (d_model -> d_model)
    k_proj : (d_root  -> d_attn)      was (d_root  -> d_model)
    v_proj : (d_root  -> d_attn)      was (d_root  -> d_model)
    o_proj : (d_attn  -> d_model)     was (d_model -> d_model)
    per-head dim = d_attn / num_heads (was d_model / num_heads)

`d_attn = None` reproduces the shipped module EXACTLY (d_attn := d_model), which is what
equiv_check.py proves bit-for-bit before anything is launched.  The shipped file is never
modified; the four live arms (RCA_FROZEN_CTL, RCA_UNFREEZE_A2, ...) keep importing it.
"""
import difflib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE / 'root_cross_attn.py.orig'
DST = HERE / 'root_cross_attn_width.py'

EDITS = [
    # ---- 1. class docstring: document the lever -------------------------------------
    (
        '''    All arithmetic is float32 (the trunk is bf16); the residual is cast back to the trunk
    dtype before the add.  `gate` starts at exactly 0 and `o_proj` at a small random value,
    so the module is EXACTLY a no-op at init while `d(loss)/d(gate) != 0` (the same
    dead-saddle escape that the v18fix `--feat-gate` measurement established: zeroing BOTH
    o_proj and gate makes every branch gradient exactly 0 forever).
    """
''',
        '''    All arithmetic is float32 (the trunk is bf16); the residual is cast back to the trunk
    dtype before the add.  `gate` starts at exactly 0 and `o_proj` at a small random value,
    so the module is EXACTLY a no-op at init while `d(loss)/d(gate) != 0` (the same
    dead-saddle escape that the v18fix `--feat-gate` measurement established: zeroing BOTH
    o_proj and gate makes every branch gradient exactly 0 forever).

    WIDTH LEVER (2026-10-02, `d_attn`)
    ----------------------------------
    `d_attn` sets the INTERNAL width of the cross-attention (the space q/k/v and the attended
    context live in); `d_attn=None` => d_model => byte-identical to the shipped module.
    NOTE the correction this parameter exists to test: the brief called the root channel
    "64 of 896 (7 %)" citing `(896, 64)` projections.  Those tensors are
    IshtiqaqAttentionV12's DEAD root branch (`root_q_proj (896,64)`, `root_k_proj (128,64)`,
    probed: 0 of 24 calls carry root ids).  The LIVE K/V source is
    `morphemic_embed.root_embed.weight` = **(9490, 448)**, i.e. the live root channel is
    448 of 896 = **50 %** of d_model, not 7 %.  Widening it is therefore NOT available by
    narrowing/widening a projection: the source embedding is a frozen 448-dim table, so
    k_proj/v_proj are rank-bounded by 448 and the q-k bilinear form has rank <= 448 at ANY
    d_attn.  `d_attn` raises the attention's *working* width (per-head dim, score/context
    space), which is the only "widen the channel" this architecture admits.  It is a
    CAPACITY test, not an information-rate test.
    """
''',
    ),
    # ---- 2. RootHistoryCrossAttention.__init__ signature + projections ---------------
    (
        '''    def __init__(self, d_model: int, d_root: int, num_heads: int = 8,
                 head_dim: Optional[int] = None, dropout: float = 0.0,
                 out_std: float = 1e-3, proj_std: float = 0.02,
                 layer_idx: int = -1, dtype: torch.dtype = torch.float32,
                 out_norm: bool = False):
        super().__init__()
        head_dim = int(head_dim or (d_model // num_heads))
        if num_heads * head_dim != d_model:
            raise ValueError(f'num_heads*head_dim must equal d_model '
                             f'({num_heads}*{head_dim} != {d_model})')
        self.d_model, self.d_root = int(d_model), int(d_root)
        self.num_heads, self.head_dim, self.layer_idx = int(num_heads), int(head_dim), int(layer_idx)
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.out_norm_enabled = bool(out_norm)

        self.q_proj = nn.Linear(d_model, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_root, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_root, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, d_model, bias=False, dtype=dtype)
''',
        '''    def __init__(self, d_model: int, d_root: int, num_heads: int = 8,
                 head_dim: Optional[int] = None, dropout: float = 0.0,
                 out_std: float = 1e-3, proj_std: float = 0.02,
                 layer_idx: int = -1, dtype: torch.dtype = torch.float32,
                 out_norm: bool = False, d_attn: Optional[int] = None):
        super().__init__()
        # WIDTH LEVER: internal attention width.  None => d_model => shipped behaviour.
        d_attn = int(d_model if d_attn is None else d_attn)
        head_dim = int(head_dim or (d_attn // num_heads))
        if num_heads * head_dim != d_attn:
            raise ValueError(f'num_heads*head_dim must equal d_attn '
                             f'({num_heads}*{head_dim} != {d_attn})')
        self.d_model, self.d_root, self.d_attn = int(d_model), int(d_root), d_attn
        self.num_heads, self.head_dim, self.layer_idx = int(num_heads), int(head_dim), int(layer_idx)
        self.scale = 1.0 / (self.head_dim ** 0.5)
        self.out_norm_enabled = bool(out_norm)

        self.q_proj = nn.Linear(d_model, self.d_attn, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(d_root, self.d_attn, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(d_root, self.d_attn, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.d_attn, d_model, bias=False, dtype=dtype)
''',
    ),
    # ---- 3. RootCrossAttentionStack.__init__ ----------------------------------------
    (
        '''    def __init__(self, d_model: int, root_embed: nn.Embedding, layer_indices: Sequence[int],
                 num_heads: int = 8, head_dim: Optional[int] = None, dropout: float = 0.0,
                 out_std: float = 1e-3, dtype: torch.dtype = torch.float32,
                 out_norm: bool = False):
        super().__init__()
        self.layer_indices = [int(i) for i in layer_indices]
        d_root = int(root_embed.weight.shape[1])
        self.mods = nn.ModuleList([
            RootHistoryCrossAttention(d_model, d_root, num_heads=num_heads, head_dim=head_dim,
                                      dropout=dropout, out_std=out_std, layer_idx=i,
                                      dtype=dtype, out_norm=out_norm)
            for i in self.layer_indices])
''',
        '''    def __init__(self, d_model: int, root_embed: nn.Embedding, layer_indices: Sequence[int],
                 num_heads: int = 8, head_dim: Optional[int] = None, dropout: float = 0.0,
                 out_std: float = 1e-3, dtype: torch.dtype = torch.float32,
                 out_norm: bool = False, d_attn: Optional[int] = None):
        super().__init__()
        self.layer_indices = [int(i) for i in layer_indices]
        d_root = int(root_embed.weight.shape[1])
        self.d_attn = int(d_model if d_attn is None else d_attn)
        self.mods = nn.ModuleList([
            RootHistoryCrossAttention(d_model, d_root, num_heads=num_heads, head_dim=head_dim,
                                      dropout=dropout, out_std=out_std, layer_idx=i,
                                      dtype=dtype, out_norm=out_norm, d_attn=d_attn)
            for i in self.layer_indices])
''',
    ),
    # ---- 4. attach() banner: report the width so the log is self-documenting ---------
    (
        '''        print(f'[*] root cross-attention attached to trunk layers {self.layer_indices} '
              f'({sum(p.numel() for p in self.parameters())/1e6:.2f}M new params, '
              f'd_root={self.d_root}, heads={self.mods[0].num_heads}x{self.mods[0].head_dim}, '
              f'observe={"r_{<=t}" if not self.exclude_current else "r_{<t}"})', flush=True)
''',
        '''        print(f'[*] root cross-attention attached to trunk layers {self.layer_indices} '
              f'({sum(p.numel() for p in self.parameters())/1e6:.2f}M new params, '
              f'd_root={self.d_root}, d_attn={self.d_attn}, '
              f'heads={self.mods[0].num_heads}x{self.mods[0].head_dim}, '
              f'q={tuple(self.mods[0].q_proj.weight.shape)}, '
              f'k={tuple(self.mods[0].k_proj.weight.shape)}, '
              f'o={tuple(self.mods[0].o_proj.weight.shape)}, '
              f'observe={"r_{<=t}" if not self.exclude_current else "r_{<t}"})', flush=True)
''',
    ),
]


def main():
    src = SRC.read_text()
    out = src
    for i, (old, new) in enumerate(EDITS, 1):
        n = out.count(old)
        if n != 1:
            print(f'FATAL: edit {i} matched {n} times, expected 1', file=sys.stderr)
            return 2
        out = out.replace(old, new)
    DST.write_text(out)
    diff = difflib.unified_diff(src.splitlines(True), out.splitlines(True),
                                'root_cross_attn.py', 'root_cross_attn_width.py')
    (HERE / 'root_cross_attn_width.diff').write_text(''.join(diff))
    print(f'wrote {DST.name}: {len(src.splitlines())} -> {len(out.splitlines())} lines, '
          f'{len(EDITS)} edits applied')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
