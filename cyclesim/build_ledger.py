#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build derived TP4 shape/weight ledger from an explicit model configuration (no model load)."""
import argparse
import json
from pathlib import Path
from cyclesim.utilities import dump, sha256


def weight(n, k, fmt):
    if fmt == 'bf16':
        return 2 * n * k
    if fmt == 'int4_g128':
        return n * k // 2 + n * (k // 128) * 2
    if fmt == 'int8_g128':
        return n * k + n * (k // 128) * 2
    if fmt == 'int8_channel':
        return n * k + n * 2
    if fmt == 'nvfp4_s16':
        return n * k // 2 + n * (k // 16) + 4
    raise ValueError('unknown weight format')


def build(release, config_path=None, historical=False):
    if config_path is None:
        raise ValueError('explicit model configuration required')
    config = json.loads(config_path.read_text())
    h, tp = config['hidden_size'], config.get('tp', 4)
    m, d = config['moe_intermediate_size'] // tp, config['intermediate_size'] // tp
    attn = [('q_a', config['q_lora_rank'], h), ('kv_a', config['kv_lora_rank'] + config['qk_rope_head_dim'], h),
            ('q_b', config['num_attention_heads'] // tp * config['qk_head_dim'], config['q_lora_rank']),
            ('o_proj', h, config['num_attention_heads'] // tp * config['v_head_dim']),
            ('kv_b', config['num_attention_heads'] // tp * (config['qk_nope_head_dim'] + config['v_head_dim']), config['kv_lora_rank'])]
    shared = [('gate', m, h), ('up', m, h), ('down', h, m)]
    dense = [('gate', d, h), ('up', d, h), ('down', h, d)]
    rows = []
    def add(group, shapes, layers, fmt, decode=True):
        for name, n, k in shapes:
            rows.append({'group': group, 'projection': name, 'n_per_rank': n, 'k_per_rank': k,
                         'layers': layers, 'format': fmt, 'weight_bytes_per_layer_per_rank': weight(n, k, fmt),
                         'decode_read': decode and name != 'kv_b', 'status': 'derived',
                         'basis': 'model dimensions + release nvfp4 format/inventory; ignores reloads and activation traffic'})
    fmt = 'int8_g128' if historical else 'nvfp4_s16'
    add('attention', attn, 77, fmt)
    add('shared', shared, 75, fmt)
    add('dense', dense, 2, fmt)
    add('layer0_attention', attn, 1, 'bf16')
    add('layer0_dense', dense, 1, 'bf16')
    add('experts', shared, 75, 'int4_g128')
    add('mtp_attention', attn, 1, 'int8_channel' if historical else 'nvfp4_s16')
    add('mtp_shared', shared, 1, 'int8_channel' if historical else 'nvfp4_s16')
    add('mtp_experts', shared, 1, 'int8_channel' if historical else 'nvfp4_s16')
    add('mtp_eh', [('eh_proj', h, 2*h)], 1, 'bf16')
    add('router', [('gate', config['n_routed_experts'], h)], 76, 'bf16')
    # Router source is float32; keep the exception explicit.
    rows[-1]['format'] = 'fp32'
    rows[-1]['weight_bytes_per_layer_per_rank'] *= 2
    add('indexer', [('wq_b', 4096, 2048), ('wk', 128, h), ('weights_proj', 32, h)], 21, 'bf16')
    add('absorbed_ukuv', [('UK', 4096, 512), ('UV', 512, 3072)], 78, 'bf16')
    add('target_head', [('head', config['vocab_size'] // tp, h)], 1, 'bf16')
    add('draft_head', [('head', config['vocab_size'] // tp, h)], 1, 'bf16')
    add('draft_head_DH', [('head', config['vocab_size'] // tp, h)], 1, 'nvfp4_s16')
    def total(groups):
        return sum(r['weight_bytes_per_layer_per_rank'] * r['layers'] for r in rows if r['group'] in groups and r['decode_read'])
    budgets = {}
    for c, expert_union in ((1, 17.6), (2, 42.5), (4, 59.0)):
        expert_bytes = sum(weight(n, k, 'int4_g128') for _, n, k in shared) * 75 * expert_union
        budgets[str(c)] = {'target': {'marlin_moe': expert_bytes,
            ('int8_gemm' if historical else 'nvfp4_gemm'): total(('attention', 'shared', 'dense'))},
            'draft': {('int8_gemm' if historical else 'nvfp4_gemm'): total(('mtp_attention', 'mtp_shared')),
                      'marlin_moe': total(('mtp_experts',)) * min(256, c * 8),
                      'bf16_gemm':total(('mtp_eh',))}}
    sources = [{'path': config_path.name, 'sha256': sha256(config_path)}]
    return {'schema': 1, 'stack': 'historical-int8-k2' if historical else 'calibrated-nvfp4-k3',
            'tp': tp, 'rows': rows, 'sources': sources, 'vocab_head_bytes': config['vocab_size'] // tp * h * 2,
            'head_bytes': {'target': {'bf16': weight(config['vocab_size']//tp, h, 'bf16')},
                           'draft': {'bf16': weight(config['vocab_size']//tp, h, 'bf16'),
                                     'nvfp4': weight(config['vocab_size']//tp, h, 'nvfp4_s16')}},
            'draft_expert_union': {'1': 8, '2': 16, '4': 32, 'status': 'assumed; topk_ids required'},
            'kv': {'mla_bytes_per_token_layer': 576 if historical else 368, 'indexer_bytes_per_token_layer': 132,
                   'mla_layers': 79, 'indexer_layers': 22, 'status': 'derived',
                   'read_tokens_per_cycle': 'unknown: capture context and selected MLA pages required'},
            'expert_union': {'1': 17.6, '2': 42.5, '4': 59.0, 'status': 'assumed',
                             'basis': 'historical routing union prior; topk_ids capture needed'},
            'cycle_budgets': budgets,
            'limits': ['Class traffic apportioned across kernel names by calls, not shape-resolved CUPTI.',
                       'Working-set flags are assumptions, never verified L2 residency.',
                       'No estimated GB/s for unbound attention/KV, dense/indexer or elementwise kernels.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--historical', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    dump(a.out, build(None, a.config, a.historical))


if __name__ == '__main__':
    main()
