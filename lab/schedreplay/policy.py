# SPDX-License-Identifier: Apache-2.0
"""Existing independent fast policy, metadata subset only."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS

from lab.schedreplay.native_fixture import Pause, Queue, Status, request, scheduler

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Policy:
    chunk: int = 4096
    decode_steps: int = 40
    off: bool = False
    variant: str = 'none'
    reserve_tokens: int = 512
    prefill_cap: int = 1024

    @classmethod
    def parse(cls, value, **kwargs):
        if value.lower() == 'off':
            return cls(off=True, **kwargs)
        try:
            kind, chunk, steps = value.split(':')
            chunk, steps = int(chunk), int(steps)
        except (ValueError, TypeError):
            raise ValueError('policy must be OFF or fair:CHUNK:DECODE_STEPS')
        if kind != 'fair' or chunk not in (512, 1024, 2048, 4096) or steps < 0:
            raise ValueError('invalid fair policy')
        return cls(chunk, steps, **kwargs)

    def __post_init__(self):
        if self.chunk not in (512,1024,2048,4096) or type(self.decode_steps) is not int or self.decode_steps<0:
            raise ValueError('invalid fair chunk/decode_steps')
        if self.variant not in ('none', 'reserve', 'shortest', 'cap'):
            raise ValueError('invalid variant')
        if not 1 <= self.reserve_tokens <= 2048 or not 1 <= self.prefill_cap <= 2048:
            raise ValueError('reserve/prefill cap must be in [1,2048]')

    @property
    def name(self):
        base = 'OFF' if self.off else 'fair:%d:%d' % (self.chunk, self.decode_steps)
        suffix = '' if self.variant == 'none' else '+' + self.variant
        if self.variant == 'reserve': suffix += ':' + str(self.reserve_tokens)
        if self.variant == 'cap': suffix += ':' + str(self.prefill_cap)
        return base + suffix


def make_scheduler(max_seqs=4, draft_slots=0):
    s = scheduler(draft_slots=draft_slots)
    s.max_model_len = 262144
    s.max_num_running_reqs = max_seqs
    # No growing allocation log in sweeps. Inert KV boundary, explicit below.
    blocks = NS(get_block_ids=lambda: ([1],))
    s.kv_cache_manager.allocate_slots = lambda *a, **kw: blocks
    return s


def prepare(s, policy):
    if policy.variant == 'shortest':
        s.waiting.sort(key=lambda r: r.num_prompt_tokens - max(r.hits, r.num_computed_tokens))


def need(r, s):
    if s.current_step < r.next_decode_eligible_step:
        return 0
    if (r.num_output_placeholders > 0 and r.num_computed_tokens + 2 - r.num_output_placeholders
            >= r.num_prompt_tokens + r.max_tokens):
        return 0
    n = r.num_tokens_with_spec + r.num_output_placeholders - r.num_computed_tokens
    threshold = s.scheduler_config.long_prefill_token_threshold
    if threshold > 0: n = min(n, threshold)
    return max(0, min(n, s.max_model_len - r.num_computed_tokens - s.num_sampled_tokens_per_step))


def reserve_waiting(s, policy):
    if policy.variant != 'reserve' or len(s.running) >= s.max_num_running_reqs:
        return 0
    if not s.waiting: return 0
    r = s.waiting[0]
    remaining = r.num_prompt_tokens - max(r.num_computed_tokens, r.hits)
    # Strict waiting order: reserve only when head is short. No FIFO bypass.
    return remaining if 0 < remaining <= policy.reserve_tokens else 0


class Fast:
    """Subset: FCFS text-only, local APC, inert successful KV, no preemption/LoRA/MM.

    Policy variants are existing offline hypotheses; native runtime adapters
    are not bundled in this portable package.
    """
    def __init__(self, policy):
        self.policy = policy

    def schedule(self, s):
        p = self.policy
        prepare(s, p)
        s.current_step += 1
        serving = min(s.max_num_scheduled_tokens, 2048) == 2048
        limit = min(s.max_num_scheduled_tokens, 2048)
        input_limit = min(s.scheduler_config.max_num_batched_tokens, 2048)
        serving = serving and input_limit == 2048
        decodes = {r.request_id: need(r, s) for r in s.running
                   if r.num_computed_tokens >= r.num_prompt_tokens and need(r, s) > 0}
        prefills = {r.request_id for r in s.running if r.num_computed_tokens < r.num_prompt_tokens}
        has_prefill = bool(prefills) or any(r.num_computed_tokens < r.num_prompt_tokens
                                           for r in s.waiting + s.skipped_waiting)
        active = not p.off and serving and bool(decodes) and has_prefill
        if active: limit = input_limit = 4096
        token_budget, input_budget = limit, input_limit
        if s._pause_state == Pause.PAUSED_ALL: token_budget = 0
        slots = s.vllm_config.speculative_config.max_num_new_slots_for_drafting
        emitted = {}
        before_prefill = {r.request_id:r.is_prefill_chunk for r in s.running}
        since = getattr(s, '_glm_df_since_prefill', p.decode_steps)
        reserve = reserve_waiting(s, p)

        def cap(r, computed):
            nonlocal limit, input_limit, token_budget, input_budget
            prefill = computed < r.num_prompt_tokens
            if active:
                if not prefill: value = 4096
                elif since < p.decode_steps: value = 0
                else:
                    prefills.add(r.request_id)
                    used = sum(n for rid, n in emitted.items() if rid in prefills)
                    pending = [decodes[x.request_id] for x in s.running
                               if x.request_id in decodes and x.request_id not in emitted]
                    value = max(0, min(p.chunk - used, token_budget - sum(pending),
                                      input_budget - sum(pending) - (len(pending)+1)*slots))
            else:
                value = 2048 if prefill else 4096
                if serving and prefill and r.num_prompt_tokens >= 16384:
                    token_budget += 4096-limit
                    input_budget += 4096-input_limit
                    limit = input_limit = 4096
                    value = 4096
            if prefill and r in s.running and s.waiting:
                if p.variant == 'reserve':
                    value = min(value, max(0, token_budget-reserve))
                    if active:
                        used = sum(n for rid,n in emitted.items() if rid in prefills)
                        value = min(value,max(0,p.chunk-used-reserve))
                if p.variant == 'cap': value = min(value, p.prefill_cap)
            return value

        for r in list(s.running):
            if token_budget <= 0 or input_budget <= slots: break
            n = need(r, s)
            if not n: continue
            n = min(n, cap(r, r.num_computed_tokens), token_budget, input_budget-slots)
            if n <= 0: continue
            if s.kv_cache_manager.allocate_slots(r, n) is None:
                raise ValueError('running KV preemption outside fast replay scope')
            emitted[r.request_id] = n
            token_budget -= n
            input_budget -= n+slots
            r.spec_token_ids = []
        while s.waiting and token_budget > 0 and input_budget > slots:
            if len(s.running) >= s.max_num_running_reqs: break
            r = s.waiting[0]
            computed = r.num_computed_tokens if r.num_computed_tokens else r.hits
            value = cap(r, computed)
            if value == 0: break
            n = r.num_tokens - computed
            # Native uniform speculative newcomer padding occurs before the
            # threshold clamp and refuses an unpadded budget/context tail.
            prefill_scheduled = any(before_prefill.get(rid, False) for rid in emitted)
            if (s.num_spec_tokens > 0 and s.dynamic_sd_lookup is None
                    and s.num_sampled_tokens_per_step > 0 and n == 1
                    and any(rid in before_prefill for rid in emitted) and not prefill_scheduled):
                n = 1+s.num_spec_tokens
                if n > min(value,token_budget,input_budget-slots) or computed+n > s.max_model_len:
                    break
            threshold = s.scheduler_config.long_prefill_token_threshold
            if threshold > 0: n = min(n, threshold)
            n = min(n, value, token_budget, input_budget-slots)
            if n <= 0: raise ValueError('fully cached prompt requires last-token recomputation')
            if s.kv_cache_manager.allocate_slots(r, n, full_sequence_must_fit=s.scheduler_reserve_full_isl) is None:
                break
            s.waiting.pop(0)
            s.running.append(r)
            r.status = Status.RUNNING
            r.num_computed_tokens = computed
            emitted[r.request_id] = n
            token_budget -= n
            input_budget -= n+slots
        if not p.off:
            if not active: s._glm_df_since_prefill = p.decode_steps
            elif any(emitted.get(rid, 0) > 0 for rid in prefills): s._glm_df_since_prefill = 0
            elif any(emitted.get(rid, 0) > 0 for rid in decodes):
                s._glm_df_since_prefill = min(p.decode_steps, since+1)
        s._update_after_schedule(NS(num_scheduled_tokens=emitted))
        return emitted

