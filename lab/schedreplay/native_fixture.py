# SPDX-License-Identifier: Apache-2.0
"""Inert fixture adapted from the CPU scheduler test fixture.
See CREDITS.md. No model, worker or device imports.
"""
from enum import Enum
from types import SimpleNamespace as NS

class Request(NS):
    __hash__ = object.__hash__


class Status(Enum):
    WAITING = 1
    RUNNING = 2
    PREEMPTED = 3
    WAITING_FOR_REMOTE_KVS = 4


class Pause(Enum):
    UNPAUSED = 1
    PAUSED_ALL = 2


class Queue(list):
    def peek_request(self):
        return self[0]

    def pop_request(self):
        return self.pop(0)

    def prepend_requests(self, requests):
        self[:0] = requests

    def prepend_request(self, request):
        self.insert(0, request)


def request(name, prompt, computed=0, decode=False, hits=0):
    return Request(request_id=name, num_prompt_tokens=prompt,
        num_computed_tokens=computed, num_tokens=prompt + int(decode),
        num_tokens_with_spec=prompt + (4 if decode else 0),
        num_output_placeholders=0, next_decode_eligible_step=0,
        max_tokens=512, is_prefill_chunk=not decode, has_encoder_inputs=False,
        spec_token_ids=[1, 2, 3] if decode else [], lora_request=None,
        num_stale_output_tokens=0, drop_stale_output=False, prefill_stats=None,
        num_preemptions=0, status=Status.RUNNING if computed else Status.WAITING,
        hits=hits, _all_token_ids=[])


def scheduler(running=(), waiting=(), cap=2048, draft_slots=2):
    blocks = NS(get_block_ids=lambda: ([1],))
    allocations = []

    def allocate(req, tokens, **kwargs):
        recorded = dict(kwargs)
        if 'new_computed_blocks' in recorded:
            recorded['new_computed_blocks'] = recorded['new_computed_blocks'].get_block_ids()
        allocations.append((req.request_id, tokens, recorded))
        return blocks

    s = NS(current_step=0, max_num_scheduled_tokens=cap,
        scheduler_config=NS(max_num_batched_tokens=cap, long_prefill_token_threshold=0,
                            enable_chunked_prefill=True),
        _prefill_capacity=4096,
        vllm_config=NS(speculative_config=NS(max_num_new_slots_for_drafting=draft_slots)),
        _pause_state=Pause.UNPAUSED, max_num_encoder_input_tokens=0,
        prefill_capacity_bound=False, running=list(running), waiting=Queue(waiting),
        skipped_waiting=Queue(), policy='fcfs', connector=None, ec_connector=None,
        lora_config=None, max_model_len=200000, num_sampled_tokens_per_step=1,
        need_mamba_block_aligned_split=False, num_lookahead_tokens=3,
        num_waiting_for_streaming_input=0, max_num_running_reqs=4,
        num_spec_tokens=3, dynamic_sd_lookup=None, is_encoder_decoder=False,
        scheduler_reserve_full_isl=True, log_stats=False,
        _inflight_prefills=set(), kv_cache_config=NS(kv_cache_groups=[1]),
        use_v2_model_runner=True, reset_preempted_req_ids=set(), finished_req_ids=set(),
        defer_block_free=False, sched_step_seq=0, observability_config=None,
        encoder_cache_manager=NS(get_freed_mm_hashes=lambda: [], get_manager_metadata=lambda: None),
        _get_new_block_ids_to_zero=lambda: None, allocations=allocations)
    s.kv_cache_manager = NS(new_step_starts=lambda: None, allocate_slots=allocate,
        get_blocks=lambda _: blocks, record_prefix_cache_stats=lambda *_: None,
        get_num_common_prefix_blocks=lambda _: [0], take_kv_cache_block_copies=lambda: ([], []),
        empty_kv_cache_blocks=blocks)
    s._select_waiting_queue_for_scheduling = lambda: s.waiting
    s._is_blocked_waiting_status = lambda _: False
    s._get_local_prefix_cache_hit = lambda r: (blocks, r.hits, 0, False)
    s._make_cached_request_data = lambda *args: [r.request_id for r in args[0]]

    def update(output):
        for r in s.running:
            r.num_computed_tokens += output.num_scheduled_tokens.get(r.request_id, 0)
            r.is_prefill_chunk = r.num_computed_tokens < r.num_tokens+r.num_output_placeholders
    s._update_after_schedule = update
    return s
