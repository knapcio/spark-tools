# SPDX-License-Identifier: Apache-2.0
"""Two-deep schedule-ahead queue with an explicitly assumed serial device clock."""
from collections import deque
from dataclasses import dataclass, field
import heapq
import math

from lab.schedreplay.policy import Fast, make_scheduler, request
from lab.schedreplay.timing import Timing

POOL = 4135*64
BLOCK = 64


def quantile(values, q):
    if not values: return None
    values = sorted(values)
    i = (len(values)-1)*q
    a = int(i); b = min(a+1,len(values)-1)
    return values[a]+(values[b]-values[a])*(i-a)


def percentiles(values):
    return {name:quantile(values,q) for name,q in [('p50',.5),('p90',.9),('p99',.99)]}


def bucket(prompt):
    if prompt <= 8192: return '0-8K'
    if prompt <= 32768: return '8K-32K'
    if prompt <= 131072: return '32K-128K'
    return '128K-256K'


def demand(work):
    # Shared prefix caching is disabled: APC hits do NOT reduce admission reserve.
    return ((work.prompt_tokens+BLOCK-1)//BLOCK+(work.max_tokens+BLOCK-1)//BLOCK)*BLOCK


@dataclass
class State:
    work: object
    arrival: float = None
    admitted: float = None
    submitted: float = None
    started: float = None
    first: float = None
    end: float = None
    status: str = 'pending'
    tokens: int = 0
    credit: float = 0
    events: list = field(default_factory=list)
    row: object = None
    reservation: int = 0
    hold_until: float = None
    prefill_tokens: int = 0


class Replay:
    def __init__(self, policy, mode='fast', max_seqs=4, admission=False, timing=None,
                 width=4, engine_floor_tokens=0, queue_depth=2, max_steps=2000000,
                 max_time_s=86400, trace=None):
        if mode != 'fast': raise ValueError('portable replay supports fast mode only')
        if max_seqs not in (4,6): raise ValueError('max-num-seqs must be 4 or 6')
        if type(width) is not int or width != 4: raise ValueError('only configured K3 (requested width 4) supported')
        if queue_depth not in (1,2): raise ValueError('queue depth must be 1 or 2')
        if not 0 <= engine_floor_tokens <= POOL: raise ValueError('engine KV floor outside pool')
        if max_steps <= 0 or not math.isfinite(max_time_s) or max_time_s <= 0: raise ValueError('invalid replay bound')
        self.policy, self.mode = policy, mode
        self.s = make_scheduler(max_seqs)
        self.stepper = Fast(policy)
        self.timing = timing or Timing()
        self.max_seqs, self.admission, self.width = max_seqs, admission, width
        self.floor, self.depth, self.max_steps, self.max_time = engine_floor_tokens, queue_depth, max_steps, max_time_s
        self.trace = trace
        self.queue = deque()
        self.ledger, self.admission_queue = {}, deque()
        self.steps = 0
        self.prefill_work_s, self.prefill_count, self.mixed_steps = 0,0,0
        self.peak_queue = 0
        self.peak_reserved = 0
        self.engine_capacity_refusals = 0
        self.now = 0

    def engine_used(self):
        # Synthetic current KV usage (APC memory charged), not allocator proof.
        live = {r.request_id for r in self.s.running+self.s.waiting}
        outstanding = {item[0] for job in self.queue for item in job['rows']}
        total = 0
        for rid in live|outstanding:
            state = self.states[rid]
            if state.row is not None:
                computed = min(state.work.prompt_tokens+state.work.max_tokens,
                               max(state.row.num_computed_tokens,state.work.cached_prefix_tokens))
                total += ((computed+BLOCK-1)//BLOCK)*BLOCK
        return max(total,self.floor)

    def ledger_used(self):
        return sum(s.reservation for s in self.ledger.values())

    def finish(self, state, status, at):
        if state.end is not None: return
        state.status, state.end = status,at
        rid = state.work.request_id
        self.s.running[:] = [r for r in self.s.running if r.request_id != rid]
        self.s.waiting[:] = [r for r in self.s.waiting if r.request_id != rid]
        self.admission_queue = deque(x for x in self.admission_queue if x.work.request_id != rid)
        if rid in self.ledger:
            if status == 'cancelled' and state.submitted is not None:
                # Engine-idle release or 15 s degraded hold. Clock starts
                # when cancellation is visible to this simulated coordinator.
                state.hold_until = self.now+15
            else: self.ledger.pop(rid,None)
        for child in self.children.get(rid,[]):
            arrival = max(child.work.arrival_s,at+child.work.think_time_s)
            heapq.heappush(self.pending,(arrival,self.order[child.work.request_id],child.work.request_id))

    def admit(self, state):
        state.admitted, state.submitted, state.status = self.now,self.now,'engine-waiting'
        state.row = request(state.work.request_id,state.work.prompt_tokens,
                            hits=min(state.work.cached_prefix_tokens,state.work.prompt_tokens-1))
        state.row.max_tokens = state.work.max_tokens
        self.s.waiting.append(state.row)
        if self.admission:
            state.reservation = demand(state.work)
            self.ledger[state.work.request_id] = state
            self.peak_reserved = max(self.peak_reserved,self.ledger_used())

    def drain_inputs(self):
        while self.pending and self.pending[0][0] <= self.now+1e-12:
            arrival,_,rid = heapq.heappop(self.pending)
            state = self.states[rid]
            state.arrival = arrival
            if state.work.cancel_at_s is not None and state.work.cancel_at_s <= self.now:
                self.finish(state,'cancelled',max(arrival,state.work.cancel_at_s));continue
            state.status = 'admission-waiting'
            if self.admission:
                if len(self.admission_queue)>=64: self.finish(state,'queue-full',self.now)
                else: self.admission_queue.append(state)
            else: self.admit(state)
        for state in self.states.values():
            if (state.arrival is not None and state.end is None and state.work.cancel_at_s is not None
                    and state.work.cancel_at_s<=self.now):
                self.finish(state,'cancelled',max(state.arrival,state.work.cancel_at_s))
        for rid,state in list(self.ledger.items()):
            if state.hold_until is not None:
                engine_idle = not self.s.running and not self.s.waiting and not self.queue and self.floor==0
                if engine_idle or self.now>=state.hold_until: self.ledger.pop(rid)
        while self.admission_queue:
            state = self.admission_queue[0]
            if self.now-state.arrival >= 540:
                self.finish(state,'max-wait',self.now);continue
            charge = demand(state.work)
            if self.ledger_used()+charge>POOL or self.engine_used()+charge>POOL: break
            self.admission_queue.popleft()
            self.admit(state)

    def schedule(self):
        # Text-only full-ISL check at waiting allocation. No fabricated KV
        # preemptions: unsupported pressure is reported, never interpreted as
        # a prediction of native victim selection.
        blocks = self.s.kv_cache_manager.empty_kv_cache_blocks
        def allocate(r,n,**kw):
            if kw.get('full_sequence_must_fit'):
                running_usage = sum(x.num_prompt_tokens+x.max_tokens for x in self.s.running)
                candidate = r.num_prompt_tokens+r.max_tokens
                if running_usage+candidate > POOL or self.floor+candidate > POOL:
                    self.engine_capacity_refusals += 1
                    return None
            return blocks
        self.s.kv_cache_manager.allocate_slots = allocate
        # Fast path shares only the successful/refused KV boundary, not policy
        # budgeting. See fast/exact equivalence corpus.
        eligible = sum(r.num_computed_tokens >= r.num_prompt_tokens and
                       self.s.current_step+1 >= r.next_decode_eligible_step and
                       not (r.num_output_placeholders > 0 and
                            r.num_computed_tokens+2-r.num_output_placeholders >= r.num_prompt_tokens+r.max_tokens)
                       for r in self.s.running)
        execution_width = self.timing.execution_width(eligible)
        for r in self.s.running:
            if r.num_computed_tokens >= r.num_prompt_tokens:
                # Expected-value output/rollback boundary. Prefill schedule-ahead
                # remains native; decode positions follow emitted outputs.
                state = self.states[r.request_id]
                # Outstanding placeholders advance the scheduling cursor, while
                # acknowledged output sets the non-speculative token frontier.
                r.num_computed_tokens = r.num_prompt_tokens+state.tokens-1+r.num_output_placeholders
                r.num_tokens = r.num_prompt_tokens+state.tokens
                r.num_tokens_with_spec = r.num_tokens+execution_width-1
                r.spec_token_ids = list(range(execution_width-1))
                r.is_prefill_chunk = False
        before = {r.request_id:(r.num_computed_tokens if r in self.s.running else max(r.hits,r.num_computed_tokens))
                  for r in self.s.running+self.s.waiting}
        emitted = self.stepper.schedule(self.s)
        self.steps += 1
        if self.steps>self.max_steps: raise ValueError('replay step bound exceeded; no partial result claimed')
        prefills,decoders,items = [],[],[]
        for rid,n in emitted.items():
            state = self.states[rid];work = state.work
            computed = before[rid]
            if state.started is None: state.started = self.now
            state.status = 'running'
            if computed < work.prompt_tokens:
                end = min(computed+n,work.prompt_tokens)
                prefills.append((computed,end))
                items.append((rid,'prefill',end-computed,end>=work.prompt_tokens,n))
            else:
                decoders.append((work.kind,work.prompt_tokens+state.tokens))
                items.append((rid,'decode',0,True,n))
        # Account for each in-flight output batch, including prompt tails.
        # Worker rollback/acceptance is still the expected-value boundary;
        # placeholders are retired in complete(), once per queued batch.
        placeholders = {}
        for rid,phase,prefill,outputs,n in items:
            if outputs:
                count = 1+max(0,n-prefill) if phase == 'prefill' else n
                self.states[rid].row.num_output_placeholders += count
                placeholders[rid] = count
        if any(self.states[rid].work.cached_prefix_tokens for rid,phase,_,_,_ in items if phase == 'prefill'):
            self.timing.unsupported.add('APC marginal timing has no retained-prefix calibration')
        regimes = [self.policy.chunk if decoders and not self.policy.off else
                   (4096 if self.states[rid].work.prompt_tokens >= 16384 else 2048)
                   for rid,phase,_,_,_ in items if phase == 'prefill']
        duration,evidence = self.timing.cost(prefills,decoders,self.width,regimes=regimes)
        if not emitted: return False
        start = max(self.now,self.queue[-1]['end'] if self.queue else self.now)
        job = {'step':self.steps,'scheduled_at_s':self.now,'start':start,'end':start+duration,
               'rows':items,'scheduled_tokens':emitted,'cost_evidence':evidence,'output_placeholders':placeholders,
               'decode_concurrency':len(decoders),'effective_decode_width':self.timing.execution_width(len(decoders))}
        self.queue.append(job)
        self.peak_queue = max(self.peak_queue,len(self.queue))
        if prefills:
            self.prefill_work_s += duration
            self.prefill_count += sum(end-start for start,end in prefills)
        if prefills and decoders: self.mixed_steps += 1
        if self.trace:
            import json
            self.trace.write(json.dumps(job,sort_keys=True)+'\n')
        return True

    def next_front_event(self):
        events = []
        if self.pending: events.append(self.pending[0][0])
        if self.admission_queue: events.append(self.admission_queue[0].arrival+540)
        for state in self.states.values():
            if state.end is None and state.arrival is not None and state.work.cancel_at_s is not None:
                events.append(state.work.cancel_at_s)
            if state.hold_until is not None and state.work.request_id in self.ledger:
                events.append(state.hold_until)
        return min((x for x in events if x>self.now+1e-12),default=None)

    def complete(self):
        job = self.queue[0]
        # The hive runs independently of EngineCore's blocking result().
        # Arrivals, admission deadlines and cancellation timers advance while
        # the scheduler stays blocked. Input is visible at the next schedule.
        event = self.next_front_event()
        while event is not None and event <= job['end']:
            self.now = event
            self.drain_inputs()
            event = self.next_front_event()
        self.queue.popleft()
        self.now = job['end']
        # Cancellation during device blocking suppresses later output. Jobs
        # already committed still consume their cost and hold KV until drain.
        for rid,phase,prefill,outputs,n in job['rows']:
            state = self.states[rid]
            state.row.num_output_placeholders -= job['output_placeholders'].get(rid,0)
            assert state.row.num_output_placeholders >= 0
            if state.end is not None: continue
            cancel = state.work.cancel_at_s
            if cancel is not None and cancel<=self.now:
                self.finish(state,'cancelled',max(state.arrival,cancel));continue
            state.prefill_tokens += prefill
            if not outputs: continue
            if phase=='prefill': count = 1
            else:
                state.credit += min(n,self.timing.committed(state.work.kind,job['decode_concurrency']))
                count = int(state.credit+1e-12)
                state.credit -= count
            count = min(count,state.work.output_tokens-state.tokens)
            if count:
                if state.first is None: state.first = self.now
                state.tokens += count
                state.events.append((self.now,count))
            if state.tokens >= state.work.output_tokens: self.finish(state,'completed',self.now)

    def run(self, works, metadata=None, already_decoding=()):
        if not works: raise ValueError('empty workload')
        if len(works)>50000: raise ValueError('workload exceeds 50000-row bound')
        self.timing.sources.clear()
        self.timing.unsupported.clear()
        self.states = {w.request_id:State(w) for w in works}
        if len(self.states)!=len(works): raise ValueError('duplicate workload request ids')
        self.children, self.pending = {},[]
        self.order = {w.request_id:i for i,w in enumerate(works)}
        for work in works:
            if work.request_id in already_decoding:
                if work.arrival_s != 0 or work.after_request_id or work.cancel_at_s is not None:
                    raise ValueError('conditional already-decoding input must start at zero without dependency/cancel')
                state = self.states[work.request_id]
                state.arrival = 0
                self.admit(state)
                self.s.waiting.remove(state.row)
                self.s.running.append(state.row)
                state.row.num_computed_tokens = work.prompt_tokens
                state.tokens = 1
                state.first = state.started = 0
                state.events = [(0,1)]
                continue
            if work.after_request_id:
                if work.after_request_id not in self.states: raise ValueError('unknown dependency')
                self.children.setdefault(work.after_request_id,[]).append(self.states[work.request_id])
            else: heapq.heappush(self.pending,(work.arrival_s,self.order[work.request_id],work.request_id))
        while True:
            if self.now>self.max_time: raise ValueError('replay time bound exceeded; no partial result claimed')
            self.drain_inputs()
            if all(s.end is not None for s in self.states.values()) and not self.queue: break
            launched = self.schedule() if self.s.running or self.s.waiting else False
            # Match EngineCore: A1 returns without a block; A2 is committed
            # before blocking on A1. Serial timing, capacity exactly two.
            if launched and len(self.queue)<self.depth: continue
            if self.queue: self.complete();continue
            events = []
            if self.pending: events.append(self.pending[0][0])
            for state in self.admission_queue: events.append(state.arrival+540)
            for state in self.states.values():
                if state.end is None and state.arrival is not None and state.work.cancel_at_s is not None:
                    events.append(state.work.cancel_at_s)
                if state.hold_until is not None and state.work.request_id in self.ledger:
                    events.append(state.hold_until)
            events = [x for x in events if x>self.now+1e-12]
            if not events:
                if self.s.waiting:
                    for r in list(self.s.waiting): self.finish(self.states[r.request_id],'engine-capacity-unknown',self.now)
                    continue
                raise ValueError('workload dependency deadlock')
            self.now = min(events)
        return self.summary(metadata)

    def summary(self, metadata):
        rows, buckets, waits, stalls = [],{},[],[]
        for state in self.states.values():
            work = state.work
            ttft = state.first-state.arrival if state.first is not None else None
            wait = state.admitted-state.arrival if state.admitted is not None else (state.end-state.arrival if state.arrival is not None else None)
            gaps = [b[0]-a[0] for a,b in zip(state.events,state.events[1:])]
            stall = max(gaps,default=None)
            if state.first is not None and state.end>state.events[-1][0]:
                stall = max(stall or 0,state.end-state.events[-1][0])
            if stall is not None: stalls.append(stall)
            if wait is not None: waits.append(wait)
            group = buckets.setdefault(bucket(work.prompt_tokens),{'ttft':[],'submitted':0,'no_first':0})
            group['submitted']+=1
            if ttft is not None: group['ttft'].append(ttft)
            else: group['no_first']+=1
            rows.append({'request_id':work.request_id,'session_id':work.session_id,'arrival_s':state.arrival,
                         'prompt_tokens':work.prompt_tokens,'cached_prefix_tokens':work.cached_prefix_tokens,
                         'status':state.status,'admitted_s':state.admitted,'submitted_s':state.submitted,'started_s':state.started,'first_s':state.first,
                         'end_s':state.end,'ttft_s':ttft,'queue_wait_s':wait,
                         'engine_queue_wait_s':state.started-state.admitted if state.started is not None else None,
                         'output_tokens':state.tokens,'max_decode_stall_s':stall,'prefill_tokens':state.prefill_tokens,
                         'kv_reservation':demand(work) if self.admission else None,
                         'evidence':'estimated replay; output event timestamps assumed batch completion',
                         'input_evidence':work.evidence})
        during = []
        for prefill in self.states.values():
            if prefill.first is None or prefill.started is None: continue
            lo,hi = prefill.started,prefill.first
            if hi<=lo: continue
            for decoder in self.states.values():
                if decoder is prefill or decoder.first is None or decoder.first>=lo: continue
                end = min(hi,decoder.end)
                if end<=lo: continue
                tokens = sum(n for t,n in decoder.events if lo<t<=end)
                before = [t for t,n in decoder.events if t<=lo]
                after = [t for t,n in decoder.events if t>lo]
                event_times = ([before[-1]] if before else [lo])+[t for t in after if t<=end]
                # Boundary-censored maximum includes the right interval edge.
                event_times.append(end)
                max_gap = max((min(b,end)-max(a,lo) for a,b in zip(event_times,event_times[1:])),default=0)
                during.append({'prefill_request':prefill.work.request_id,'decode_request':decoder.work.request_id,
                               'interval_s':end-lo,'tokens':tokens,'tok_s':tokens/(end-lo),
                               'max_stall_s':max_gap,'evidence':'estimated batch output events; boundary-censored gaps'})
        ttfts = [r['ttft_s'] for r in rows if r['ttft_s'] is not None]
        return {'schema':'schedreplay-result v1','policy':self.policy.name,'mode':self.mode,
                'evidence':'estimated offline replay; no hardware prediction guarantee',
                'sequence_evidence':'hash-pinned native Scheduler.schedule with inert KV/output boundaries' if self.mode=='exact' else 'independent stdlib metadata subset',
                'max_num_seqs':self.max_seqs,'memory_gate':'unqualified; hypothetical max-num-seqs 6' if self.max_seqs==6 else 'configured serving max-num-seqs 4; no memory qualification',
                'admission':self.admission,'admission_evidence':'assumed strict FIFO/64 queue/540s/block-rounded reservations, synthetic engine floor; cancel drain idle or 15s',
                'engine_kv_floor_tokens':self.floor,'batch_queue_depth':self.depth,'peak_batch_queue':self.peak_queue,
                'queue_evidence':'audited two-deep schedule-ahead behavior; assumed serial device timing',
                'verify_width':self.width,'execution_contract':'configured K3; c1 K3/kstop width4, c2/c4 uniform K2/reuse width3',
                'cycle_provider':self.timing.cycle_provider,'timing_sources':sorted(self.timing.sources),'workload':metadata or {},
                'timing_supported':not self.timing.unsupported,'unsupported_timing_regimes':sorted(self.timing.unsupported),
                'steps':self.steps,'mixed_steps':self.mixed_steps,'duration_s':self.now,
                'ttft_s':percentiles(ttfts),'ttft_by_prompt_bucket':{k:{**percentiles(v['ttft']),'n':len(v['ttft']),
                    'submitted':v['submitted'],'no_first':v['no_first']} for k,v in buckets.items()},
                'queue_wait_s':percentiles(waits),'max_decode_stall_s':max(stalls,default=None),
                'decode_during_prefills':during,
                'decode_during_prefills_tok_s':sum(x['tokens'] for x in during)/sum(x['interval_s'] for x in during) if during else None,
                'prefill_throughput_tok_s':self.prefill_count/self.prefill_work_s if self.prefill_work_s else None,
                'prefill_throughput_evidence':'estimated uncached prompt tokens / charged prefill+mixed step time; excludes pure decode floors',
                'peak_reserved_tokens':self.peak_reserved,'engine_capacity_refusals':self.engine_capacity_refusals,
                'total_requests':len(rows),'completed':sum(r['status']=='completed' for r in rows),'no_first':sum(r['first_s'] is None for r in rows),
                'requests':rows}


def pareto(results, baseline):
    """Whole-run frontier within supported cases, plus metric-scoped comparisons."""
    from lab.schedreplay.comparisons import METRICS, compare
    names = list(METRICS)[:6]
    table = compare(results,baseline,names,'supported whole-run cases; excluded cases listed per row')
    for row,result in zip(table,results):
        row.update({name:getter(result) for name,(getter,_) in METRICS.items() if name in names})
        value,base = result['ttft_s']['p90'],baseline['ttft_s']['p90']
        row['ttft_p90_change_pct'] = (value/base-1)*100 if value is not None and base not in (None,0) else None
        row['completed'] = result['completed']
        row['memory_gate'] = result['memory_gate']
        row['metric_comparisons'] = {name:compare([result],baseline,[name],'whole-run '+name)[0]
                                     for name in names}
    return table
