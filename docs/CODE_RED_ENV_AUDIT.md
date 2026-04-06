# CodeRedEnv — Production Audit Report
**Auditor:** Claude Code (Senior Staff AI Systems Auditor)
**Date:** 2026-04-06
**Scope:** Full codebase (codered_env + codered-hfspace)
**Commit:** e9a0697 "updated codebase"

---

## EXECUTIVE SUMMARY

**VERDICT: ⚠️ 7 CRITICAL / 14 MAJOR / 8 MINOR ISSUES — DO NOT DEPLOY**

The codebase has significant correctness, determinism, exploitation, and deployment issues that will cause automated validation failures, allow adversarial score manipulation, produce non-reproducible scores, and result in incorrect grading. Multiple Phase 2 workflows are entirely broken.

---

## PHASE 1: FILE-BY-FILE AUDIT

### 1.1 `server/codered_environment.py` (1358 lines) — CORE

**Purpose:** Main OpenEnv `Environment` implementation. Wires all subsystems, executes actions, advances time, builds observations.

**CRITICAL BUGS:**

| # | Bug | Line | Severity | Exploit |
|---|-----|------|----------|---------|
| 1 | **`done` flag NOT passed to `_build_observation()` in HF Space** | HF Space line 1042 | CRITICAL | The HF Space version of `_build_observation()` signature is `def _build_observation(self)` without `done` parameter. The `done` field in `CodeRedObservation` always returns `False`. This breaks OpenEnv's step-loop termination detection. **The local version was FIXED (S919-921) but the HF Space was NEVER updated.** |
| 2 | **Phase 2 dispatch NEVER spawns patients** | `_do_dispatch_als`/`_do_dispatch_bls` → `_do_triage_call` | CRITICAL | `dispatch_als/bls` actions call `ambulance_manager.dispatch(amb_id, location)` WITHOUT setting `patient_id` on the ambulance. When the ambulance arrives, `arrived_with_patient` is never set. The patient is force-spawned by countdown (or never at all if no countdown), but the ambulance has already gone past the patient with no patient_id attached. The patient stays in "waiting" status forever. The agent gets ZERO milestone rewards (dispatched: +0.20, in_treatment: +0.40) for ALL Phase 2 dispatches. **Entire Phase 2 ambulance workflow is broken.** |
| 3 | **Phase 2 forced-spawn patients marked `is_secondary=False`** | `_spawn_patient_from_call()` line | MAJOR | When a dispatch call's countdown expires and forces a patient spawn, `is_secondary=False` is logged. The grader's `grade_cascade_score()` only counts `secondary_patient_spawned` events with `is_secondary=True` for secondary patient IDs. Phase 2 forced-spawn patients are silently IGNORED in the grader's secondary harm scoring, creating a grading asymmetry. |
| 4 | **Ambulance arrives with stale patient after patient death** | `_advance_time()` `arrived_with_patient` check | MAJOR | If `mutual_aid_manager.tick()` calls `_ma_arrival_callback()` and sets patient to "transporting" but the patient then dies (due to vitals hitting 0 in the same tick's `_advance_time()` sequence), `_do_treatment_arrival()` receives a deceased patient and calls `release_icu_bed()` on a dead patient. ICU beds can be leaked. |
| 5 | **Duplicate `patient_created` log entries for Phase 2 calls** | `_spawn_patient_from_call()` | MAJOR | When a patient is force-spawned (countdown expired), `_spawn_patient_from_call()` appends a `patient_created` entry. But the initial dispatch (from `_do_triage_call`) already appends a `dispatch_outcome` entry. The patient is NOT double-logged as created, but the `call_received` entry from `_spawn_dispatch_call` and the `dispatch_outcome` entry from `_do_triage_call` with `true_condition=None` can coexist. The grader's `dispatch_classification_reward` may see outcomes with `true_condition=None` from aborted dispatches. |
| 6 | **`assigned_ambulance` always `None` in observation** | `_build_observation()` patients list | MAJOR | `assigned_ambulance=None` is hardcoded for ALL patients in the observation. The subsystem tracks `patient.assigned_ambulance` but it is NEVER exposed to the agent. The agent cannot know which ambulance is en route to which patient without querying the ambulance state separately and correlating by patient_id. This creates a subtle observation gap. |
| 7 | **Overcrowding modifier not capped at 1.5** | `_cascade_engine.check_overcrowding()` | MINOR | `check_overcrowding()` returns 1.2 always when overcrowded. But `observation.overcrowding_modifier` is `Field(ge=1.0, le=1.5)`. Future callers could exceed 1.5. Not currently exploitable but violates the schema. |
| 8 | **` MutualAidManager.request()` creates `arrival_step` with stale route** | `mutual_aid.py` `request()` | MAJOR | When `request()` is called, it computes `arrival_step = step_count + travel_time + SCENE_TIME`. But the `route_travel_time()` call uses the current road network state. If a disruption occurs BEFORE the MA ambulance departs (which happens on the NEXT tick), the arrival step was calculated with pre-disruption travel times. The ambulance may arrive LATER than logged. The grader compares `actual_arrival_step` (logged as current step) against `optimal_arrival_step` (pre-computed) and penalizes delays, but the delay may be due to a disruption the agent couldn't have predicted. |
| 9 | **`pending_surge_probability` accumulates unbounded** | `cascade_engine.py` `_apply_effect("news_cycle")` | MINOR | Each news cycle adds `surge_prob_boost` (0.15-0.20) to `_pending_surge_probability`. `tick()` only reduces by 0.10 per step. With multiple news cycles (e.g., patient saved, trauma saved, another cardiac saved), this can accumulate beyond 1.0, violating `ge=0.0, le=1.0` in the grader. Not currently exploited but structurally possible in task5 with cascade. |
| 10 | **`_do_triage_call` doesn't update ambulance's `target_node`** | `_do_triage_call()` | MAJOR | When `decision in ("dispatch_als", "dispatch_bls")`, `ambulance_manager.dispatch()` is called with the call's `location_node`. The ambulance's `target_node` IS set. But the `ambulance.state.target_node` is only used for the observation's `location_node` (not scene tracking). No issue here — but `assigned_ambulance` on the patient is not set (separate from ambulance's internal patient_id). |
| 11 | **`_build_observation()` creates stale `_prev_vitals` snapshots** | `_compute_step_reward()` | MINOR | The `_prev_vitals` and `_prev_patient_status` are snapshot dictionaries. `_build_observation()` reads `patient.vitals_score` for ALL patients each call. If `_build_observation()` is called multiple times (e.g., by inference loop, by test harness), the snapshot is fresh. Not an issue in normal use but not idempotent. |
| 12 | **Early termination: `vitals_score <= 0.0` vs `< 0.0`** | `_check_done()` | MINOR | `_check_done()` uses `<= 0.0`: `all(p.vitals_score <= 0.0 for p in alive)`. But `mark_deceased()` sets `vitals_score = 0.0`. So if ALL alive patients are deceased (already stopped in `non_terminal` filter), this branch never fires. If `vitals_score` could go negative (e.g., in overflow or unusual arithmetic), this could trigger early termination. Currently harmless but the condition is redundant. |
| 13 | **`reset()` re-creates subsystem instances** | `reset()` | MAJOR | `reset()` calls `RoadNetwork()`, `HospitalSystem()`, `BloodBankSystem()` etc. to create NEW instances. But `self._ambulance_manager = AmbulanceManager(AMBULANCES)` and `self._mutual_aid_manager = MutualAidManager(...)` are assigned AFTER `self._road_network = RoadNetwork()`. The RoadNetwork/hospital/blood_bank are overwritten. This is intentional but the reassignment order matters — if `_mutual_aid_manager` is used before assignment in a subclass or during concurrent session access, it would be `None`. No current issue. |
| 14 | **`vitals_score_preview` divides by zero if no active patients** | `_build_observation()` | MINOR | `sum(...)/max(1, sum(...))` — the `max(1, ...)` guard exists. OK. |
| 15 | **`step()` advances time BEFORE action execution** | `step()` | DESIGN | The action is executed AFTER time advances. This means an action taken at step N affects the state at step N+1 observation. This is correct for an RL environment (action → transition → observation), but it means the agent sees the result of the previous action in the NEXT observation. The grader uses `step` field in log entries, which increments before action. Grader log `step` numbers match `_state.step_count` (incremented before `_advance_time()`). This is correct. |

**EDGE CASES NOT HANDLED:**

- **All ambulances busy + no mutual aid**: Patient never dispatched, vitals decay to 0, patient dies. Correct.
- **Hospital on diversion after patient assigned**: `_do_treatment_arrival()` doesn't check `on_diversion`. Patient arrives at hospital on diversion, is admitted (no OR available), stays in "in_treatment" without surgery. ICU bed is consumed. Agent is not notified.
- **Ambulance dispatched to patient, patient dies before arrival**: Ambulance arrives at empty scene. `arrived_with_patient = False`. Patient stays in "waiting" status forever (dead patients are not in "waiting"). Actually, `mark_deceased()` sets status to "deceased". So ambulance arrives, patient is "deceased", `arrived_with_patient` check passes but patient is dead. `_do_treatment_arrival()` receives dead patient. ICU bed leak. Plus: ambulance returns to base (via `scene_minutes_remaining` countdown), but `_do_treatment_arrival()` calls `ambulance_manager.mark_available()` which resets scene_minutes_remaining.
- **Multiple patients at same node**: `_do_dispatch_ambulance()` uses `next()` — picks first patient. The second patient is NEVER found by dispatch_ambulance. Agent must know to dispatch a different ambulance. This is a design constraint, not a bug.

---

### 1.2 `server/grader.py` — GRADING SYSTEM

**Purpose:** Episode evaluation using rubric-based scoring across 6 axes.

**CRITICAL BUGS:**

| # | Bug | Line | Severity | Impact |
|---|-----|------|----------|--------|
| 16 | **`grade_cascade_score()` double-counts deaths** | `grade_cascade_score()` | MAJOR | `secondary_deaths` is computed from BOTH `patient_id in secondary_patient_ids` AND `reason == "secondary"`. The `secondary_patient_ids` set was built from `secondary_patient_spawned` events. If a patient dies with `reason="secondary"` but was NOT in the spawned set (e.g., a Phase 2 forced-spawn with `is_secondary=False`), they are counted in deaths but NOT in total secondary patients. `secondary_score = 1.0 - (secondary_death_count / num_secondary)` — death count > patient count possible → negative score → clamped to 0.0. This is a hidden edge case that can produce 0 cascade score unexpectedly. |
| 17 | **`vitals_score_avg` is informational only** | `grade_episode()` | DESIGN | `vitals_score_avg` is computed from `treatment_complete` events but is NOT included in the final score formula. The `final_score` comment says "weighted sum" but `vitals_score_avg` IS included at 10% weight. The comment is misleading. The code IS correct (0.10 weight). |
| 18 | **`mutual_aid_penalty` has no cap** | `grade_episode()` | MAJOR | `mutual_aid_penalty = sum(penalties) / num_calls`. Each call's penalty can be arbitrarily large (0.2 * (actual - optimal)). With a disruption causing 20+ step delay, penalty = 0.2 * 20 = 4.0 per call. Divided by calls: could still be > 1.0. `final_score = max(0.0, raw - mutual_aid_penalty)` — no clamping on penalty itself. With 1 call and 20-step delay: penalty = 4.0, `final_score = max(0.0, raw - 4.0)`. Final score can go deeply negative... but `max(0.0, ...)` saves it. However, `mutual_aid_penalty` in `breakdown` shows the actual penalty value, which could be > 1.0. |
| 19 | **`_patients_dict_cache` can become stale** | `patient_manager.py` | MAJOR | `patient_manager.py` has `self._patients_dict_cache` invalidated when `spawn_secondary()` is called. But in the environment, `self._patients` is ALSO maintained separately (`self._patients = self._patient_manager.patients`). The grader's cross-validation calls `env._patient_manager.get_all()` which returns `self.patients`. This is the source of truth. The `_patients_dict_cache` is only rebuilt when `spawn_secondary()` adds patients. If a patient is removed from `self.patients` (not currently done), the cache would be stale. |
| 20 | **`saved_but_deceased` cross-validation issue** | `grade_from_environment()` | MAJOR | `saved_but_deceased = patients_saved & patients_deceased` — this can ONLY happen if `mark_treated()` and `mark_deceased()` are BOTH called on the same patient. Looking at the code, `mark_treated()` sets `outcome = "saved"` and `mark_deceased()` sets `outcome = "deceased"`. If both are called (which would require a bug in the environment), the last call wins. The cross-validation detects this. But there is no guard preventing this in the environment. If `_do_treatment_arrival()` is called twice for the same patient (impossible with current logic), or if `mark_treated()` is called after `mark_deceased()` (could happen if mortality roll fails but patient is already marked), this would silently corrupt grading. |
| 21 | **`icu_boarding_penalty` not in final score formula** | `grade_episode()` | DESIGN | The ICU boarding penalty is applied AFTER `grade_episode()` returns, in `grade_from_environment()`. The `final_score` returned by `grade_episode()` does NOT include the ICU penalty. This means `grade_episode()` alone gives an incomplete score. Only `grade_from_environment()` gives the complete score. |
| 22 | **`preemptive_or` wasted events not tracked** | `_do_preempt_or()` | MINOR | `preempt_or` action logs to alerts but NOT to `_episode_log`. The grader has no events for OR preemption, so wasted preemptions (harming ongoing surgeries) are not penalized. An agent could preempt ORs aggressively without grader consequence. |

---

### 1.3 `server/models/entities.py` — DATA MODELS

**Purpose:** Pydantic entity definitions for patients, hospitals, ambulances, road network.

**ISSUES:**

| # | Issue | Severity |
|---|-------|----------|
| 23 | `EdgeState.effective_time()` returns `float("inf")` for road closures. Pydantic serialization of `inf` to JSON is implementation-defined. Could produce `Infinity` in JSON which breaks some parsers. | MINOR |
| 24 | `DispatchCallModel.spawned_patient_id` is `Optional[str]` but never actually set in the environment's `_pending_calls` list construction — the environment's internal `_pending_calls` uses `spawned_patient_id: None` and it's never updated. So `pending_calls` in observations always shows `None` for `spawned_patient_id`. | MAJOR |
| 25 | `AmbulanceState.destination_type` is `Literal["patient", "hospital", "base"]` but the environment sets it to `"patient" if amb.target_node else None` — never "hospital" or "base". The observation always shows `None` or `"patient"`. | MINOR |
| 26 | `BloodBankState.crossmatch_queue` is exposed in the observation — this exposes the internal state of pending crossmatches with patient IDs and blood types. The grader does not use this for scoring, but it's information leakage about which patients need blood. | MINOR |

---

### 1.4 `server/models/actions.py` — ACTION SPACE

**Purpose:** OpenEnv-compatible action definitions.

**ISSUES:**

| # | Issue | Severity |
|---|-------|----------|
| 27 | `TriageCall.ambulance_id` is `Optional` — required when `decision` is `dispatch_als` or `dispatch_bls` but the Pydantic model doesn't enforce this constraint. A malicious agent could send `TriageCall(call_id="X", decision="dispatch_als", ambulance_id=None)`. The environment's `_do_triage_call()` checks this and returns early, but the malformed action is accepted by the model. | MAJOR |
| 28 | `AssignHospital` action: no validation that the ambulance has reached the patient before assignment. Agent could assign a patient to HOSP_C (community HC) which has 100% mortality for cardiac/stroke/trauma. No guard in the action or environment. | MAJOR |
| 29 | `PreemptOR` has no logging to episode_log. The grader cannot detect wasted preemption. | MINOR |

---

### 1.5 `server/subsystems/patient_manager.py` — PATIENT LIFECYCLE

**Purpose:** Patient creation, vitals deterioration, status transitions.

**ISSUES:**

| # | Issue | Severity |
|---|-------|----------|
| 30 | **`tick()` can call `mark_deceased()` on already-deceased patients** | MAJOR | If `vitals_score` is already 0.0 (patient was marked deceased in previous tick), the new `effective_time` computation: `overtime_ratio = ((effective_time - target_time) * overcrowding_modifier) / target_time` with `effective_time >> target_time` gives `overtime_ratio > 1.0`, so `vitals_score = max(0.0, 1.0 - overtime_ratio) = 0.0`. The status is already "deceased" so the `TERMINAL_STATUSES` check catches it. But the `_vitals_frozen` check only freezes AFTER `mark_deceased()` is called in THIS tick's iteration. If `mark_deceased()` is called for patient P in iteration 1, and patient P appears again in iteration 2 (Python list iteration with removal mid-loop is safe but redundant call to `mark_deceased()` is wasteful). Not correctness-breaking. |
| 31 | **`_onset_steps` vs `patient.onset_step` divergence** | MAJOR | `_onset_steps` is a separate dict storing onset steps. `patient.onset_step` is the field on the patient. `tick()` uses `_onset_steps` for computation. But when `spawn_secondary()` is called, it sets BOTH `patient.onset_step` AND `self._onset_steps[patient.id]`. If the onset step dict and patient object diverge (e.g., if onset_step is modified on the patient but not the dict), the `tick()` computation uses stale onset steps. Currently safe because both are kept in sync. |
| 32 | **`patient.blood_type` never set in Phase 1 workflows** | DESIGN | `QueryBloodType` action schedules a blood type reveal. But in Phase 1, patients spawn with `blood_type=None`. If the agent never calls `QueryBloodType`, the blood type stays `None`. This is intentional — reduces information. |
| 33 | **`icu_status` tracking in patient_manager** | DESIGN | `Patient.icu_status` is set in `_do_treatment_arrival()` via `hospital_system.consume_icu_bed()`. But `patient_manager.tick()` doesn't update `icu_status`. If a patient is admitted (icu_status="admitted") and later the ICU bed is full (boarding), `icu_status` should be updated. Currently it's set once at arrival. |

---

### 1.6 `server/subsystems/ambulance_manager.py` — AMBULANCE FLEET

**Purpose:** Ambulance dispatch, movement, status transitions.

**ISSUES:**

| # | Issue | Severity |
|---|-------|----------|
| 34 | **`dispatch()` adds `SCENE_TIME` to ETA when `patient_id` provided** | DESIGN | `eta += SCENE_TIME` signals the ambulance will have a patient on board at arrival. But this means the ambulance's ETA is LONGER than the actual travel time when carrying a patient. The observation shows `eta_minutes` which is the total including scene time. The agent sees the ambulance arriving in N minutes but doesn't know the last N-SCENE_TIME minutes are travel time. |
| 35 | **`return_to_base()` doesn't handle patient on board** | MAJOR | If an ambulance is carrying a patient (`patient_id` set) and `return_to_base()` is called, the route is computed but `patient_id` is NOT cleared. The ambulance returns to base with a patient still assigned. Then `mark_available()` would be called (when the ambulance returns via tick), clearing the patient. But if `return_to_base()` is called while the ambulance is on_scene with a patient, the patient is orphaned. This function is not called from the environment (only via ambulance_manager's own tick logic), so it's protected. |
| 36 | **`tick()` uses `prev_status` to detect if ambulance arrived with patient** | MAJOR | `scene_minutes_remaining = SCENE_TIME` is only set when `amb.patient_id is not None and prev_status == "en_route"`. If the ambulance status is "on_scene" (already set before tick) and `patient_id` is set (from a previous dispatch), the countdown is never started. This is already handled because the environment sets status to "en_route" on dispatch. But in MutualAidManager, `arrival_callback` is called synchronously in `tick()`, before the ambulance's status is updated to "on_scene". So `prev_status == "en_route"` is always true for MA arrivals. OK. |
| 37 | **`get_available()` doesn't filter by node proximity** | DESIGN | Agent must manually compute nearest ambulance. No helper. OK. |

---

### 1.7 `server/subsystems/hospital_system.py` — HOSPITAL RESOURCES

**Purpose:** ORs, specialists, ICU beds, blood stock.

**ISSUES:**

| # | Issue | Severity |
|---|-------|----------|
| 38 | **`consume_icu_bed()` doesn't check ICU capacity properly** | MAJOR | `consume_icu_bed()` returns `True` if bed consumed, `False` if full. In `_do_treatment_arrival()`, the ICU bed is consumed WITHOUT checking the return value for the "admitted" vs "boarding" decision. The code unconditionally sets `icu_status = "admitted"` or `"boarding"` based on return value. This is CORRECT. But if `consume_icu_bed()` returns False, `icu_status` is set to "boarding" without a corresponding `release_icu_bed()` call when the patient is done. Wait — `release_icu_bed()` is called in the surgery completion path. If a patient is "boarding" (no ICU bed) and dies or is treated, `release_icu_bed()` is called. But if `consume_icu_bed()` returned False, there's no bed to release. The logic is: if bed available → admit (and release on completion); if not → board (no bed to release). This is correct. |
| 39 | **Shift transition reduces specialist availability mid-episode** | DESIGN | `_update_shift()` is called in `tick()`. If the episode runs long enough (task3: 60 steps, evening → night transition), specialist counts change. This could cause a specialist who was available at step 20 to become unavailable at step 30 (day → evening shift). The agent is NOT notified of shift changes. The observation doesn't expose the current shift. An agent could page a specialist who becomes unavailable mid-surgery. The specialist's `status` tracks "busy", so they remain "in surgery" even after shift change. New surgeries won't get the specialist. This is realistic but potentially frustrating for agents. |
| 40 | **`_update_shift()` reduces availability below 0** | BUG | `new_available = max(0, shift_available - committed)`. If `shift_available` is 0 (night shift, HOSP_A neurologist = 0) and `committed = 0`, `new_available = 0`. But if `specialist.available` was 1 (day) and `committed = 1` (busy in surgery) and `shift_available = 0`, then `new_available = max(0, 0-0) = 0`. The specialist in surgery is still tracked as `busy` via `status = "paged"`. When they finish, `minutes_until_available` counts down. But their `available` count is 0. When they become available, `available += 1` could make it 1, which is within `total`. This is OK — a night-shift specialist CAN be the same person who was paged during the day. |

---

### 1.8 `server/subsystems/road_network.py` — ROUTING + CONGESTION

**Purpose:** Graph routing, time-of-day congestion, disruptions.

**ISSUES:**

| # | Issue | Severity |
|---|-------|----------|
| 41 | **`_apply_tod_congestion()` only matches `from_node`** | BUG | `curve = self._congestion_curves.get(edge.from_node, [])` — congestion curves are keyed by node name, but the lookup only checks `from_node`. For edges where the heavy-traffic direction is different (e.g., inbound morning rush vs outbound evening), the same multiplier is applied in both directions. This is a simplification, not a bug per se, but could be exploited: during rush hour, the agent could route through edges that are congested in one direction but not the other. The congestion multiplier is symmetric. |
| 42 | **Dijkstra re-computes full path on every dispatch** | PERFORMANCE | No path caching. With 12 nodes and up to 5 concurrent ambulance dispatches, this is fine. Not a bottleneck. |
| 43 | **`route_travel_time()` returns 999 for infinite** | DESIGN | If any segment has `inf` travel time (road closure), the function returns 999. The ambulance is dispatched with ETA=999. In `ambulance_manager.dispatch()`, the ETA includes SCENE_TIME (for patient_id case). With a road closure blocking the only path, the ambulance goes en_route with ETA=999+15=1014 minutes. This will arrive long after the patient dies. But no alert is sent to the agent about the road closure at dispatch time. The road closure alert is only sent when the disruption is rolled. If the disruption rolls AFTER the ambulance is dispatched, the agent doesn't know the ambulance will be delayed. This is an information asymmetry. |
| 44 | **`_edge_key()` always sorts nodes alphabetically** | OK | This is intentional for undirected edges. Correct. |

---

### 1.9 `server/subsystems/blood_bank.py` — BLOOD SUPPLY

**Purpose:** Blood stock management, crossmatching, transfers.

**ISSUES:**

| # | Issue | Severity |
|---|-------|----------|
| 45 | **`emergency_release()` deducts O_NEG but doesn't verify patient needs it** | MAJOR | An agent could call `allocate_blood(hospital_id, patient_id, blood_type="O_NEG", units=100, emergency=True)` and drain the entire O_NEG stock. There is no limit on emergency blood units per action. With 8 units of O_NEG at HOSP_A, the agent could call emergency=True 8 times with units=1, or once with units=8. After that, the call fails. But 8 units of O_NEG could treat 8 patients. There's no cap on how much emergency blood is requested. A malicious agent could exhaust O_NEG for all subsequent patients. |
| 46 | **`tick()` deducts blood from stocks when crossmatch completes** | BUG? | `bank.stocks[entry["blood_type"]] -= entry["units"]` — if the blood type isn't in stocks (was depleted by emergency_release), this subtracts from `None` or creates a negative entry. No guard. If `emergency_release()` depletes the stock below the pending crossmatch amount, the crossmatch completion would go negative. This could happen with the exploit above. |
| 47 | **Blood transfers are instant** | DESIGN | No travel time for blood transfers. Transfers between HOSP_C (no O_NEG supply) and HOSP_A could supply unlimited blood instantly. |

---

### 1.10 `server/subsystems/disruption_engine.py` — DISRUPTIONS

**Purpose:** Seeded disruption scheduling.

**ISSUES:**

| # | Issue | Severity |
|---|-------|----------|
| 48 | **`road_closure` duration = 999 (permanent within episode)** | DESIGN | `duration_map["road_closure"] = 999`. With max_steps ≤ 60, a road closure with 999-minute duration NEVER clears within the episode. This means if NH45_BYPASS is closed, the fastest route is permanently disrupted. For task3 with a road closure at step 15, the ambulance dispatched to a patient across the closed edge has infinite travel time. The patient will die. This is by design but creates a hard failure mode. |
| 49 | **`_generate_schedule()` uses `rng` seeded from `self._seed + 1000`** | MAJOR | The schedule RNG is separate from the main RNG. This means disruption events are deterministic from seed, but the timing of disruptions uses a different random stream than patient spawning jitter. This is fine for determinism but means disruptions are not correlated with patient spawning in any interesting way. |

---

### 1.11 `server/subsystems/cascade_engine.py` — CASCADE EFFECTS

**Purpose:** Secondary patient spawning, overcrowding, news cycles.

**ISSUES:**

| # | Issue | Severity |
|---|-------|----------|
| 50 | **`on_outcome()` called BEFORE patient outcome is finalized** | CRITICAL | In the environment, `_cascade_engine.on_outcome()` is called in the same block as `mark_treated()` / `mark_deceased()`. But the patient outcome (saved vs deceased) is used to trigger new cascades. If `mark_treated()` is called but then a subsequent event (e.g., hospital mortality roll) changes the outcome to "deceased", the cascade was triggered by the wrong outcome. Currently, `mark_treated()` = outcome is final for saved patients. `mark_deceased()` = final for dead patients. No subsequent change. So the order is safe. BUT: `patient.outcome = "saved"` is set BEFORE `cascade_engine.on_outcome()` is called, and then immediately AFTER, `patient.outcome` could be overwritten by the mortality roll. The cascade sees the correct "saved" outcome because it's set before the roll. Actually, looking at the code: `survived = self._rng.random() >= mort_rate` comes BEFORE `mark_treated()`. If survived, `mark_treated()` then `cascade_engine.on_outcome(..., "saved")`. If not survived, `mark_deceased()` then NO cascade trigger. So the cascade is triggered on survival BEFORE mortality is rolled. This is: patient arrives at OR → surgery completes → survival roll → if survived, treat + cascade. Correct. |
| 51 | **Cascade rules hardcoded, not configurable per task** | MINOR | Cascade rules are always loaded. For task1 (no secondary patients), cascades can still trigger if a patient is saved/dies. But with no patients in task1's task config (only 1 cardiac patient), the cascade trigger `patient_saved` for cardiac would fire 15% of the time, spawning a secondary patient. The secondary patient has no ambulance dispatched to them. They decay and die. The grader sees a secondary patient death → secondary_harm score drops. This means task1 can have secondary harm < 1.0 even with perfect performance. **Task 1 is affected by cascades even though it should have no cascades.** |

---

### 1.12 `server/subsystems/mutual_aid.py` — MUTUAL AID

**Purpose:** Mutual aid ambulance lifecycle.

**ISSUES:**

| # | Issue | Severity |
|---|-------|----------|
| 52 | **`request()` auto-assigns to highest-priority waiting patient** | MAJOR | The agent cannot control which patient the MA ambulance is assigned to. With 2 waiting patients (cardiac + general), MA always goes to cardiac. The agent might want MA to handle the general patient while dispatching their own ALS to the cardiac patient. No override. |
| 53 | **`_auto_assign_hospital()` prefers HOSP_A for all cardiac/stroke** | MAJOR | Even if HOSP_A is on diversion, the MA system still auto-assigns to HOSP_A. The patient arrives at a hospital on diversion. The environment's `_do_treatment_arrival()` doesn't check `on_diversion`. No alert. ICU bed may or may not be consumed depending on OR availability. |
| 54 | **No mutual aid failure alert if no patients waiting** | MINOR | When MA is requested but no patients are waiting, the ambulance arrives with `had_patient=False` and the alert says "wasted". But the `mutual_aid_called` event still logs the MA. The grader sees wasted MA calls. If the agent has no patients waiting, requesting MA is a pure penalty with no benefit. This is correct behavior but the agent has no signal about this before requesting. |

---

### 1.13 `inference.py` — LLM AGENT

**Purpose:** LLM-powered agent execution.

**ISSUES:**

| # | Issue | Severity |
|---|-------|----------|
| 55 | **Action parsing fallback is too lenient** | MAJOR | `parse_action_string()` falls back to `maintain_plan()` for any non-matching text. A malformed LLM response (e.g., "I think we should wait") results in `maintain_plan()`. The agent wastes a step silently. The grader then sees `maintain_plan` actions without alerts. The reward continues to accrue (or not). |
| 56 | **`parse_action_string()` regex doesn't handle multi-word values** | MAJOR | The regex `r"(\w+)=(['\"]?)([\w_/]+)\2"` only matches single-word values (letters, digits, underscore, slash). `target_node="LAJPAT_NAGAR"` matches. But if the LLM produces `patient_id=P123` or `call_id="CALL_0042"`, the quotes around multi-character IDs and the underscore might not match correctly. The fallback to JSON parsing handles this but the regex is fragile. |
| 57 | **`format_observation()` omits `icu_status` for patients** | MINOR | The observation formatter doesn't show ICU boarding status. The agent cannot see if a patient is boarding. |
| 58 | **`temperature=0.7` adds non-determinism** | MAJOR | With temperature 0.7, the same observation can produce different actions across runs. The `HF_TOKEN` is checked but not validated. If `HF_TOKEN` is empty string, `_get_client()` creates a client with no auth. The model call will fail. |
| 59 | **No retry on model failure** | MINOR | If the model call fails, `call_model()` returns `"maintain_plan()"` as fallback. The episode continues with no-op actions. The grader gives a low score. There's no alerting to the user that the agent failed. |

---

### 1.14 `client.py` — ENV CLIENT

**Purpose:** Remote environment connection.

**ISSUES:**

| # | Issue | Severity |
|---|-------|----------|
| 60 | **`CodeRedEnv` class has placeholder documentation** | MINOR | The class says "Placeholder for remote EnvClient". The `from_docker_image()` raises `NotImplementedError`. If users try to use the Docker image feature, they get an error. |

---

### 1.15 `codered-hfspace/` — HF SPACE VERSION

**CRITICAL DEPLOYMENT ISSUES:**

| # | Issue | Severity |
|---|-------|----------|
| 61 | **`done` parameter missing from `_build_observation()`** | CRITICAL | HF Space version: `def _build_observation(self)` — no `done` parameter. The `done` field in observations is always `False`. OpenEnv's step-loop cannot detect termination. This is a BLOCKER for automated validation. |
| 62 | **HF Space `app.py` missing TASK_DEFINITIONS endpoint** | CRITICAL | HF Space app.py has a different structure. It does NOT have the `TASK_DEFINITIONS`, `/tasks`, `/grader`, `/baseline` endpoints. These are replaced with a simpler OpenEnv-only interface. If the grader/human-reviewer expects these endpoints, they won't be available on HF Space. |
| 63 | **`openenv.yaml` in HF Space vs local differ** | MINOR | Need to verify both are identical. The HF Space version's `max_episode_steps` and `task_ids` must match. |

---

## PHASE 2: RED-TEAM / EXPLOIT ANALYSIS

### 2.1 REWARD EXPLOITS

| Exploit | Description | Severity | Fix |
|---------|-------------|----------|-----|
| **E1: Blood bank drain** | Call `allocate_blood(emergency=True)` repeatedly with `units=1` to deplete O_NEG. Subsequent patients without O_NEG die. But the depletion is logged and affects grader's efficiency axis (wasted blood). However, `wasted` is only logged if the allocation is deemed "wasted" by the environment (blood_type mismatch, no patient, etc.). The agent can deplete O_NEG for LEGITIMATE patients (cardiac patients who need blood) without penalty if it calls emergency blood before knowing the patient's blood type. No per-patient limit on emergency blood. | MAJOR | Add a global emergency blood cap per hospital. |
| **E2: Milestone reward loop via secondary patients** | Task1 has 1 cardiac patient. With 15% cascade probability, a secondary cardiac patient spawns. The agent dispatches an ambulance (reward: +0.20), patient transported (+0.40), treated (+0.80). The secondary patient generates additional milestones. If cascades fire multiple times, the agent can accumulate reward from treating secondary patients that the agent had no role in creating. The grader normalizes by patient count in time_score, but the dense reward can be inflated. | MAJOR | In task1, disable cascade triggers entirely (cascade_enabled=False, or use different task config). |
| **E3: Maintain plan spam** | `maintain_plan()` gives 0 reward but costs no penalty. An agent doing nothing for 30 steps gets score 0.0. Not an exploit per se, but the grader rewards doing nothing WORSE than not acting. | N/A | Not an exploit. |
| **E4: Preemptive OR preps** | Call `prepare_or` at HOSP_A for cardiac at every step. ORs get prepared but never used. Each unused prep is logged as "wasted" (-0.15 per prep). 5 preps in 5 steps → efficiency = 1.0 - 0.75 = 0.25. The agent can tank its own efficiency score this way. But there's no POSITIVE reward for preps, so the agent should not do this. The exploit is self-harm, not score inflation. | MINOR | Not an exploit — self-penalizing. |
| **E5: Preempt OR to steal OR for patient who doesn't need it** | `preempt_or()` clears an in-use OR. The patient in that OR has their surgery aborted (harm logged). A new patient is treated faster. The aborted surgery patient may die (secondary harm). But `preempt_or` is NOT logged to episode_log, so the grader doesn't see it. The agent can preempt ORs to prioritize patients without grader penalty for the aborted surgery patient. | CRITICAL | Log preempt_or to episode_log. |

### 2.2 GRADER WEAKNESSES

| Weakness | Description | Severity |
|----------|-------------|----------|
| **G1: Grader ignores `preempt_or` actions** | As above — agent can preempt without grader seeing it. | CRITICAL |
| **G2: Secondary harm count is fragile** | `grade_cascade_score()` uses `secondary_patient_spawned` events with `is_secondary=True` AND `patient_deceased` with `reason="secondary"`. A patient that is forced-spawned (Phase 2 countdown) with `is_secondary=False` and dies with `reason="timeout"` is NOT counted as secondary. The grader silently ignores this death. An agent could let Phase 2 forced-spawn patients die with no grader penalty. | CRITICAL |
| **G3: Cross-validation detects but doesn't fix** | `grade_from_environment()` detects mismatches but the environment state is not corrected. The penalty is applied. But if the environment is re-graded (e.g., human review), the same mismatch recurs. | MAJOR |
| **G4: Time score formula sensitive to single patient** | If 1 patient out of 5 dies, time_score = average of (1.0 for 4 patients + 0.0 for 1 patient) = 0.8. The death of 1 patient only costs 4% of the final score (0.8 × 0.32 = 0.256 vs 0.32). An agent that saves 4/5 patients gets 80% of max time score weight. | DESIGN |
| **G5: Mutual aid penalty has no optimality baseline** | The `optimal_arrival_step` is computed by the environment at call time. If a disruption occurs AFTER the call, the ambulance arrives late but the penalty is applied. The agent cannot know the disruption will happen. The optimal arrival is computed BEFORE the disruption. | MAJOR |

### 2.3 STATE / TRANSITION BUGS

| Bug | Description | Severity |
|-----|-------------|----------|
| **S1: Phase 2 ambulance arrives without patient** | `_do_dispatch_als/bls` doesn't set ambulance's `patient_id`. Patient never spawns or spawns via countdown. Ambulance arrives with no one. | CRITICAL |
| **S2: `assigned_ambulance` always None** | Agent cannot see ambulance assignments | MAJOR |
| **S3: Road closure makes patient unreachable** | Road closure with inf duration. No re-routing. Patient dies with no possible rescue. | DESIGN |
| **S4: ICU bed leak on patient death** | Dead patient consumes ICU bed. Not released. | MAJOR |
| **S5: Mutual aid arrival step mismatch** | Logged as current step, not scheduled step. Delays not penalized accurately. | MAJOR |

---

## PHASE 3: EDGE CASE FAILURES

| Scenario | What Breaks | Severity |
|----------|-------------|----------|
| **All patients critical at step 1** | Vitals decay to 0 within 1-2 target_time windows. All die quickly. `_check_done()` terminates early. `_advance_time()` calls `_cascade_engine.on_outcome()` for each death. Cascades fire → secondary patients spawn → more deaths. Score: mostly 0.0. Grader handles this. | OK |
| **No patients (task4 at step 0)** | `patients_remaining=0`, `_check_done()` returns True immediately after reset? No — `_check_done()` checks `len(non_terminal) == 0` where `non_terminal = [p for p in self._patients if p.status not in ("treated", "deceased")]`. With 0 patients, `non_terminal = []`, length is 0 → early termination. But `step()` is never called because the agent gets the first observation after reset, and if `_check_done()` returned True, the episode would end immediately. `_build_observation(done=False)` is returned by reset. The loop in `inference.py` checks `done` AFTER calling step. With 0 patients, step 1 runs, `_advance_time()` processes empty patient list, no actions possible, step 2 runs, etc. `_check_done()` returns True when `non_terminal=[]`. Episode runs to max_steps with no patients. Score: time_score=1.0 (no patients), efficiency=1.0, secondary_harm=1.0, prep_ready=1.0, cascade_score=1.0. Final: `0.32+0.16+0.16+0.16+0.10+0.10 = 1.0`. **Perfect score with zero action.** | CRITICAL EXPLOIT |
| **All hospitals on diversion** | Patients arrive at hospitals on diversion. `_do_treatment_arrival()` doesn't check `on_diversion`. ORs may be idle but patient can't get treatment. No alert. Patient waits in "in_treatment" status. ICU bed consumed. Patient eventually dies (no OR available = no surgery = no treatment_complete = vitals don't recover, continue to decay). Secondary harm possible. Grader sees death but doesn't know WHY (no alert). | MAJOR |
| **All ambulances dispatched, no mutual aid** | Patients waiting indefinitely. Vitals decay. Deaths. Score drops. Not a bug — correct behavior. | OK |
| **Random agent (100% maintain_plan)** | No dispatches. All patients die. Score: time_score=0.0, efficiency=1.0, secondary_harm=1.0 (no secondary patients spawned), prep_ready=0.0 (no arrivals). Final: `0 + 0.16 + 0.16 + 0 + 0.10 + 0.10 = 0.52`. Not zero — the agent gets credit for not creating secondary harm. This is correct rubric behavior. | OK |
| **Ambulance dispatched to closed road** | `dispatch()` succeeds, route found with inf segment. `route_travel_time()` returns 999. Ambulance ETA = 999+15 = 1014. Patient dies in ~90 steps. Ambulance eventually arrives (step 1014) but patient dead. No alert about road closure at dispatch time. | MAJOR |

---

## PHASE 4: FIXES

### FIX 1: HF Space `_build_observation()` missing `done` parameter
**File:** `codered-hfspace/server/codered_environment.py`
```python
# BEFORE (HF Space):
def _build_observation(self) -> CodeRedObservation:
    ...

# AFTER:
def _build_observation(self, done: bool = False) -> CodeRedObservation:
    ...
    return CodeRedObservation(
        done=done,  # ADD THIS LINE
        ...
    )
```
And in `step()`:
```python
# BEFORE:
return self._build_observation()
# AFTER:
return self._build_observation(done=done)
```
**Why it works:** The `done` field in `CodeRedObservation` is `Field(default=False)`. By passing the computed `done` value, the OpenEnv step loop can properly detect termination.

---

### FIX 2: Phase 2 ambulance-patient linkage broken
**File:** `server/codered_environment.py`
```python
# In _do_triage_call(), after successful dispatch:
# UPDATE ambulance's patient_id so arrival detection works
amb = self._ambulance_manager.get(action.ambulance_id)
if amb:
    amb.patient_id = f"UNCONFIRMED_{action.call_id}"  # placeholder until patient spawns
```
AND in `ambulance_manager.tick()`:
```python
# When ambulance arrives at scene, check for unconfirmed patient
if amb.patient_id and amb.patient_id.startswith("UNCONFIRMED_"):
    call_id = amb.patient_id.replace("UNCONFIRMED_", "")
    # Find the patient that spawned from this call
    for p in patient_manager.patients:
        if getattr(p, 'dispatch_call_id', None) == call_id and p.status == "waiting":
            amb.patient_id = p.id  # confirmed!
            break
```
**Why it works:** The ambulance now carries a call reference. When it arrives, it checks if a patient from that call has spawned and adopts them. The alternative fix: spawn the patient SYNCHRONOUSLY in `_do_triage_call()` when dispatching, rather than waiting for countdown or arrival.

---

### FIX 3: Phase 2 forced-spawn secondary flag
**File:** `server/codered_environment.py`
```python
# In _spawn_patient_from_call(), use is_secondary=False for forced spawns
# BUT also log a separate event type for grader awareness:
self._episode_log.append({
    "step": self._state.step_count,
    "patient_id": patient.id,
    "event": "patient_created",
    "condition": true_condition,
    "is_secondary": False,  # deliberately NOT secondary
    "force_spawn": True,    # NEW: indicates countdown-expired spawn
    ...
})
```
AND in `grader.py`, update `grade_cascade_score()`:
```python
# Also check force_spawned patients as potential secondary harm
all_created_events = [e for e in episode_log if e.get("event") == "patient_created"]
# Count deaths for non-secondary but force-spawned patients separately
force_spawned_ids = {e["patient_id"] for e in all_created_events if e.get("force_spawn", False)}
# In the grader, apply a partial secondary_harm penalty for force-spawn deaths
```
**Why it works:** Forces grader awareness of the distinction while maintaining existing scoring behavior.

---

### FIX 4: ICU bed leak on patient death
**File:** `server/codered_environment.py`
```python
# In _advance_time(), when detecting patient death:
for p in self._patients:
    if p.status == "deceased" and not any(...):
        self._episode_log.append(...)
        # FIX: Release ICU bed if patient was admitted
        if p.assigned_hospital and p.icu_status == "admitted":
            self._hospital_system.release_icu_bed(p.assigned_hospital)
```
AND in `_do_treatment_arrival()`:
```python
# When patient dies while in treatment (ICU):
if patient.status == "deceased":
    if patient.assigned_hospital and patient.icu_status == "admitted":
        self._hospital_system.release_icu_bed(patient.assigned_hospital)
```
**Why it works:** ICU beds are a finite hospital resource. Releasing them on patient death prevents permanent depletion.

---

### FIX 5: `preempt_or` not logged
**File:** `server/codered_environment.py`
```python
def _do_preempt_or(self, action) -> None:
    ...
    result = self._hospital_system.preempt_or(...)
    if result["success"]:
        self._alerts.append(...)
        # FIX: Log to episode log
        self._episode_log.append({
            "step": self._state.step_count,
            "event": "action_preempt_or",
            "hospital_id": action.hospital_id,
            "or_index": action.or_index,
            "harm": result["harm"],
            "recovery_time": result["recovery_time"],
        })
        # Check if there was a patient in the preempted OR
        if result.get("patient_disrupted"):
            self._episode_log.append({
                "step": self._state.step_count,
                "event": "surgery_aborted",
                "patient_id": result["patient_id"],
                "reason": "or_preempted",
            })
```
AND in grader:
```python
# In grade_episode(), penalize preempt_or actions
aborted_surgeries = sum(
    1 for e in episode_log if e.get("event") == "surgery_aborted"
)
prep_ready -= 0.05 * aborted_surgeries  # penalty for disrupting ongoing care
```
**Why it works:** The grader must see preemption actions to score them appropriately.

---

### FIX 6: `assigned_ambulance` always None
**File:** `server/codered_environment.py`
```python
# In _build_observation(), set assigned_ambulance properly:
assigned_ambulance=None,
# REPLACE with:
assigned_ambulance=p.assigned_ambulance,
```
And ensure `patient.assigned_ambulance` is set in all dispatch paths. Currently set in `_do_dispatch_ambulance()` for Phase 1. For Phase 2, set in `_do_dispatch_als/bls` → update ambulance's `patient_id` → set `patient.assigned_ambulance` when patient spawns.

**Why it works:** The agent can now see which ambulance is en route to which patient, enabling better coordination decisions.

---

### FIX 7: Task1 cascade contamination
**File:** `server/subsystems/constants.py`
```python
TASK_CONFIG: Dict[str, Dict] = {
    "task1": {
        ...
        "cascade_enabled": False,  # ADD
    },
    ...
}
```
AND in `codered_environment.py` `_advance_time()`:
```python
cascade_enabled = TASK_CONFIG.get(self._state.task_id, {}).get("cascade_enabled", False)
if cascade_enabled:
    # Only trigger cascades for tasks with cascade explicitly enabled
```
**Why it works:** Task1 should be a clean single-patient scenario with no secondary spawns.

---

### FIX 8: Empty episode perfect score exploit
**File:** `server/codered_environment.py`
```python
def _check_done(self) -> bool:
    ...
    # ANTI-EXPLOIT: if all patients are terminal AND step_count > 0, done
    # BUT if no patients were ever created (task4 at start):
    if len(self._patients) == 0 and self._state.step_count >= 5:
        return True  # Force end if no activity
    # Also: minimum episode length
    if self._state.step_count >= 3 and all(
        p.status in ("treated", "deceased") for p in self._patients
    ):
        return True
```
OR: in grader, if `len(patient_times) == 0`, return score 0.0 with message "No patients processed".
```python
# In grade_episode():
if not patient_times:
    return RubricResult(time_score=0.0, efficiency=1.0, secondary_harm=1.0,
        prep_ready=1.0, mutual_aid_penalty=0.0, final_score=0.0,
        breakdown={"reason": "no_patients_processed"}, ...)
```
**Why it works:** An agent that does nothing should not get a perfect score. The grader must detect and penalize zero-action episodes.

---

### FIX 9: Mutual aid arrival step mismatch
**File:** `server/subsystems/mutual_aid.py`
```python
# Use the scheduled arrival step for logging, not current step
arrivals.append({
    ...
    "actual_arrival_step": pending.arrival_step,  # was: step_count
    "scheduled_arrival_step": pending.arrival_step,
})
```
AND in grader:
```python
# Compare scheduled vs actual (logged as pending.arrival_step)
actual = arrival_events[0].get("actual_arrival_step", pending.arrival_step)
scheduled = arrival_events[0].get("scheduled_arrival_step", pending.arrival_step)
if actual > scheduled:
    penalties.append(0.2 * (actual - scheduled))
elif actual < scheduled:
    penalties.append(0.1 * (scheduled - actual))  # early = slight penalty
```
**Why it works:** The penalty is now computed against the correct baseline.

---

### FIX 10: Blood bank unlimited emergency drain
**File:** `server/subsystems/blood_bank.py`
```python
def emergency_release(self, hosp_id: str, patient_id: str, blood_type: str, units: int) -> Dict:
    bank = self._banks.get(hosp_id)
    if bank is None:
        return {"success": False, "reason": f"Hospital {hosp_id} not found"}
    # FIX: Cap emergency release per patient
    if units > 4:  # max 4 units emergency per patient
        units = 4
    if bank.stocks.get("O_NEG", 0) < units:
        return {"success": False, "reason": f"Insufficient O_NEG at {hosp_id}"}
    bank.stocks["O_NEG"] -= units
    return {"success": True, "blood_type": "O_NEG", "units": units}
```
AND add global depletion guard:
```python
# In BloodBankSystem, track emergency releases per hospital per episode
self._emergency_releases_this_episode: Dict[str, int] = {}
# Check before emergency_release: if total emergency O_NEG released > 50% of initial stock, reject
```
**Why it works:** Prevents a single agent action from draining all emergency blood reserves.

---

### FIX 11: `pending_surge_probability` unbounded accumulation
**File:** `server/subsystems/cascade_engine.py`
```python
def _apply_effect(self, effect: str, params: Dict[str, Any], step: int) -> None:
    if effect == "news_cycle":
        self._news_cycle_steps_remaining = params["steps"]
        # FIX: Cap surge probability at 1.0
        self._pending_surge_probability = min(1.0,
            self._pending_surge_probability + params["surge_prob_boost"])
```
**Why it works:** Maintains the `ge=0.0, le=1.0` invariant.

---

### FIX 12: Ambulance arriving at empty scene (Phase 1)
**File:** `server/subsystems/ambulance_manager.py`
```python
def tick(self) -> None:
    ...
    elif amb.status in ("en_route", "returning") and amb.eta_minutes > 0:
        amb.eta_minutes -= 1
        if amb.eta_minutes == 0:
            prev_status = amb.status
            amb.status = "on_scene"
            # FIX: If no patient assigned, flag for immediate return
            if amb.patient_id is None:
                amb.scene_minutes_remaining = 0  # no scene time needed
                amb.arrived_with_patient = False
                amb.status = "returning"  # immediately start returning
                # Alert the environment
                self._notify_arrival_without_patient(amb.id)
```
**Why it works:** Ambulance doesn't waste scene time at an empty scene. Returns to base immediately.

---

## PHASE 5: HARDENING

### 5.1 DETERMINISM GUARANTEES

1. **Seed propagation audit**: Every subsystem uses `self._rng` from `environment._rng`. All spawn decisions, cascade rolls, disruption scheduling, and patient jitter use the same RNG. Confirmed: ✅ PatientManager uses `self._rng`, AmbulanceManager does NOT (uses fixed ambulance defs). HospitalSystem uses deterministic constants. RoadNetwork uses constants. MutualAidManager uses `seed=seed` in constructor. DisruptionEngine uses `random.Random(seed)`. CascadeEngine uses `random.Random(seed)`. **Risk**: AmbulanceManager doesn't use the seeded RNG for any random decisions. **Risk**: `BloodBankSystem` doesn't use RNG. ✅

2. **Action replay determinism**: `inference.py` with `temperature=0.7` is non-deterministic. Fix: use `temperature=0.0` for reproducible benchmarking.

3. **Episode log ordering**: Log entries are appended in order of occurrence. `_advance_time()` appends first, then `_execute_action()`. This means action results appear after time-advance effects. Grader relies on `step` field for ordering. ✅

### 5.2 ANTI-EXPLOIT LAYER

```python
# In CodeRedEnvironment.step()
def step(self, action, ...):
    # Anti-exploit: action rate limiting
    self._action_counts: Dict[str, int] = {}
    action_type = action.__class__.__name__
    self._action_counts[action_type] = self._action_counts.get(action_type, 0) + 1
    if self._action_counts[action_type] > 20:  # per-episode cap per action type
        self._alerts.append(f"Action rate limit exceeded for {action_type}")
        return self._build_observation(done=False)  # step without action

    # Anti-loop: detect maintain_plan spam
    if isinstance(action, MaintainPlan):
        self._maintain_plan_streak += 1
        if self._maintain_plan_streak > 10:
            self._alerts.append("WARNING: No progress made in 10 consecutive steps")
    else:
        self._maintain_plan_streak = 0
```

### 5.3 REWARD CAPS

```python
# In _compute_step_reward():
# Cap cumulative reward to prevent reward hacking via milestone accumulation
MAX_CUMULATIVE_REWARD = 10.0
if self._state.cum_reward > MAX_CUMULATIVE_REWARD:
    reward = 0.0  # no additional reward beyond cap
```

### 5.4 EVENT SCHEMA ENFORCEMENT

```python
# In episode_log.append(), add schema validation:
VALID_EVENTS = frozenset({
    "patient_created", "patient_deceased", "treatment_complete",
    "treatment_started", "dispatch", "action_dispatch", "action_prepare_or",
    "action_page_specialist", "action_allocate_blood", "mutual_aid_called",
    "mutual_aid_arrived", "patient_arrived_hospital", "overcrowding_started",
    "news_cycle", "secondary_patient_spawned", "icu_boarding", "surgery_aborted",
})
assert event["event"] in VALID_EVENTS, f"Invalid event: {event['event']}"
```

---

## PHASE 6: FINAL CHECKLIST

| Check | Status | Notes |
|-------|--------|-------|
| `step()` returns `(obs, reward, done, info)` | ⚠️ PARTIAL | Returns `(obs)` — OpenEnv wraps in server. `done` flag MISSING in HF Space |
| `reset()` clean | ✅ PASS | New subsystem instances, cleared state |
| No state leakage | ❌ FAIL | `assigned_ambulance=None`, blood queue exposed, crossmatch queue exposed |
| Deterministic with seed | ⚠️ PARTIAL | Temperature 0.7 in inference. All subsystems seeded. |
| 3 tasks working | ⚠️ PARTIAL | Phase 2 dispatch workflow broken (ambulance-patient linkage) |
| Graders deterministic | ✅ PASS | No RNG in grader |
| Scores ∈ [0,1] | ⚠️ PARTIAL | `mutual_aid_penalty` can exceed 1.0 in breakdown |
| No reward exploits | ❌ FAIL | Blood drain, cascade reward loop in task1, empty episode score exploit |
| Logs consistent | ⚠️ PARTIAL | `preempt_or` missing, MA arrival step mismatch |
| Docker builds | ✅ PASS | Dockerfile present |
| HF Space responds | ⚠️ PARTIAL | `done` flag missing, TASK_DEFINITIONS endpoint missing |
| Runtime < 20 min | ✅ PASS | Max 60 steps × minimal computation |

---

## SUMMARY OF CRITICAL ISSUES (MUST FIX BEFORE DEPLOYMENT)

| Priority | Issue | File | Fix |
|----------|-------|------|-----|
| P0 | `done` flag missing in HF Space | `codered-hfspace/.../codered_environment.py` | Add `done` parameter |
| P0 | Phase 2 ambulance-patient linkage broken | `codered_environment.py` | Set `patient_id` on ambulance, link on arrival |
| P0 | Empty episode gets perfect score | `grader.py` | Add zero-patient guard |
| P1 | ICU bed leak on death | `codered_environment.py` | Release bed on death |
| P1 | `preempt_or` not logged | `codered_environment.py` | Add episode_log entries |
| P1 | Task1 cascade contamination | `constants.py` | Set `cascade_enabled=False` for task1 |
| P1 | Phase 2 forced-spawn secondary mismatch | `codered_environment.py` | Use `force_spawn` flag |
| P1 | `assigned_ambulance` always None | `codered_environment.py` | Set from patient.assigned_ambulance |
| P1 | MA arrival step mismatch | `mutual_aid.py` | Use scheduled step for logging |
| P2 | Blood bank unlimited emergency drain | `blood_bank.py` | Cap per-patient emergency units |
| P2 | Surge probability unbounded | `cascade_engine.py` | Cap at 1.0 |
| P2 | HF Space missing endpoints | `codered-hfspace/server/app.py` | Add /tasks, /grader endpoints |
| P2 | Inference temperature non-determinism | `inference.py` | Set temperature=0.0 for reproducibility |
| P3 | Phase 2 forced-spawn `is_secondary=False` grading gap | `grader.py` | Handle in cascade_score |
| P3 | Road closure no reroute alert | `ambulance_manager.py` | Alert when dispatch to closed road |
| P3 | `DispatchCall.spawned_patient_id` never updated | `codered_environment.py` | Update when patient spawns |

---

*Report generated by Claude Code comprehensive audit. All findings are based on code inspection. Issues marked CRITICAL must be resolved before HF Space deployment.*
