# Teaching Mesa agents to stop watching the clock

**Google Summer of Code 2026 · [Project Mesa](https://github.com/mesa/mesa) · Aman Bihari ([@codebreaker32](https://github.com/codebreaker32))**

Mesa is the agent-based modelling framework for Python. I spent this summer adding a way for an agent's state to change *continuously* — and for the model to be told the exact moment that state crosses a line, instead of checking every tick to see whether it has yet.

Ten pull requests merged, one still in review. The core of it is a new module, `mesa.experimental.states`, which is about 490 lines of implementation and 440 of tests. Everything below links to the actual PR, so you can check any of it.

---

## The thing that was wrong

Mesa 4 is event-driven underneath. There's no step counter; `Model.time` is a float, and everything — including your own `step()` — runs off a single event list. That's a genuinely nice engine.

The problem is that nothing exposed it to the person writing the model. Here's `Animal.step` from Mesa's own wolf–sheep example. This is on `main` right now:

```python
def step(self):
    self.move()

    self.energy -= 1          # decay, by hand, once per tick

    self.feed()

    if self.energy < 0:       # threshold, re-checked, once per tick
        self.remove()
    elif self.random.random() < self.p_reproduce:
        self.spawn_offspring()
```

Two lines there are the whole problem.

The decay is arithmetic you have to remember to write, and it's only correct because the tick happens to be `1.0`. Change the timestep and it silently means something different.

The death check is a poll. It runs for every animal on every tick, and it can only ever notice starvation *at a tick boundary* — never at the moment it actually happened. Mesa's engine could have scheduled that death exactly. Nothing let you say so.

Before proposing anything I built a baseline model in stock Mesa — a needs-based homeostatic agent, continuous decay, priority-driven decisions, spatial foraging — specifically to write down where it hurt, and posted that as [discussion #3721](https://github.com/mesa/mesa/discussions/3721). Jan Kwakkel replied with the framing the whole project ended up hanging off:

> Event scheduling … provides a locality of time. Activity scanning … provides a locality of state. Process interaction … provides a locality of object. Mesa 4 is event scheduling based, defaulting to incremental time advancement.

That's the diagnosis, and it's sharper than anything I'd written. Mesa's *engine* is event scheduling. Its *modelling API* made you write activity scanning on top of it.

To be clear about credit: I didn't spot this first. Ewout ter Hoeven had opened discussions about [Tasks](https://github.com/mesa/mesa/discussions/2526), [Continuous States](https://github.com/mesa/mesa/discussions/2529) and [a Behavioral Framework](https://github.com/mesa/mesa/discussions/2538) back in December 2024. What didn't exist was evidence — a concrete account of where a real model breaks, and in what order to attack it.

---

## What I built

Two pieces, meeting at the event list Mesa already had.

**`ContinuousState`** stores a value as a *trajectory* rather than a number: a baseline, a timestamp, and a rate. Reading the attribute extrapolates to the current model time, so it's exact whenever you ask, not just on tick boundaries. Rates can be callables, and they chain — `position' = speed`, `speed' = acceleration` — in which case the extrapolation picks up the second-order term and goes piecewise quadratic.

**`Threshold`** turns that trajectory into a scheduled event. Because the trajectory is known in closed form, the moment it reaches a limit can be *solved for* instead of watched for. Each threshold keeps exactly one live event on the heap and re-solves itself whenever the trajectory or the limit changes.

The whole tram example agent looks like this. There is no `step()`:

```python
class Tram(Agent, HasEmitters):
    acceleration = Observable(fallback_value=0.0)
    speed        = ContinuousState(fallback_value=0.0, rate=lambda a: a.acceleration)
    position     = ContinuousState(fallback_value=0.0, rate=lambda a: a.speed)   # chained
    brake_point  = Observable(fallback_value=float("inf"))
    cruise_speed = Observable(fallback_value=15.0)

    _cruise_threshold  = Threshold(state=speed,    limit=cruise_speed,
                                   callback="start_coasting",   direction="rising")
    _braking_threshold = Threshold(state=position, limit=brake_point,
                                   callback="brake",            direction="rising")
    _stop_threshold    = Threshold(state=speed,    limit=0.0,
                                   callback="arrive_at_station", direction="falling")

    def depart(self):
        self.brake_point = self.next_station - self.braking_distance()
        self.acceleration = self.acceleration_rate

    def start_coasting(self):
        self.acceleration = 0.0

    def brake(self):
        self.brake_point = float("inf")
        self.acceleration = -self.deceleration_rate
```

`brake_point` is an `Observable`, not a constant, so assigning to it re-solves the threshold. There's no `rearm()` call — that's the point, and I'll come back to why.

Here's what it actually does, from running the merged example ([PR #3796](https://github.com/mesa/mesa/pull/3796)):

![Tram speed and position over 75 seconds, with the exact threshold crossing times marked](img-1-tram-trajectory.png)

The tram coasts at 7.5 s, brakes at 14.583333 s, arrives at 19.583333 s. That last one is 175/12 — a time no integer tick would ever land on. It stops at exactly 200.0000 m, dwells for exactly 5 s, and does it again.

## Why "exact" is worth caring about

I wrote a plain forward-Euler loop that polls its thresholds every tick — the code this API replaces — and ran it on the same segment. That loop lives in [`make_figures.py`](make_figures.py) in this repo; it isn't Mesa and it isn't a Mesa benchmark. It's there so the numbers are checkable.

![Zoom on the arrival: every polled tick overshoots the platform, the analytic solve lands on it](img-2-polling-vs-exact.png)

| Method | Arrival detected | Time error | Stops at | Position error |
|---|---:|---:|---:|---:|
| Polling, 1.0 s tick | 21.000 s | +1.417 s | 219.0000 m | **+19.00 m** |
| Polling, 0.5 s tick | 20.000 s | +0.417 s | 206.2500 m | +6.25 m |
| Polling, 0.25 s tick | 19.750 s | +0.167 s | 202.5000 m | +2.50 m |
| Polling, 0.1 s tick | 19.700 s | +0.117 s | 202.6700 m | +2.67 m |
| `ContinuousState` + `Threshold` | 19.583333 s | — | 200.0000 m | **0.00 m** |

At a one-second tick the tram sails 19 metres past the platform. Shrinking the tick costs linearly more work and never reaches zero error — and notice the 0.1 s row is *worse* than 0.25 s, because error accumulation isn't monotonic in step size. That's the kind of thing that's very annoying to discover in a model you've already published results from.

---

## The action layer

**This part is built on someone else's work and I want that stated up front.** The `Action` primitive — the lifecycle, `start_action`, `interrupt_for`, `cancel_action`, the `on_start`/`on_complete`/`on_interrupt` hooks — is Ewout ter Hoeven's, from [PR #3461](https://github.com/mesa/mesa/pull/3461) (+1388 lines). `git blame` on `actions.py` puts 334 lines with him and 127 with me.

What I added, in three stacked PRs tracked by [issue #3798](https://github.com/mesa/mesa/issues/3798):

**Actions can fail now.** ([#3801](https://github.com/mesa/mesa/pull/3801)) Before this, an action could only succeed. A sheep starts foraging, the completion event is scheduled immediately, and five time units later `on_complete()` fires and the sheep gains energy whether or not the grass is still there. The workaround was re-checking inside `on_complete()` and returning early — the same guard copy-pasted into every action, and the action still ends up `COMPLETED` having accomplished nothing.

So: two predicate lists and a fifth state. `start_requirements` gates entry, checked *before* duration and priority resolve, so a failing action fires no `on_start()` and schedules no completion event. `completion_requirements` gates the effect at completion time. A failure moves to `ActionState.FAILED` and calls `on_fail()`.

They're two separate lists rather than one list checked twice because they answer different questions. A rival resource should be *claimed* at the start — and what the agent does when the claim fails (take what's left, go elsewhere, wait) is behaviour, not a boolean. The completion list is for things nobody can reserve: a market still open when a trade lands, a counterparty still solvent at settlement.

Neither is re-checked in between, deliberately. Continuously revalidating every active action means rescanning the world on every change, which is activity scanning wearing a different hat. An action that must abort the instant a condition breaks belongs to a `Threshold` calling `cancel_action()`.

**`Action.priority` does something now.** ([#3805](https://github.com/mesa/mesa/pull/3805)) It was resolved at `start()` and read by nothing. `git grep` on `main` found two occurrences, both writes. A priority-1 action would happily preempt a priority-100 one, and the release notes advertised the feature. I added `Agent.should_interrupt(current, incoming)`, defaulting to `current.interruptible and incoming.priority >= current.priority`.

`>=` rather than `>` because priorities default to `0.0`, so a strict `>` would silently stop existing models from interrupting at all — it broke three tests that were already on `main`. Choosing `>=` made the new hook a no-op for every model written before it.

**Agents get told when they're free.** ([#3833](https://github.com/mesa/mesa/pull/3833) — **still open, not merged**) `Agent.on_idle(previous)` fires when an action ends and nothing replaces it. It's dispatched as a zero-delay `Priority.LOW` event rather than called synchronously, so a zero-duration action started from inside `on_idle` can't recurse, and every agent finishing at time *t* decides after all completions at *t* have run.

The action test suite went from 61 tests to 101 across the three.

---

## Paying for it

Solving crossings analytically means re-solving whenever a trajectory changes, and every re-solve cancels an event. Mesa cancels with tombstones — the dead event stays on the heap to preserve the heap invariant and gets discarded when it surfaces. So a threshold whose projected time keeps moving grows the heap without bound. On a cancel-and-reschedule workload with 2000 agents over 800 steps, the heap peaked at 239,731 entries, 99.2% of them tombstones.

A `compact()` method had existed since [#3359](https://github.com/mesa/mesa/pull/3359) (souro26's work, not mine) and a comment in `remove()` already described cancelled events as something that "may trigger adaptive compaction if they dominate the heap". Nothing in the library ever called it.

[#3800](https://github.com/mesa/mesa/pull/3800) wired it up:

![Peak heap and wall time, before and after adaptive compaction](img-3-compaction.png)

Peak memory at 1000 agents drops from 29.0 MB to 1.3 MB. The same counter makes `__len__` O(1) instead of a scan of the whole heap — 2.6 ms to 0.12 µs at 50k events — which `is_empty()` and `peek_ahead()` inherit for free.

The interesting design question was *where to put the check*. It hangs off `cancel()`, not `add_event` or `pop_event`, because cancelling is the only operation that *adds* dead weight — the only point at which a heap can newly go past the threshold. Putting it in `pop_event`, which is the obvious place and what the original sketch proposed, means paying for it on the hot path.

I'll be honest that the neat version of that argument is a bit too neat: popping a *live* event does raise the instantaneous ratio, since it shrinks the heap without removing a tombstone, so a heap can sit above the ratio between cancellations. What the cancel-side check actually guarantees is a bound on accumulated dead weight, not that the ratio is never momentarily exceeded. That's enough for the memory bound, which is what the workload needed.

The ratio itself was picked by measurement, not taste. I shipped 0.5 first; a reviewer asked for the evidence, so I ran a sweep on an 800-agent workload and posted the table in review. Peak heap is bounded by `L / (1 − ratio)`. 0.25 gave the best balance, and the same sweep showed a `COMPACTION_FLOOR` constant I'd added earned nothing above ~192 live events. So it came out — a tuning knob deleted rather than added.

**Where this doesn't help:** at 200 agents the timing difference is inside the noise; the win there is purely memory. And the whole approach loses when nearly every agent crosses a threshold on nearly every tick, because then the solver runs as often as the poll would have and the event machinery is pure overhead. Continuous state pays off when transitions are *rare relative to ticks* — transit, logistics, physiology. Not a lattice model where every cell updates every step.

---

## The part where I built the wrong thing

My first two attempts — [#3754](https://github.com/mesa/mesa/pull/3754) and [#3755](https://github.com/mesa/mesa/pull/3755), both closed unmerged — built a columnar NumPy backend: a `StateTensor` holding every agent's continuous state in pre-allocated arrays, plus a `ContinuousScheduler` keeping a single master event at the global-minimum crossing.

The motivation was real. I'd measured a Sugarscape model producing over 600,000 tombstone events by step 300 when deaths were threshold-driven. One master event makes that number one. And the CI benchmark bot was emphatic: Sugarscape run time down 78%.

![The CI benchmark on the abandoned PR: one green column, three red ones](img-4-false-start.png)

That's the same benchmark comment. I read the first bar and not the other three.

Every model paid for the tensor — it was allocated in every `Model.__init__` at a fixed capacity of 10,000 rows, so models that never touch a continuous state got 148% and 169% slower to construct. Past 10,000 agents it raised `MemoryError`. Ewout's summary was that it amounted to "a second runtime running alongside Mesa's existing one", and that the target should be "80% of the use-cases with 20% of the complexity". Jan pointed out that Sugarscape was the wrong model to design against in the first place — a tram simulation, where cancellation is rare, needs no tensor at all. I'd picked the workload that flattered my architecture.

I closed #3755 with one line — "closing in favour of #3766" — and rebuilt on top of `mesa_signals` and the existing event queue. The merged design allocates nothing per model; its benchmark run shows init times flat or slightly improved across the board.

The tombstone problem didn't go away by being ignored. It just stopped being an excuse for a parallel runtime, and got a targeted fix four weeks later in 78 lines of `events.py`.

**Two other things didn't ship as proposed.** My March prototype had a `DecisionSystem` — a declarative rules engine where you registered conditions and priorities and it chose the agent's next action. I didn't drop it; Jan rejected it, on the grounds that Mesa shouldn't force a finite-state-machine style of logic on modellers. Arguing it through, I came round: a rules engine that re-evaluates every agent's rules to decide what's next is activity scanning again, which is the thing the project was supposed to remove. I also withdrew `Task`/`TaskManager` in August, on the narrower ground that nothing meaningfully distinguished them from the `Action` class that already existed.

So two of my three proposed pillars didn't ship under their proposed names. That's a real scope reduction against what I pitched in March, and I'd rather say so than quietly redefine what was promised.

---

## The trail

| PR | What | Diff | State |
|---|---|---:|---|
| [#3482](https://github.com/mesa/mesa/pull/3482) | Diffing engine for dependency autodiscovery in `mesa_signals` | +32 / −17 | merged |
| [#3754](https://github.com/mesa/mesa/pull/3754) | First attempt: `ContinuousState` on a tensor backend | +567 / −4 | closed |
| [#3755](https://github.com/mesa/mesa/pull/3755) | Second attempt, with a dedicated continuous scheduler | +618 / −8 | closed |
| [#3764](https://github.com/mesa/mesa/pull/3764) | Extract `ComputedState.evaluate()` | +52 / −34 | merged |
| [#3774](https://github.com/mesa/mesa/pull/3774) | Add `HasEmitters.peek()` for non-reactive reads | +74 / −0 | merged |
| [#3785](https://github.com/mesa/mesa/pull/3785) | Add the `examples/experimental/` tier | +14 / −7 | merged |
| [#3788](https://github.com/mesa/mesa/pull/3788) | Generate Read the Docs pages for the new tier | +8 / −14 | merged |
| **[#3766](https://github.com/mesa/mesa/pull/3766)** | **`ContinuousState` and `Threshold`** | **+938 / −0** | **merged** |
| [#3796](https://github.com/mesa/mesa/pull/3796) | Tram route model — a worked example with no agent `step()` | +702 / −0 | merged |
| [#3800](https://github.com/mesa/mesa/pull/3800) | Adaptive event-list compaction and O(1) `__len__` | +293 / −2 | merged |
| [#3801](https://github.com/mesa/mesa/pull/3801) | Requirement lists and an `ActionState.FAILED` path | +376 / −8 | merged |
| [#3805](https://github.com/mesa/mesa/pull/3805) | `Agent.should_interrupt`, giving `Action.priority` its first effect | +181 / −21 | merged |
| [#3833](https://github.com/mesa/mesa/pull/3833) | `Agent.on_idle` as a deferred low-priority wake | +263 / −22 | **open, awaiting review** |

Three of those touch stable modules (`mesa/time/events.py`, `mesa/agent.py`); the rest are under `mesa/experimental/`, which carries no semver guarantee. The defaults in the stable ones were chosen so no model written before this work behaves differently after it.

Two enabling PRs came out of review pressure and are worth a note, because neither was in my plan. The branch originally reached into `model._time` privately, and a reviewer asked why it couldn't read the public attribute. The answer turned out to be the interesting bit: `model.time` is itself an `Observable`, so reading it inside a rate evaluation registers the clock as a dependency of *every* trajectory — and then every tick re-snapshots every state, shattering the parabolas into microscopic linear segments. The fix needed a way to read an `Observable` without subscribing to it, which became `peek()` in #3774. Ten lines of implementation, sixty-three of test, and the main PR couldn't land cleanly without it.

The bigger change also came from review. The example originally read `Tram._brake_point.set_limit(self, brake_at); Tram._cruise.rearm(self)` and Jan asked, reasonably, what was going on there. Letting `limit` accept an `Observable` and having `bind()` subscribe to *its* `CHANGED` signal deleted `set_limit()`, `rearm()` and three more methods, and collapsed those three calls into one assignment. It also deleted the entire class of bug where a model forgets to re-arm and the threshold silently never fires again. Smaller API, fewer ways to hold it wrong — and it wasn't my idea first.

Full history, if you want to check the rest: [64 PRs opened, 48 merged](https://github.com/mesa/mesa/pulls?q=is%3Apr+author%3Acodebreaker32) since December 2025. Before the coding period I'd also [reviewed 17 PRs](https://github.com/mesa/mesa/pulls?q=is%3Apr+reviewed-by%3Acodebreaker32+-author%3Acodebreaker32) by other people and opened 14 issues.

---

## What you actually get from `main` today

Separate from the story, because these are different questions:

- `from mesa.experimental.states import ContinuousState, Threshold` works. 24 tests, experimental namespace, no semver guarantee.
- `Action` takes `start_requirements` and `completion_requirements`, has `ActionState.FAILED` and `on_fail()`, and `Agent.should_interrupt` gates every preemption. **`Agent.on_idle` is not there yet** — that's #3833.
- `EventList` compacts itself and `len()` on it is O(1). This is in stable `mesa/time/`, so it applies to every model that cancels events, not just mine.
- `solara run mesa/examples/experimental/tram_model/app.py` runs the worked example.
- **No narrative documentation.** The docstrings are thorough. The guide pages aren't written — that's section 5 of #3798 and it's still open.

Other things I'd flag as unfinished: `_find_chained_rate` looks exactly one link deep, so a three-deep chain will under-integrate rather than raise, and no test covers that. Whether an agent should have exactly one action slot, and whether the action cluster belongs on core `Agent` or behind an experimental mixin, are both recorded as open questions rather than answered. And there's no general ODE support — that's the natural next piece, and it would cost `Threshold` its exact analytic crossing time in exchange for bisection on an integrator's output.

---

## What I'd tell myself in June

**Read the whole benchmark, not the column you were hoping for.** The 78% win was real and so was the 148% regression, and they were in the same CI comment. A performance claim isn't a number, it's a distribution over workloads — including the workloads that don't use your code at all.

**Pick the model that will hurt you.** I designed against Sugarscape because it made the architecture look necessary. The model you validate against silently encodes an assumption about who your users are.

**Declarative beats imperative when the alternative is an API someone can forget to call.** `set_limit()` and `rearm()` weren't wrong exactly. They were just a way to have a bug.

**Stack small PRs.** #3766 took 41 days and 24 inline comments. The action work — comparable code, split into three PRs with one idea each, stacked, tracked by a public issue — went from opened to merged in under two weeks apiece.

**Writing the design down first isn't overhead.** The baseline evaluation cost about a week and produced the one sentence the whole project turned on. I wouldn't have found that framing by writing code.

---

## Thanks

**[Jan Kwakkel](https://github.com/quaquel)** reviewed nearly all of this, and two of his comments changed the project rather than the code: naming the three simulation world-views turned my list of complaints into a diagnosis, and refusing the `DecisionSystem` kept a rules engine out of a framework that shouldn't have one. He also asked for the evidence behind a magic constant, which is how `COMPACTION_RATIO` ended up measured instead of guessed.

**[Ewout ter Hoeven](https://github.com/EwoutH)** opened the discussions this grew out of and wrote the `Action` primitive I built on. He's also the reason the tensor backend died at two weeks instead of two months.

**[Jackie Kazil](https://github.com/jackiekazil)** and **[Tom Pike](https://github.com/tpike3)** reviewed and merged much of the surrounding work, and pushed back on the examples reorganisation until the docs actually built.

---

*Every figure here is generated by [`make_figures.py`](make_figures.py) in this repo — figures 1 and 2 by running the merged tram example and instrumenting the threshold callbacks, figure 3 from the numbers reported in #3800, figure 4 from the CI benchmark bot on #3754. The benchmark figures are my own measurements on one machine; treat them as the author's numbers on a stated workload, not an independent audit.*
