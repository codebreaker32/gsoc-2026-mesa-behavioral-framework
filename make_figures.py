#!/usr/bin/env python3
"""Render the four figures used in the GSoC write-up gist.

Run from a checkout of mesa/mesa with the tram example present:

    python tools/make_gist_figures.py

Figures 1 and 2 come from actually running
``mesa.examples.experimental.tram_model`` and reading the agent's states, so
they cannot drift from the merged code. Figure 3 plots the numbers reported in
PR #3800. Figure 4 plots the CI benchmark bot's output on the abandoned PR
#3754.

Output goes to ``gist/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent

PAPER = "#fdfcfa"
INK = "#14161a"
INK2 = "#4a4f57"
INK3 = "#6b7179"
ACCENT = "#ab3a17"
TEAL = "#1b5b68"
GREY = "#9a958a"
RULE = "#ddd9d0"

ACCEL, DECEL, CRUISE, SPACING, N_STATIONS, DWELL = 2.0, 3.0, 15.0, 200.0, 4, 5.0

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.facecolor": PAPER,
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.edgecolor": RULE,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": INK3,
    "ytick.color": INK3,
    "axes.titlecolor": INK,
    "axes.grid": True,
    "grid.color": "#eceae3",
    "grid.linewidth": 0.9,
    "axes.axisbelow": True,
    "figure.dpi": 130,
})


def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3)


# ---------------------------------------------------------------- real model
def run_tram():
    """Run the merged tram example; return samples and exact crossing times."""
    from mesa.examples.experimental.tram_model.agents import Tram
    from mesa.examples.experimental.tram_model.model import TramScenario, TransitSystem

    fired = []
    for name in ("start_coasting", "brake", "arrive_at_station", "depart"):
        original = getattr(Tram, name)

        def wrap(hook=name, orig=original):
            def wrapper(self):
                fired.append({"t": float(self.model.time), "event": hook,
                              "position": float(self.position), "speed": float(self.speed)})
                return orig(self)
            return wrapper

        setattr(Tram, name, wrap())

    model = TransitSystem(scenario=TramScenario(
        n_stations=N_STATIONS, station_spacing=SPACING, cruise_speed=CRUISE,
        acceleration_rate=ACCEL, deceleration_rate=DECEL, dwell_time=DWELL))
    tram = model.tram

    ts, ps, vs = [0.0], [0.0], [0.0]
    t = 0.05
    while t <= 75.0:
        model.run_until(t)
        ts.append(model.time)
        ps.append(float(tram.position))
        vs.append(float(tram.speed))
        t += 0.05
    return np.array(ts), np.array(ps), np.array(vs), fired


def euler(dt):
    """The polling loop the new API replaces. Written here, not part of Mesa."""
    brake_point = SPACING - CRUISE**2 / (2 * DECEL)
    x = v = t = 0.0
    acc, braking = ACCEL, False
    xs, ts = [0.0], [0.0]
    while t < 60:
        if not braking and v >= CRUISE and acc > 0:
            acc = 0.0
        if not braking and x >= brake_point:
            acc, braking = -DECEL, True
        if braking and v <= 0:
            break
        x += v * dt
        v += acc * dt
        t += dt
        xs.append(x)
        ts.append(t)
    return np.array(ts), np.array(xs), t, x


def analytic():
    t_acc = CRUISE / ACCEL
    d_acc = 0.5 * ACCEL * t_acc**2
    bp = SPACING - CRUISE**2 / (2 * DECEL)
    t_brake = t_acc + (bp - d_acc) / CRUISE
    return t_acc, t_brake, t_brake + CRUISE / DECEL, bp


# --------------------------------------------------------------- figure one
def fig_trajectory(ts, ps, vs, fired):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1], "hspace": 0.16})

    events = [e for e in fired if e["event"] != "depart" and e["t"] <= 75]
    for e in events:
        for ax in (ax1, ax2):
            ax.axvline(e["t"], color=RULE, lw=1, ls=(0, (2, 4)), zorder=1)

    ax1.plot(ts, vs, color=ACCENT, lw=2.1, solid_capstyle="round", zorder=3)
    ax1.scatter([e["t"] for e in events], [e["speed"] for e in events],
                s=26, color=ACCENT, edgecolor=PAPER, lw=1.4, zorder=4)
    ax1.set_ylabel("speed  (m/s)")
    ax1.set_ylim(-1.4, 18.6)
    ax1.set_yticks([0, 5, 10, 15])

    ax2.plot(ts, ps, color=TEAL, lw=2.1, solid_capstyle="round", zorder=3)
    for i in range(N_STATIONS):
        ax2.axhline(SPACING * i, color="#e6e2d8", lw=1, zorder=0)
    ax2.set_ylabel("position  (m)")
    ax2.set_xlabel("model time  (s)")
    ax2.set_ylim(-30, 660)
    ax2.set_yticks([0, 200, 400, 600])
    ax2.set_xlim(-1.5, 76.5)

    first = events[:3]
    label = {"start_coasting": "coast", "brake": "brake", "arrive_at_station": "arrive"}
    note = "   ".join(f"{label[e['event']]} {e['t']:.6f} s" for e in first)
    ax1.set_title("One tram, three stops. Every transition is a solved threshold crossing.",
                  loc="left", fontsize=11.5, fontweight="bold", pad=16)
    ax1.text(0, 1.015, note, transform=ax1.transAxes, fontsize=9,
             family="DejaVu Sans Mono", color=ACCENT)

    ax2.annotate("stops at exactly 200.0000 m,\nnot 200-and-a-bit",
                 xy=(19.583, 200), xytext=(26, 78), fontsize=8.8, color=INK2,
                 arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.9,
                             "connectionstyle": "arc3,rad=-0.2"})
    ax2.annotate("5 s dwell", xy=(22, 207), xytext=(20.5, 288), fontsize=8.8, color=INK3,
                 arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.9})

    for ax in (ax1, ax2):
        tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "img-1-tram-trajectory.png", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- figure two
def fig_polling():
    t_acc, t_brake, t_stop, bp = analytic()
    fig, ax = plt.subplots(figsize=(9, 4.2))

    ax.axhline(SPACING, color=ACCENT, lw=1.3, ls=(0, (5, 4)), zorder=2)
    ax.text(22.3, 200.9, "the platform, 200 m", color=ACCENT, fontsize=9)

    shades = ["#b5b0a3", "#a49f92", "#938e82", "#827d72"]
    # tick 0.25 and tick 0.1 stop 0.17 m apart, so their labels are pushed
    # apart by hand and given leader lines rather than sitting on the points.
    dy = {1.0: 0, 0.5: 0, 0.25: -11, 0.1: 11}
    for i, dt in enumerate((1.0, 0.5, 0.25, 0.1)):
        ts, xs, t_end, x_end = euler(dt)
        m = ts >= 17.0
        ax.plot(ts[m], xs[m], color=shades[i], lw=1.5, zorder=3)
        ax.scatter([t_end], [x_end], s=26, color=shades[i], edgecolor=PAPER, lw=1.3, zorder=4)
        ax.annotate(f"tick {dt} s   +{x_end - SPACING:.2f} m",
                    xy=(t_end, x_end), xytext=(9, dy[dt]), textcoords="offset points",
                    fontsize=9, color=shades[i], va="center",
                    family="DejaVu Sans Mono",
                    arrowprops=None if dy[dt] == 0 else
                    {"arrowstyle": "-", "color": shades[i], "lw": 0.7,
                     "shrinkA": 0, "shrinkB": 3})

    tt = np.linspace(17.0, t_stop, 400)
    xx = np.where(tt <= t_brake,
                  0.5 * ACCEL * t_acc**2 + CRUISE * (tt - t_acc),
                  bp + CRUISE * (tt - t_brake) - 0.5 * DECEL * (tt - t_brake) ** 2)
    ax.plot(tt, xx, color=ACCENT, lw=2.4, zorder=5)
    ax.scatter([t_stop], [SPACING], s=40, color=ACCENT, edgecolor=PAPER, lw=1.5, zorder=6)
    ax.annotate("ContinuousState   0.00 m", xy=(t_stop, SPACING), xytext=(6, -14),
                textcoords="offset points", fontsize=9, color=ACCENT, va="center",
                fontweight="bold", family="DejaVu Sans Mono")

    ax.set_xlim(17.2, 23.6)
    ax.set_ylim(188, 224)
    ax.set_xlabel("model time  (s)")
    ax.set_ylabel("position  (m)")
    ax.set_title("Every polled tick overshoots the platform. Shrinking it never reaches zero.",
                 loc="left", fontsize=11.5, fontweight="bold", pad=12)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "img-2-polling-vs-exact.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------- figure three
def fig_compaction():
    """Numbers as reported in PR #3800."""
    agents = ["200 agents", "1000 agents", "2000 agents"]
    heap_before = [14256, 89379, 239731]
    heap_after = [399, 2001, 4000]
    time_before = [0.12, 1.66, 15.37]
    time_after = [0.11, 0.97, 5.26]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    y = np.arange(3)
    h = 0.34

    ax1.barh(y + h / 2, heap_before, h, color=GREY, label="before", zorder=3)
    ax1.barh(y - h / 2, heap_after, h, color=ACCENT, label="after", zorder=3)
    ax1.set_xscale("log")
    ax1.set_yticks(y, agents)
    ax1.invert_yaxis()
    ax1.set_xlabel("peak entries on the event heap  (log scale)")
    ax1.set_xlim(100, 700000)
    ax1.legend(frameon=False, fontsize=9, loc="upper right")
    for i in range(3):
        ax1.text(heap_before[i] * 1.25, y[i] + h / 2, f"{heap_before[i]:,}",
                 va="center", fontsize=8.5, color=INK2, family="DejaVu Sans Mono")
        ax1.text(heap_after[i] * 1.25, y[i] - h / 2, f"{heap_after[i]:,}",
                 va="center", fontsize=8.5, color=ACCENT, family="DejaVu Sans Mono")
    ax1.set_title("Heap size", loc="left", fontsize=11, fontweight="bold")

    ax2.barh(y + h / 2, time_before, h, color=GREY, zorder=3)
    ax2.barh(y - h / 2, time_after, h, color=ACCENT, zorder=3)
    ax2.set_yticks(y, [])
    ax2.invert_yaxis()
    ax2.set_xlabel("wall time  (s)")
    ax2.set_xlim(0, 20)
    for i in range(3):
        ax2.text(time_before[i] + 0.4, y[i] + h / 2, f"{time_before[i]:.2f}s",
                 va="center", fontsize=8.5, color=INK2, family="DejaVu Sans Mono")
        ax2.text(time_after[i] + 0.4, y[i] - h / 2, f"{time_after[i]:.2f}s",
                 va="center", fontsize=8.5, color=ACCENT, family="DejaVu Sans Mono")
    ax2.set_title("Wall time", loc="left", fontsize=11, fontweight="bold")

    for ax in (ax1, ax2):
        tidy(ax)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Cancel-and-reschedule workload, before and after adaptive compaction (PR #3800)",
                 x=0.008, ha="left", fontsize=11.5, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.text(0.008, -0.03,
             "At 200 agents the time difference is inside the noise \u2014 the win there is memory. "
             "My own measurements, on one machine.",
             ha="left", fontsize=8.6, color=INK3, style="italic")
    fig.savefig(OUT / "img-3-compaction.png", bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------- figure four
def fig_false_start():
    """The CI benchmark bot on PR #3754 — the number I read, and the one I missed."""
    labels = ["Sugarscape\nrun time", "WolfSheep\ninit", "BoidFlockers\ninit", "BoltzmannWealth\ninit"]
    deltas = [-78.0, +30.9, +131.8, +147.8]
    colors = [TEAL, GREY, GREY, ACCENT]

    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    x = np.arange(4)
    ax.bar(x, deltas, 0.56, color=colors, zorder=3)
    ax.axhline(0, color=INK3, lw=1)
    ax.set_xticks(x, labels, fontsize=9.2)
    ax.set_ylabel("change vs main  (%)")
    ax.set_ylim(-105, 190)

    for i, d in enumerate(deltas):
        ax.text(i, d + (7 if d > 0 else -13), f"{d:+.1f}%", ha="center",
                fontsize=9.5, fontweight="bold", color=colors[i],
                family="DejaVu Sans Mono")

    ax.annotate("the column I read", xy=(0.32, -74), xytext=(0.62, -52),
                fontsize=9, color=TEAL,
                arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1})
    ax.annotate("the columns I didn't:\nmodels that never touch\na continuous state",
                xy=(2.66, 133), xytext=(0.55, 112), fontsize=9, color=ACCENT, va="top",
                arrowprops={"arrowstyle": "->", "color": ACCENT, "lw": 1,
                            "connectionstyle": "arc3,rad=-0.12"})

    ax.set_title("The benchmark that sold me the wrong design (abandoned PR #3754)",
                 loc="left", fontsize=11.5, fontweight="bold", pad=12)
    tidy(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "img-4-false-start.png", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ts, ps, vs, fired = run_tram()
    fig_trajectory(ts, ps, vs, fired)
    fig_polling()
    fig_compaction()
    fig_false_start()

    print("crossing times from the real run:")
    for e in fired[:5]:
        print(f"  {e['event']:<18} t={e['t']:.6f}  pos={e['position']:.4f}  spd={e['speed']:.4f}")
    print("\npolling comparison:")
    _, _, t_stop, _ = analytic()
    for dt in (1.0, 0.5, 0.25, 0.1):
        _, _, t_end, x_end = euler(dt)
        print(f"  tick {dt:<5} arrives {t_end:6.3f}s (+{t_end - t_stop:.3f})  "
              f"at {x_end:9.4f} m (+{x_end - SPACING:.4f})")
    print(f"\nwrote 4 figures to {OUT}")


if __name__ == "__main__":
    main()
