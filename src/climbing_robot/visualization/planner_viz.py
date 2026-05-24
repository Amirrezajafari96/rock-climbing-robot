"""
Visualization tools for the climbing planner and stability analysis.

Uses matplotlib for 2D wall plots showing:
  - Hold layout with occupancy markers
  - Support polygon and COM projection
  - Planned trajectory overlays
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def plot_wall_overview(
    hold_system: Any,
    active_grip_ids: list[str] | None = None,
    com_xz: tuple[float, float] | None = None,
    support_polygon: np.ndarray | None = None,
    planned_moves: list[Any] | None = None,
    title: str = "Climbing Wall — Hold Layout",
    save_path: str | None = None,
) -> None:
    """
    Render a 2D top-down (X-Z plane) view of the climbing wall.

    Shows:
      - All holds (colored by type)
      - Active grips (highlighted)
      - COM projection (red dot)
      - Support polygon (dashed outline)
      - Planned move arrows

    Args:
        hold_system:      HoldSystem instance.
        active_grip_ids:  Hold IDs currently gripped.
        com_xz:           (x, z) COM projection on wall plane.
        support_polygon:  (N,2) polygon vertices in (x,z).
        planned_moves:    List of ClimbMove with target_hold set.
        title:            Plot title.
        save_path:        If set, save PNG to this path.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
    except ImportError:
        logger.error("matplotlib not installed. Run: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(8, 12))

    active = set(active_grip_ids or [])

    # Color map for hold types
    color_map = {
        "JUG": "#F5A623",
        "CRIMP": "#D0021B",
        "SLOPER": "#7ED321",
        "PINCH": "#9B59B6",
        "GOAL": "#2ECC71",
    }

    for hold in hold_system:
        x, z = hold.x, hold.z
        color = color_map.get(hold.hold_type.name, "#95A5A6")
        marker_size = 200 if hold.hold_id in active else 120
        edge_color = "black" if hold.hold_id in active else "gray"
        edge_width = 2 if hold.hold_id in active else 0.5
        ax.scatter(x, z, c=color, s=marker_size,
                   edgecolors=edge_color, linewidths=edge_width, zorder=3)
        ax.annotate(hold.hold_id, (x, z), textcoords="offset points",
                    xytext=(5, 3), fontsize=7, color="dimgray")

    # Support polygon
    if support_polygon is not None and len(support_polygon) >= 3:
        poly = Polygon(support_polygon, closed=True, fill=True,
                       facecolor="cyan", alpha=0.25, edgecolor="blue",
                       linestyle="--", linewidth=1.5)
        ax.add_patch(poly)

    # COM projection
    if com_xz is not None:
        ax.scatter(*com_xz, c="red", s=300, marker="*",
                   zorder=5, label="COM", edgecolors="darkred")

    # Planned move arrows
    if planned_moves:
        for move in planned_moves:
            if move.target_hold:
                try:
                    h = hold_system.get(move.target_hold)
                    ax.annotate(
                        "", xy=(h.x, h.z), xytext=(h.x, h.z - 0.2),
                        arrowprops=dict(arrowstyle="->", color="purple", lw=2)
                    )
                except KeyError:
                    pass

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=t) for t, c in color_map.items()
    ]
    legend_patches.append(mpatches.Patch(color="cyan", alpha=0.4, label="Support polygon"))
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    ax.set_xlabel("X position [m]", fontsize=11)
    ax.set_ylabel("Z height [m]", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(0, 2.7)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved wall plot to %s", save_path)
    else:
        plt.show()
    plt.close()


def plot_joint_trajectory(
    trajectory: list[np.ndarray],
    joint_names: list[str] | None = None,
    title: str = "Joint Trajectory",
    save_path: str | None = None,
) -> None:
    """Plot joint angles over trajectory steps."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed")
        return

    traj = np.array(trajectory)
    n_steps, n_joints = traj.shape
    names = joint_names or [f"q{i}" for i in range(n_joints)]
    steps = np.arange(n_steps)

    fig, axes = plt.subplots(n_joints, 1, figsize=(10, 2.5 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, names)):
        ax.plot(steps, np.degrees(traj[:, i]), "b-", linewidth=1.8)
        ax.set_ylabel(f"{name} [°]", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

    axes[-1].set_xlabel("Step", fontsize=10)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved trajectory plot to %s", save_path)
    else:
        plt.show()
    plt.close()
