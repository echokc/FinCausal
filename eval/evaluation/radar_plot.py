import argparse
import html
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List


DEFAULT_PILLARS = [
    "Temporal Causality",
    "Statistical Causality",
    "System Feedback Causality",
    "Regime Causality",
    "Risk Causality",
]


def load_eval_records(input_path: str) -> List[Dict[str, Any]]:
    with open(input_path, "r", encoding="utf-8") as handle:
        text = handle.read().strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("Expected a JSON array or JSONL records")
        return payload
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def pillar_pass_rates(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    totals: Dict[str, int] = defaultdict(int)
    passes: Dict[str, int] = defaultdict(int)
    quarantines: Dict[str, int] = defaultdict(int)

    for record in records:
        pillar = str(record.get("pillar") or "Unknown")
        decision = str(record.get("decision") or record.get("score", {}).get("status") or "").lower()
        totals[pillar] += 1
        if decision == "pass":
            passes[pillar] += 1
        elif decision == "quarantine":
            quarantines[pillar] += 1

    metrics = {}
    for pillar, total in totals.items():
        metrics[pillar] = {
            "total": float(total),
            "pass_count": float(passes[pillar]),
            "quarantine_count": float(quarantines[pillar]),
            "pass_rate": passes[pillar] / total if total else 0.0,
        }
    return metrics


def render_pillar_radar_svg(
    metrics: Dict[str, Dict[str, float]],
    *,
    title: str = "FinCausal Recipe Eval Radar",
    pillars: List[str] | None = None,
    size: int = 720,
) -> str:
    pillars = pillars or list(DEFAULT_PILLARS)
    extras = [pillar for pillar in metrics if pillar not in pillars]
    pillars = pillars + sorted(extras)
    if len(pillars) < 3:
        raise ValueError("Radar plot requires at least three pillars")

    center = size / 2
    radius = size * 0.31
    label_radius = size * 0.40
    chart_top = size * 0.10
    center_y = center + size * 0.04
    levels = [0.25, 0.50, 0.75, 1.00]
    count = len(pillars)

    def point(index: int, value: float, r: float = radius) -> tuple[float, float]:
        angle = -math.pi / 2 + 2 * math.pi * index / count
        return center + math.cos(angle) * r * value, center_y + math.sin(angle) * r * value

    grid = []
    for level in levels:
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(idx, level) for idx in range(count)))
        grid.append(f'<polygon points="{points}" fill="none" stroke="#d0d7de" stroke-width="1"/>')

    axes = []
    labels = []
    for idx, pillar in enumerate(pillars):
        x, y = point(idx, 1.0)
        axes.append(f'<line x1="{center:.1f}" y1="{center_y:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#d0d7de" stroke-width="1"/>')
        lx, ly = point(idx, label_radius / radius)
        anchor = "middle"
        if lx < center - 20:
            anchor = "end"
        elif lx > center + 20:
            anchor = "start"
        rate = metrics.get(pillar, {}).get("pass_rate", 0.0)
        total = int(metrics.get(pillar, {}).get("total", 0))
        label = html.escape(pillar)
        labels.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
            f'font-family="Arial, sans-serif" font-size="13" fill="#24292f">{label}</text>'
        )
        labels.append(
            f'<text x="{lx:.1f}" y="{ly + 17:.1f}" text-anchor="{anchor}" '
            f'font-family="Arial, sans-serif" font-size="12" fill="#57606a">{rate:.0%} pass, n={total}</text>'
        )

    data_points = " ".join(
        f"{x:.1f},{y:.1f}"
        for x, y in (point(idx, metrics.get(pillar, {}).get("pass_rate", 0.0)) for idx, pillar in enumerate(pillars))
    )
    dots = []
    for idx, pillar in enumerate(pillars):
        x, y = point(idx, metrics.get(pillar, {}).get("pass_rate", 0.0))
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#0969da"/>')

    tick_labels = []
    for level in levels:
        _, y = point(0, level)
        tick_labels.append(
            f'<text x="{center + 6:.1f}" y="{y + 4:.1f}" font-family="Arial, sans-serif" '
            f'font-size="11" fill="#6e7781">{level:.0%}</text>'
        )

    escaped_title = html.escape(title)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{center:.1f}" y="{chart_top:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#24292f">{escaped_title}</text>
  <text x="{center:.1f}" y="{chart_top + 26:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#57606a">Pass rate by causal pillar</text>
  <g>
    {"".join(grid)}
    {"".join(axes)}
    {"".join(tick_labels)}
    <polygon points="{data_points}" fill="#0969da" fill-opacity="0.20" stroke="#0969da" stroke-width="2"/>
    {"".join(dots)}
    {"".join(labels)}
  </g>
</svg>
"""


def write_pillar_radar_plot(input_path: str, output_path: str, *, title: str = "FinCausal Recipe Eval Radar") -> Dict[str, Dict[str, float]]:
    records = load_eval_records(input_path)
    metrics = pillar_pass_rates(records)
    svg = render_pillar_radar_svg(metrics, title=title)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(svg)
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a pillar-level radar plot from recipe eval JSONL records.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--title", default="FinCausal Recipe Eval Radar")
    args = parser.parse_args()

    metrics = write_pillar_radar_plot(args.input_path, args.output_path, title=args.title)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
