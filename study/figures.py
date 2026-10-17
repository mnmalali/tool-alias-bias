"""Dependency-free SVG figures for the later manuscript draft."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from pilot.registry import CONDITIONS


SERIES = (
    ("jev", "P2", "#1463a5", "Jev raw option"),
    ("jev", "P3", "#39a5d8", "Jev grouped action"),
    ("luna", "P2", "#6846b6", "Luna raw option"),
    ("luna", "P3", "#a48be1", "Luna grouped action"),
    ("haiku", "P2", "#ae4c2b", "Haiku raw option"),
    ("haiku", "P3", "#e39b39", "Haiku grouped action"),
)


def _svg_start(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img">',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#182536} .small{font-size:13px} .tick{font-size:12px;fill:#506273} .title{font-size:20px;font-weight:700} .subtitle{font-size:13px;fill:#506273}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def _panel(lines: list[str], *, top: int, title: str, field: str, ymax: float,
           summaries: dict[tuple[str, str, str, str], dict[str, Any]]) -> None:
    left, right, height = 80, 925, 178
    bottom = top + height
    lines.append(f'<text x="80" y="{top - 25}" class="title">{escape(title)}</text>')
    for fraction in (0, 0.25, 0.5, 0.75, 1):
        y = bottom - fraction * height
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#dbe4ea" stroke-width="1"/>')
        lines.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="tick">{fraction * ymax:.2f}</text>')
    for index, condition in enumerate(CONDITIONS):
        x = left + index * (right - left) / 5
        lines.append(f'<text x="{x:.1f}" y="{bottom + 22}" text-anchor="middle" class="small">{condition}</text>')
    for model, policy, color, _ in SERIES:
        points: list[tuple[float, float]] = []
        for index, condition in enumerate(CONDITIONS):
            value = float(summaries[(model, policy, condition, "all")][field])
            points.append((left + index * (right - left) / 5, bottom - value / ymax * height))
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        lines.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in points:
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')


def write_figures(run_dir: Path, summaries: dict[tuple[str, str, str, str], dict[str, Any]],
                  primary: dict[str, float], *, synthetic: bool) -> list[str]:
    subtitle = "Synthetic mock plumbing output — no empirical model result" if synthetic else "Held-out evaluation, frozen C0 calibration thresholds"
    lines = _svg_start(1000, 620)
    lines.append('<text x="80" y="35" class="title">Tool-alias study outcomes by catalog</text>')
    lines.append(f'<text x="80" y="56" class="subtitle">{escape(subtitle)}</text>')
    for index, (_, _, color, label) in enumerate(SERIES):
        x = 80 + (index % 3) * 290
        y = 78 + (index // 3) * 23
        lines.append(f'<line x1="{x}" y1="{y}" x2="{x + 23}" y2="{y}" stroke="{color}" stroke-width="4"/>')
        lines.append(f'<text x="{x + 30}" y="{y + 5}" class="small">{escape(label)}</text>')
    _panel(lines, top=145, title="Automatic coverage", field="coverage", ymax=1.0, summaries=summaries)
    maximum_wrong = max(float(summaries[(model, policy, condition, "all")]["accepted_wrong_rate"])
                        for model, policy, _, _ in SERIES for condition in CONDITIONS)
    wrong_axis = max(0.05, min(1.0, maximum_wrong * 1.2))
    _panel(lines, top=395, title="Accepted wrong actions / all requests", field="accepted_wrong_rate", ymax=wrong_axis, summaries=summaries)
    lines.append('</svg>')
    first = run_dir / "figure_coverage_wrong.svg"
    first.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    delta = float(primary["estimate"])
    low = float(primary["ci95_low"])
    high = float(primary["ci95_high"])
    bound = max(0.15, abs(low) * 1.15, abs(high) * 1.15, abs(delta) * 1.15)
    x0, x1 = 85, 655
    scale = lambda value: x0 + (value + bound) / (2 * bound) * (x1 - x0)
    second_lines = _svg_start(740, 265)
    second_lines.append('<text x="55" y="35" class="title">Primary endpoint: Jev raw-option coverage shift</text>')
    second_lines.append(f'<text x="55" y="56" class="subtitle">{escape(subtitle)}</text>')
    for marker, color, dashed in ((0.0, "#253647", ""), (-0.10, "#be843a", ' stroke-dasharray="5 5"'), (0.10, "#be843a", ' stroke-dasharray="5 5"')):
        x = scale(marker)
        second_lines.append(f'<line x1="{x:.1f}" y1="85" x2="{x:.1f}" y2="178" stroke="{color}" stroke-width="1.5"{dashed}/>')
    second_lines.append(f'<line x1="{scale(low):.1f}" y1="130" x2="{scale(high):.1f}" y2="130" stroke="#1463a5" stroke-width="5"/>')
    for point in (low, high):
        second_lines.append(f'<line x1="{scale(point):.1f}" y1="119" x2="{scale(point):.1f}" y2="141" stroke="#1463a5" stroke-width="3"/>')
    second_lines.append(f'<circle cx="{scale(delta):.1f}" cy="130" r="8" fill="#1463a5"/>')
    second_lines.append(f'<text x="370" y="204" class="small" text-anchor="middle">C2 minus C0 coverage; 95% family bootstrap interval</text>')
    second_lines.append(f'<text x="370" y="230" class="small" text-anchor="middle">Estimate {delta:+.3f}; interval [{low:+.3f}, {high:+.3f}]; dashed lines ±0.10 practical bound</text>')
    second_lines.append('</svg>')
    second = run_dir / "figure_primary_delta.svg"
    second.write_text("\n".join(second_lines), encoding="utf-8", newline="\n")
    return [str(first), str(second)]
