#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_scores(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    axes = data['axes']
    scores = [data['scores'].get(a, 0.0) for a in axes]
    return axes, scores, data


def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    if frame != 'circle':
        raise NotImplementedError
    return theta


def render(axes, values, title: str, out_path: Path):
    N = len(axes)
    theta = radar_factory(N)
    values = np.asarray(values, dtype=float)
    # Close the polygon
    values = np.concatenate((values, [values[0]]))
    theta = np.concatenate((theta, [theta[0]]))

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Grid and labels
    ax.set_thetagrids(theta[:-1] * 180/np.pi, labels=axes, fontsize=9)
    ax.set_ylim(0, 5)
    ax.set_rgrids([1, 2, 3, 4, 5], angle=0)

    # Plot
    ax.plot(theta, values, color='#1f77b4', linewidth=2)
    ax.fill(theta, values, color='#1f77b4', alpha=0.25)
    ax.set_title(title, y=1.08)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f'Saved radar chart to {out_path}')


def main():
    ap = argparse.ArgumentParser(description='Render Radar chart from scores JSON')
    ap.add_argument('--scores', required=True, help='Path to radar_YYYY-MM.json')
    ap.add_argument('--out', default=None, help='Path to output PNG')
    args = ap.parse_args()

    axes, values, meta = load_scores(Path(args.scores))
    month = meta.get('month', 'YYYY-MM')
    title = f'ML Radar — {month}'
    out = Path(args.out) if args.out else Path('assessments/charts') / f'radar_{month}.png'
    render(axes, values, title, out)


if __name__ == '__main__':
    main()

