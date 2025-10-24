#!/usr/bin/env python3
import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import yaml

ALPHA = 0.55
BETA = 0.35
GAMMA = 0.10
HALF_LIFE_MONTHS = 6.0


def load_axes(axes_path: Path):
    with open(axes_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    axes = list(data.keys())
    return axes


def load_test_responses(path: Path):
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def score_test(axes, responses):
    # responses: [{axis, difficulty(1..5), correct(bool), weight(optional)}]
    out = {a: 0.0 for a in axes}
    denom = {a: 0.0 for a in axes}
    for r in responses:
        a = r.get('axis')
        if a not in out:
            continue
        d = int(r.get('difficulty', 3))
        w = float(r.get('weight', 1.0 + 0.15 * (d - 3)))
        c = 1.0 if r.get('correct') else 0.0
        out[a] += w * c
        denom[a] += w
    for a in axes:
        out[a] = (out[a] / denom[a]) if denom[a] > 0 else 0.0
    return out


def month_str(dt: datetime):
    return dt.strftime('%Y-%m')


def list_biweeks(base_dir: Path, year: str):
    root = base_dir / year
    if not root.exists():
        return []
    return [p for p in root.glob('biweek-*') if p.is_dir()]


def parse_meta(meta_path: Path):
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        return {}


def load_mapping(mapping_path: Path):
    with open(mapping_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def score_reading(axes, repo_root: Path, mapping_path: Path, target_month: str):
    # Sum impacts for metas in target_month, with exponential decay for older metas if present.
    # meta.yaml expected keys: date (YYYY-MM-DD), categories: [..]
    mapping = load_mapping(mapping_path)
    impacts = {a: 0.0 for a in axes}
    # Scan years present
    for year_dir in (repo_root / '2025', repo_root / '2024'):
        if not year_dir.exists():
            continue
        for biweek in year_dir.glob('biweek-*'):
            for sub in ['theory', 'practice']:
                meta = parse_meta(biweek / sub / 'meta.yaml')
                if not meta:
                    continue
                date_str = meta.get('date') or meta.get('published')
                try:
                    dt = datetime.strptime(date_str, '%Y-%m-%d') if date_str else None
                except Exception:
                    dt = None
                month_tag = month_str(dt) if dt else year_dir.name + '-' + '01'
                # decay
                if dt:
                    target_dt = datetime.strptime(target_month + '-01', '%Y-%m-%d')
                    delta_months = (target_dt.year - dt.year) * 12 + (target_dt.month - dt.month)
                else:
                    delta_months = 0
                lamb = 0.5 ** (delta_months / HALF_LIFE_MONTHS)
                if month_tag != target_month and delta_months > 12:
                    continue
                cats = meta.get('categories', [])
                for c in cats:
                    weights = mapping.get(c, {})
                    for axis, w in weights.items():
                        if axis in impacts:
                            impacts[axis] += float(w) * float(lamb)
    # squashing
    for a in axes:
        impacts[a] = math.tanh(BETA * impacts[a]) * 5.0
    return impacts


def load_practice_checks(month_dir: Path):
    # Optional file: practice.json with entries [{axis, score in [0,1]}]
    fp = month_dir / 'practice.json'
    if not fp.exists():
        return []
    with open(fp, 'r', encoding='utf-8') as f:
        return json.load(f)


def score_practice(axes, items):
    agg = {a: [] for a in axes}
    for it in items:
        a = it.get('axis')
        s = float(it.get('score', 0.0))
        if a in agg:
            agg[a].append(max(0.0, min(1.0, s)))
    out = {}
    for a in axes:
        v = sum(agg[a]) / len(agg[a]) if agg[a] else 0.0
        out[a] = GAMMA * v * 5.0
    return out


def combine(axes, s_test, r_read, r_prac):
    res = {}
    for a in axes:
        res[a] = max(0.0, min(5.0, ALPHA * 5.0 * s_test.get(a, 0.0) + r_read.get(a, 0.0) + r_prac.get(a, 0.0)))
    return res


def main():
    ap = argparse.ArgumentParser(description='Score monthly ML Radar assessment')
    ap.add_argument('--repo-root', default='.', help='Repository root')
    ap.add_argument('--axes', default='assessments/axes.yaml')
    ap.add_argument('--mapping', default='assessments/mappings/arxiv2axis.yaml')
    ap.add_argument('--month', default=datetime.now().strftime('%Y-%m'))
    ap.add_argument('--responses', default=None, help='Path to responses JSON (optional)')
    ap.add_argument('--monthly-dir', default=None, help='Path to monthly dir with keys/practice')
    ap.add_argument('--out', default=None, help='Output JSON path in assessments/scores')
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    axes = load_axes(repo_root / args.axes)

    # inputs
    if args.responses:
        responses_path = Path(args.responses)
    else:
        responses_path = repo_root / 'assessments' / 'responses' / f'{args.month}.json'
    responses = load_test_responses(responses_path)
    s_test = score_test(axes, responses)

    r_read = score_reading(axes, repo_root, repo_root / args.mapping, args.month)

    month_dir = Path(args.monthly_dir) if args.monthly_dir else (repo_root / 'assessments' / 'monthly' / args.month)
    practice = load_practice_checks(month_dir)
    r_prac = score_practice(axes, practice)

    scores = combine(axes, s_test, r_read, r_prac)

    out_dir = repo_root / 'assessments' / 'scores'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else (out_dir / f'radar_{args.month}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'month': args.month,
            'axes': axes,
            'test': s_test,
            'reading': r_read,
            'practice': r_prac,
            'scores': scores,
        }, f, ensure_ascii=False, indent=2)
    print(f'Wrote scores to {out_path}')


if __name__ == '__main__':
    main()

