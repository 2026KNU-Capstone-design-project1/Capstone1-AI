"""N-11 geometry feasibility audit ONLY: not a hazard verdict or D' evaluation.

Uses predicted objects from spatial_judgments.json, never GT/filename labels.
Outputs candidate geometry and limitations without changing original predictions.
"""
import argparse
import csv
import json
import zipfile
from pathlib import Path


def intersection_features(item, platform):
    ix, iy, iw, ih = map(float, item['bbox'])
    px, py, pw, ph = map(float, platform['bbox'])
    overlap_x = max(0., min(ix + iw, px + pw) - max(ix, px))
    overlap_y = max(0., min(iy + ih, py + ph) - max(iy, py))
    paint_x_overlap = overlap_x / iw if iw > 0 else 0.
    paint_area_inside = overlap_x * overlap_y / (iw * ih) if iw > 0 and ih > 0 else 0.
    # Object center is horizontally over the broad platform bounding rectangle,
    # but a rectangular overlap is NOT proof of physical support/contact.
    center_x = ix + iw / 2
    return {
        'paint_x_overlap': round(paint_x_overlap, 4),
        'paint_area_inside': round(paint_area_inside, 4),
        'platform_area': round(pw * ph, 4),
        'paint_bottom': round(iy + ih, 4),
        'platform_top': round(py, 4),
        'paint_center_x_within_platform': px <= center_x <= px + pw,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zip_path', type=Path)
    ap.add_argument('--output', type=Path, default=Path('evaluation/results/n11_geometry_audit'))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip_path) as z:
        paths = [p for p in z.namelist() if p.endswith('spatial_judgments.json')]
        if len(paths) != 1:
            raise ValueError('Expected exactly one spatial_judgments.json')
        data = json.loads(z.read(paths[0]))
    items = data['items'] if isinstance(data, dict) else data

    rows = []
    for entry in items:
        if entry.get('reason_code') != 'rule_not_supported':
            continue
        checks = entry.get('checks') or []
        # ONLY use rule-check code, not image name or GT fields.
        if not any(c.get('hazard_code', c.get('code')) == 'N-11' for c in checks):
            continue
        objs = entry.get('objects') or []
        paint = [o for o in objs if o.get('code') == 'SO-30']
        platforms = [o for o in objs if o.get('code') == 'SO-03']
        pairs = [(o, p, intersection_features(o, p)) for o in paint for p in platforms]
        # Pick closest to physically interesting geometry as an exploratory feature only.
        best = max(pairs, key=lambda x: (x[2]['paint_x_overlap'], x[2]['paint_area_inside'])) if pairs else None
        features = best[2] if best else {}
        rows.append({
            'image_id': entry['image_id'],
            'paint_count': len(paint), 'platform_count': len(platforms),
            'pair_count': len(pairs),
            'has_bbox_pair': bool(pairs),
            **{k: features.get(k, '') for k in ['paint_x_overlap','paint_area_inside','platform_area','paint_bottom','platform_top','paint_center_x_within_platform']},
            'candidate_only': bool(pairs and features['paint_x_overlap'] >= 0.5),
            'risk_status': 'not_evaluable',
            'limitation': 'No system scaffold class, edge/support surface or falling-path evidence; bbox overlap is not proof of hazard',
        })

    output = args.output / 'n11_geometry_candidates.csv'
    with output.open('w', encoding='utf-8-sig', newline='') as f:
        fields = ['image_id','paint_count','platform_count','pair_count','has_bbox_pair','paint_x_overlap', 'paint_area_inside','platform_area', 'paint_bottom', 'platform_top','paint_center_x_within_platform','candidate_only','risk_status','limitation']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    n_pair = sum(int(x['has_bbox_pair']) for x in rows)
    n_candidate = sum(int(x['candidate_only']) for x in rows)
    n_giant = sum(bool(x['has_bbox_pair'] and x['platform_area'] != '' and float(x['platform_area']) >= 0.5) for x in rows)
    print('N-11 unsupported images:', len(rows))
    print('Paint + platform bbox pair:', n_pair)
    print('Exploratory horizontal-overlap candidates:', n_candidate)
    print('Broad platform boxes (area>=0.5, among chosen pairs):', n_giant)
    print('NO risk predictions changed; D\' remains untested.')
    print('saved:', output)


if __name__ == '__main__':
    main()
