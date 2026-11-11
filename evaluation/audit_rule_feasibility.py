#!/usr/bin/env python3
"""Audit observed object evidence for unsupported spatial rules.

Never accesses GT labels, gt_code_check_for_analysis_only, or image-ID labels.
This is descriptive evidence only, NOT an improvement experiment.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

TARGET_CODES = ('N-11', 'N-24', 'N-50')


def load_spatial(zip_path):
    with ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.endswith('/spatial_judgments.json') or n == 'spatial_judgments.json']
        if len(names) != 1:
            raise ValueError(f'Expected one spatial_judgments.json, found {names}')
        return json.loads(archive.read(names[0]))['items']


def codes_from_checks(item):
    return {c.get('hazard_code') for c in item.get('checks', []) if c.get('hazard_code')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, default=Path('evaluation/results/rule_feasibility'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    items = load_spatial(args.archive)
    rows = []
    summary = []
    for code in TARGET_CODES:
        cases = [x for x in items if x.get('reason_code') == 'rule_not_supported' and code in codes_from_checks(x)]
        freq = Counter()
        n_with_objects = 0
        for case in cases:
            obs = case.get('objects') or []
            if obs:
                n_with_objects += 1
            unique = {(str(o.get('code', '')), str(o.get('name', ''))) for o in obs}
            freq.update(unique)
            rows.append({
                'hazard_code': code,
                'image_id': case['image_id'],
                'object_count': len(obs),
                'object_codes': ';'.join(sorted({str(o.get('code', '')) for o in obs})),
                'object_names': ';'.join(sorted({str(o.get('name', '')) for o in obs})),
                'max_object_confidence': max((float(o.get('confidence') or 0) for o in obs), default=0),
                'evaluation_status': case.get('evaluation_status', ''),
                'reason_code': case.get('reason_code', ''),
            })
        for (object_code, object_name), n in freq.most_common():
            summary.append({'hazard_code':code, 'unsupported_cases':len(cases), 'cases_with_any_objects':n_with_objects,
                            'object_code':object_code, 'object_name':object_name, 'images_with_object':n})
        print(f'\n{code}: unsupported={len(cases)}, with_objects={n_with_objects}')
        for (obj_code, name), n in freq.most_common(15):
            print(f'  {obj_code} {name}: {n}/{len(cases)}')
    for filename, data, fields in [
        ('unsupported_object_inventory.csv', summary, ['hazard_code','unsupported_cases','cases_with_any_objects','object_code','object_name','images_with_object']),
        ('unsupported_object_cases.csv', rows, ['hazard_code','image_id','object_count','object_codes','object_names','max_object_confidence','evaluation_status','reason_code'])
    ]:
        dst = args.output / filename
        with dst.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(data)
        print('saved:', dst)
    print('WARNING: Descriptive test-set audit only. GT, embedded labels and analysis-only fields are intentionally ignored.')

if __name__=='__main__':
    main()
