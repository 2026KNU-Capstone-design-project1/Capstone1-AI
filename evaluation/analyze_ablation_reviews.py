#!/usr/bin/env python3
"""Explain A→D review transitions without using GT in predictions.
Only uses GT for analysis after predictions have been reproduced.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec


def load_evaluator(path):
    spec = spec_from_file_location('team_ablation', str(path))
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def explanation(yo, sp, evaluator):
    pred = yo.get('predictions', [])
    high = [p for p in pred if float(p.get('confidence', 0)) >= evaluator.CONF]
    good = [p for p in pred if evaluator.quality(p)]
    b_codes = {f.get('hazard_code') for f in sp.get('findings', [])
               if f.get('hazard_code') and f.get('status') in ('위험', 'hazard')}
    good_codes = {p.get('hazard_code') for p in good}
    if good_codes & b_codes:
        return 'confirmed_overlap'
    if good_codes and b_codes:
        return 'code_disagreement'
    if good_codes and not b_codes:
        return 'yolo_good_spatial_no_match'
    if b_codes and pred and not good_codes:
        return 'spatial_hazard_yolo_quality_failed'
    if b_codes and not pred:
        return 'spatial_hazard_yolo_absent'
    if pred and not good_codes:
        return 'yolo_predictions_quality_failed'
    if sp.get('evaluation_status') == 'not_evaluable':
        return 'spatial_not_evaluable_no_yolo'
    return 'normal_no_evidence'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', help='ab_experiment.zip or extracted directory')
    p.add_argument('--evaluator', default='evaluation/evaluate_team_ablation.py')
    p.add_argument('--output', default='evaluation/results/team_ablation')
    a=p.parse_args()
    e=load_evaluator(Path(a.evaluator))
    data=e.read_data(a.input)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for image_id in sorted(data['gt_labels']):
        gt=data['gt_labels'][image_id];yo=data['yolo_predictions'][image_id];sp=data['spatial_judgments'][image_id]
        decisions=e.decisions(yo,sp)
        if decisions['D'][0]!='review': continue
        pred=yo.get('predictions', [])
        high=[item for item in pred if float(item.get('confidence',0))>=e.CONF]
        good=[item for item in pred if e.quality(item)]
        b_codes={f.get('hazard_code') for f in sp.get('findings',[]) if f.get('hazard_code') and f.get('status') in ('hazard','위험')}
        rows.append(dict(image_id=image_id,gt_hazard=gt['is_hazard'],gt_code=gt.get('hazard_code') or '',a_decision=decisions['A'][0],d_decision='review',review_reason=explanation(yo,sp,e),spatial_evaluation_status=sp.get('evaluation_status',''),spatial_reason_code=sp.get('reason_code',''),n_yolo_predictions=len(pred),n_yolo_above_conf=len(high),n_yolo_above_quality=len(good),spatial_hazard_codes=';'.join(sorted(b_codes)),quality_yolo_codes=';'.join(sorted({x['hazard_code'] for x in good}))))
    assert len(rows)==261, f'Expected 261 review cases on supplied sample; got {len(rows)}. Verify input and policy.'
    path=output/'d_review_analysis.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('D review total:',len(rows))
    print('A -> D review:',dict(Counter(r['a_decision'] for r in rows)))
    print('GT hazard review:',sum(str(r['gt_hazard']).lower()=='true' for r in rows))
    print('\nExclusive review reason breakdown:')
    for reason,count in Counter(r['review_reason'] for r in rows).most_common():
        print(f'  {reason}: {count}')
    print('\nSpatial reason_code breakdown (can overlap with categories above):')
    for reason,count in Counter(r['spatial_reason_code'] for r in rows).most_common():
        print(f'  {reason}: {count}')
    print('saved:',path)

if __name__ == '__main__': main()
