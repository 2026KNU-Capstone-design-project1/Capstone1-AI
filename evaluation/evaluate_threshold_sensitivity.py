#!/usr/bin/env python3
"""Sensitivity analysis of the exact A-D policy in evaluate_team_ablation.py.

Changes ONLY its global CONF and AREA. No GT used in decisions; only scoring.
Threshold selection on this evaluation set is exploratory, not a valid held-out test.
"""
import argparse
import csv
import importlib.util
from pathlib import Path

GRID = [(0.3,0.005), (0.3,0.01), (0.5,0.005), (0.5,0.01),
        (0.5,0.02), (0.7,0.01), (0.7,0.02)]

def import_evaluator(path):
    spec = importlib.util.spec_from_file_location('team_ablation_evaluator', path)
    if not spec or not spec.loader:
        raise RuntimeError('Could not import original evaluator')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for fn in ('read_data','decisions','summarize','save_csv'):
        if not hasattr(module, fn):
            raise ValueError(f'Original evaluator missing function: {fn}')
    return module

def evaluate(maps, evaluator, conf, area):
    evaluator.CONF = conf
    evaluator.AREA = area
    details = []
    for image_id in sorted(maps['gt_labels']):
        gt = maps['gt_labels'][image_id]
        yo = maps['yolo_predictions'][image_id]
        sp = maps['spatial_judgments'][image_id]
        for model, (decision, codes) in evaluator.decisions(yo, sp).items():
            details.append({'image_id':image_id,'model':model,
                            'gt_hazard':bool(gt['is_hazard']),
                            'gt_code':gt.get('hazard_code') or '',
                            'decision':decision,
                            'predicted_codes':';'.join(sorted(codes))})
    return details, [evaluator.summarize(details, model) for model in 'ABCD']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', help='Original ab_experiment.zip or extracted folder')
    parser.add_argument('--evaluator', default='evaluation/evaluate_team_ablation.py',
                        help='Path to unmodified A-D evaluation script')
    parser.add_argument('--output', default='evaluation/results/threshold_sensitivity')
    args = parser.parse_args()
    evaluator = import_evaluator(Path(args.evaluator))
    maps = evaluator.read_data(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    results=[]
    reviews=[]
    baseline=None
    for i,(conf,area) in enumerate(GRID,1):
        rows,summaries=evaluate(maps,evaluator,conf,area)
        if (conf,area)==(0.5,0.01):
            baseline=summaries
        for stat in summaries:
            results.append({'condition':f'T{i}', 'confidence_min':conf,'bbox_area_min':area,
                            **stat})
        for row in rows:
            if row['model']=='D' and row['decision']=='review':
                reviews.append({'condition':f'T{i}', 'confidence_min':conf,
                                'bbox_area_min':area, **row})
    assert baseline is not None
    evaluator.save_csv(out/'sensitivity_all_models.csv',results)
    evaluator.save_csv(out/'sensitivity_D_reviews.csv',reviews)
    d_results=[r for r in results if r['model']=='D']
    evaluator.save_csv(out/'sensitivity_D_summary.csv',d_results)
    print('Dataset size:',len(maps['gt_labels']))
    print('D sensitivity (F1/Recall calculated only among covered cases):')
    print('condition | conf | area | F1 covered | coverage | total hazard recall | review | FP normal')
    for r in d_results:
        print(f"{r['condition']:>3} | {r['confidence_min']:.2f} | {r['bbox_area_min']:.3f} | "
              f"{r['f1_covered']!s:>10} | {r['coverage']!s:>8} | "
              f"{r['recall_all_hazard_review_as_miss']!s:>19} | "
              f"{r['n_review']:>6} | {r['fp_normal']:>9}")
    ref=next(r for r in d_results if r['condition']=='T4')
    print('\nT4 reference must reproduce the prior D run:')
    print('  F1:',ref['f1_covered'],'coverage:',ref['coverage'],
          'review:',ref['n_review'],'exact TP:',ref['tp_exact_code'])
    assert (ref['n_review'],ref['tp_exact_code'],ref['n_total'])==(261,322,1256), \
        'T4 differs from prior experiment; check evaluator or data version'
    print('Saved:',out)
    print('CAUTION: threshold comparisons on these same 1,256 images are exploratory;')
    print('do not select the best threshold and then report its score as held-out performance.')

if __name__=='__main__':
    main()
