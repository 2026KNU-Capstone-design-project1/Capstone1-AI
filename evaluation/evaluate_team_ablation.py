#!/usr/bin/env python3
"""Evaluate a preregistered illustrative A-D decision policy on team JSON inputs.

WARNING: This is an experiment policy, not proof the production pipeline uses it.
GT is only used for scoring; gt_code_check_for_analysis_only is never accessed.
"""
import argparse, csv, json, zipfile
from collections import Counter
from pathlib import Path

CONF = 0.5
AREA = 0.01


def read_data(root):
    path = Path(root)
    if path.is_file() and path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            def load(name):
                candidates = [p for p in archive.namelist() if p.endswith('/' + name + '.json') or p == name + '.json']
                if len(candidates) != 1:
                    raise ValueError(f'{name}: expected one JSON in zip, got {candidates}')
                return json.loads(archive.read(candidates[0]))['items']
    else:
        def load(name):
            return json.loads((path / (name + '.json')).read_text(encoding='utf-8'))['items']
    if path.is_file() and path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            result = {name: json.loads(archive.read(next(p for p in archive.namelist() if p.endswith('/' + name + '.json') or p == name + '.json')))['items'] for name in ('gt_labels','yolo_predictions','spatial_judgments')}
    else:
        result = {name: load(name) for name in ('gt_labels','yolo_predictions','spatial_judgments')}
    maps = {}
    for name, rows in result.items():
        ids = [r['image_id'] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f'duplicate image ID: {name}')
        maps[name] = {r['image_id']: r for r in rows}
    if not (maps['gt_labels'].keys() == maps['yolo_predictions'].keys() == maps['spatial_judgments'].keys()):
        raise ValueError('The image_id sets differ across the JSON files')
    return maps


def quality(p):
    box = p.get('bbox') or {}
    area = float(p.get('area_ratio', float(box.get('width', 0)) * float(box.get('height', 0))))
    return float(p.get('confidence', 0)) >= CONF and area >= AREA


def decisions(yo, sp):
    pred = yo.get('predictions', [])
    high = [p for p in pred if float(p.get('confidence',0)) >= CONF]
    good = [p for p in pred if quality(p)]
    findings = sp.get('findings', [])
    codes_b = {f.get('hazard_code') for f in findings if f.get('hazard_code') and f.get('status') in ('위험','hazard')}
    status = sp.get('evaluation_status')
    # Output (decision, predicted hazard codes); decision ∈ hazard / normal / review
    a = ('hazard', {p['hazard_code'] for p in high}) if high else ('normal',set())
    b = ('hazard',codes_b) if codes_b else (('review',set()) if status == 'not_evaluable' else ('normal',set()))
    c = ('hazard',{p['hazard_code'] for p in good}) if good else (('review',set()) if pred else ('normal',set()))
    # D: require quality YOLO and independent spatial finding for a confirmed code;
    # otherwise any disagreement, missing quality, or unevaluable scene -> review.
    overlap = codes_b & {p['hazard_code'] for p in good}
    d = ('hazard',overlap) if overlap else (('review',set()) if (codes_b or pred or status == 'not_evaluable') else ('normal',set()))
    return {'A':a,'B':b,'C':c,'D':d}


def summarize(rows, model):
    entries = [r for r in rows if r['model']==model]
    reviewed = [r for r in entries if r['decision']=='review']
    covered = [r for r in entries if r['decision']!='review']
    tp = sum(r['gt_hazard'] and r['decision']=='hazard' and r['gt_code'] in r['predicted_codes'].split(';') for r in covered)
    fp = sum(not r['gt_hazard'] and r['decision']=='hazard' for r in covered)
    fn = sum(r['gt_hazard'] and not (r['decision']=='hazard' and r['gt_code'] in r['predicted_codes'].split(';')) for r in covered)
    tn = sum(not r['gt_hazard'] and r['decision']=='normal' for r in covered)
    miscode = sum(r['gt_hazard'] and r['decision']=='hazard' and r['gt_code'] not in r['predicted_codes'].split(';') for r in covered)
    hazard_review = sum(r['gt_hazard'] for r in reviewed)
    normal_review = len(reviewed)-hazard_review
    div = lambda a,b: round(a/b,6) if b else None
    prec = div(tp,tp+fp+miscode)
    recall = div(tp,tp+fn)
    f1 = div(2*prec*recall,prec+recall) if prec is not None and recall is not None else None
    return {'model':model,'n_total':len(entries),'n_covered':len(covered),'n_review':len(reviewed),'coverage':div(len(covered),len(entries)),
            'tp_exact_code':tp,'fp_normal':fp,'fn_hazard_in_covered':fn,'tn':tn,'wrong_code_hazard':miscode,
            'gt_hazard_review':hazard_review,'gt_normal_review':normal_review,'precision_covered':prec,
            'recall_covered':recall,'f1_covered':f1,
            'fpr_covered':div(fp,fp+tn),'recall_all_hazard_review_as_miss':div(tp,tp+fn+hazard_review)}


def save_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input',help='ab_experiment.zip or extracted ab_experiment directory')
    parser.add_argument('--output',default='evaluation/results/team_ablation')
    args=parser.parse_args()
    maps=read_data(args.input)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    detail=[]
    for image_id in sorted(maps['gt_labels']):
        gt=maps['gt_labels'][image_id]
        yo=maps['yolo_predictions'][image_id]
        sp=maps['spatial_judgments'][image_id]
        for model,(decision,codes) in decisions(yo,sp).items():
            detail.append({'image_id':image_id,'model':model,'gt_hazard':bool(gt['is_hazard']),
                           'gt_code':gt.get('hazard_code') or '', 'decision':decision,
                           'predicted_codes':';'.join(sorted(codes))})
    summary=[summarize(detail,m) for m in 'ABCD']
    save_csv(out/'ablation_details.csv',detail)
    save_csv(out/'ablation_summary.csv',summary)
    print('validated images:',len(maps['gt_labels']))
    print('policy thresholds: confidence >=',CONF,'area >=',AREA)
    for x in summary:print(x)
    print('saved:',out)
    print('Caution: B/D are illustrative policies; validate production semantics and held-out protocol before paper use.')

if __name__=='__main__':main()
