#!/usr/bin/env python3
"""Descriptive audit of unsupported spatial rules; no rule tuning or GT leakage.

Uses image IDs for joining only. GT is used solely for offline evaluation.
Does NOT use gt_code_check_for_analysis_only or infer a hazard code from image_id.
"""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_team_ablation import read_data, decisions


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', help='ab_experiment.zip or extracted folder')
    ap.add_argument('--output', default='evaluation/results/spatial_coverage_audit')
    args = ap.parse_args()
    src = read_data(args.input)
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)

    reasons = Counter()
    outcomes = Counter()
    per_code = defaultdict(Counter)
    unsupported_rows = []
    for image_id in sorted(src['gt_labels']):
        g = src['gt_labels'][image_id]
        y = src['yolo_predictions'][image_id]
        sp = src['spatial_judgments'][image_id]
        reason = sp.get('reason_code') or '(none)'
        reasons[reason] += 1
        if reason != 'rule_not_supported':
            continue
        result = decisions(y, sp)
        a_dec, a_codes = result['A']
        d_dec, d_codes = result['D']
        gt_hazard = bool(g['is_hazard'])
        gt_code = (g.get('hazard_code') or '').strip()
        # Use *predicted* check code, not GT or image_id, to identify rule coverage gaps.
        check_codes = sorted({c['hazard_code'] for c in (sp.get('checks') or []) if c.get('hazard_code')})
        code_key = ';'.join(check_codes) if check_codes else '(no_check_code)'
        a_correct = (a_dec == 'normal' and not gt_hazard) or (
            a_dec == 'hazard' and gt_hazard and gt_code in a_codes
        )
        cat = ('A_correct_to_review' if a_correct and d_dec == 'review'
               else 'A_error_to_review' if not a_correct and d_dec == 'review'
               else 'D_' + d_dec)
        outcomes[cat] += 1
        per_code[code_key]['n'] += 1
        per_code[code_key]['gt_hazard'] += int(gt_hazard)
        per_code[code_key]['gt_normal'] += int(not gt_hazard)
        per_code[code_key]['A_correct'] += int(a_correct)
        per_code[code_key]['A_error'] += int(not a_correct)
        per_code[code_key]['D_review'] += int(d_dec == 'review')
        per_code[code_key]['D_hazard'] += int(d_dec == 'hazard')
        per_code[code_key]['D_normal'] += int(d_dec == 'normal')
        unsupported_rows.append({
            'image_id': image_id, 'rule_check_codes': code_key,
            'spatial_status': sp.get('evaluation_status'), 'reason_code': reason,
            'yolo_predicted_codes': ';'.join(sorted(a_codes)),
            'A_decision': a_dec, 'D_decision': d_dec,
            'gt_is_hazard_for_audit_only': gt_hazard,
            'gt_code_for_audit_only': gt_code,
            'A_exact_correct_for_audit_only': a_correct,
            'outcome': cat,
        })
    totals = []
    for code, counts in sorted(per_code.items(), key=lambda pair: (-pair[1]['n'], pair[0])):
        totals.append({'rule_check_codes':code, **{k:counts[k] for k in (
            'n','gt_hazard','gt_normal','A_correct','A_error','D_review','D_hazard','D_normal')}})

    write_csv(dest/'unsupported_cases.csv', unsupported_rows, [
        'image_id','rule_check_codes','spatial_status','reason_code',
        'yolo_predicted_codes','A_decision','D_decision',
        'gt_is_hazard_for_audit_only','gt_code_for_audit_only',
        'A_exact_correct_for_audit_only','outcome'])
    write_csv(dest/'unsupported_by_code.csv', totals, [
        'rule_check_codes','n','gt_hazard','gt_normal','A_correct','A_error','D_review','D_hazard','D_normal'])
    write_csv(dest/'spatial_reason_counts.csv', [
        {'reason_code':r,'n':n} for r,n in reasons.most_common()],['reason_code','n'])

    print('Total images:',len(src['gt_labels']))
    print('Spatial reason codes:',dict(reasons))
    print('Unsupported rule images:',len(unsupported_rows))
    print('Unsupported outcomes:',dict(outcomes))
    print('\nRule-check code | n | GT hazard | GT normal | A correct | A error | D review')
    for row in totals:
        print(' | '.join(str(row[k]) for k in [
            'rule_check_codes','n','gt_hazard','gt_normal','A_correct','A_error','D_review']))
    print('Saved:',dest)
    print('NOTE: This is a test-set diagnostic. Do not develop rules from its GT and re-score on the same test set.')

if __name__ == '__main__':
    main()
