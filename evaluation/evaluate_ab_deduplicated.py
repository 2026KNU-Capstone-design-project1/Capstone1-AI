"""Re-evaluate illustrative A/B/C/D decision policies after removing near-duplicate test IDs.
Does not train or change model/project files. Reads data/ab_experiment.zip and data/test_duplicates.zip.
"""
import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path('data/ab_experiment.zip')
DUP = Path('data/test_duplicates.zip')
OUT = Path('evaluation/results/ab_deduplicated')
CONF = 0.5
AREA = 0.01


def read_json_items(z, name):
    matches = [p for p in z.namelist() if p.endswith('/' + name + '.json') or p == name + '.json']
    if len(matches) != 1:
        raise ValueError(f'{name}: expected one JSON, found {matches}')
    data = json.loads(z.read(matches[0]))
    return data['items']


def indexed(rows):
    out = {}
    for row in rows:
        k = row['image_id']
        if k in out:
            raise ValueError('duplicate image_id: ' + k)
        out[k] = row
    return out


def read_duplicate_ids(z):
    matches = [p for p in z.namelist() if p.endswith('duplicate_test_ids_in_1256.txt')]
    if len(matches) != 1:
        raise ValueError(f'duplicate ID file not uniquely found: {matches}')
    ids = [x.strip() for x in z.read(matches[0]).decode('utf-8-sig').splitlines() if x.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate IDs appear more than once')
    return set(ids)


def good_quality(p):
    box = p.get('bbox') or [0, 0, 0, 0]
    area = p.get('area_ratio')
    if area is None:
        area = float(box[2]) * float(box[3])
    return float(p.get('confidence') or 0) >= CONF and float(area) >= AREA


def abcd(yo, sp):
    pred = yo.get('predictions') or []
    high = [p for p in pred if float(p.get('confidence') or 0) >= CONF]
    good = [p for p in pred if good_quality(p)]
    findings = sp.get('findings') or []
    codes_b = {f.get('hazard_code') for f in findings if f.get('hazard_code') and f.get('status') in ('위험', 'hazard')}
    status = sp.get('evaluation_status')
    a = ('hazard', {p['hazard_code'] for p in high}) if high else ('normal', set())
    b = ('hazard', codes_b) if codes_b else (('review', set()) if status == 'not_evaluable' else ('normal', set()))
    c = ('hazard', {p['hazard_code'] for p in good}) if good else (('review', set()) if pred else ('normal', set()))
    overlap = codes_b & {p['hazard_code'] for p in good}
    d = ('hazard', overlap) if overlap else (('review', set()) if (codes_b or pred or status == 'not_evaluable') else ('normal', set()))
    return {'A': a, 'B': b, 'C': c, 'D': d}


def evaluate(ids, gt, yo, sp):
    sums = {}
    details = []
    for model in 'ABCD':
        cnt = Counter()
        for image_id in sorted(ids):
            truth = gt[image_id]
            gt_hazard = bool(truth['is_hazard'])
            gt_code = truth.get('hazard_code') or ''
            decision, codes = abcd(yo[image_id], sp[image_id])[model]
            if decision == 'review':
                cnt['review'] += 1
                cnt['review_hazard' if gt_hazard else 'review_normal'] += 1
            else:
                cnt['covered'] += 1
                if gt_hazard:
                    cnt['covered_hazard'] += 1
                    if decision == 'hazard' and gt_code in codes:
                        cnt['tp_exact'] += 1
                    else:
                        cnt['fn_covered'] += 1
                    if decision == 'hazard' and gt_code not in codes:
                        cnt['wrong_code_hazard'] += 1
                else:
                    cnt['covered_normal'] += 1
                    if decision == 'hazard':
                        cnt['fp_normal'] += 1
                    else:
                        cnt['tn'] += 1
            details.append({'image_id': image_id, 'model': model, 'gt_hazard': gt_hazard, 'gt_code': gt_code,
                            'decision': decision, 'predicted_codes': ';'.join(sorted(codes))})
        n = len(ids)
        tp, fp, fn = cnt['tp_exact'], cnt['fp_normal'], cnt['fn_covered']
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        total_hazard = sum(bool(gt[i]['is_hazard']) for i in ids)
        sums[model] = {'model': model, 'n_images': n, 'total_hazard': total_hazard,
                       'coverage': cnt['covered'] / n if n else 0,
                       'covered': cnt['covered'], 'review': cnt['review'], 'review_hazard': cnt['review_hazard'],
                       'review_normal': cnt['review_normal'], 'tp_exact': tp, 'fp_normal': fp,
                       'fn_covered': fn, 'tn': cnt['tn'], 'wrong_code_hazard': cnt['wrong_code_hazard'],
                       'precision_covered': prec, 'recall_covered': rec, 'f1_covered': f1,
                       'fpr_covered': fp / cnt['covered_normal'] if cnt['covered_normal'] else 0,
                       'total_hazard_recall': tp / total_hazard if total_hazard else 0}
    return sums, details


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    if not BASE.exists() or not DUP.exists():
        raise SystemExit('필요한 파일을 찾지 못했습니다. data/ab_experiment.zip 과 data/test_duplicates.zip 위치를 확인하세요.')
    with zipfile.ZipFile(BASE) as z:
        gt = indexed(read_json_items(z, 'gt_labels'))
        yo = indexed(read_json_items(z, 'yolo_predictions'))
        sp = indexed(read_json_items(z, 'spatial_judgments'))
    with zipfile.ZipFile(DUP) as z:
        duplicate_ids = read_duplicate_ids(z)
    if not (gt.keys() == yo.keys() == sp.keys()):
        raise ValueError('prediction/GT image_id sets differ')
    if len(gt) != 1256:
        raise ValueError(f'Expected 1256 test IDs, found {len(gt)}')
    missing = duplicate_ids - gt.keys()
    if missing:
        raise ValueError(f'duplicate list IDs not in 1256 evaluation: {sorted(missing)[:5]}')
    if len(duplicate_ids) != 53:
        raise ValueError(f'Expected 53 duplicate IDs, found {len(duplicate_ids)}')
    dup_hazard = sum(bool(gt[i]['is_hazard']) for i in duplicate_ids)
    print(f'중복 ID 확인: {len(duplicate_ids)}장 (위험 {dup_hazard}, 정상 {len(duplicate_ids) - dup_hazard})')
    keep = gt.keys() - duplicate_ids
    full, full_details = evaluate(gt.keys(), gt, yo, sp)
    dedup, dedup_details = evaluate(keep, gt, yo, sp)
    OUT.mkdir(parents=True, exist_ok=True)
    comparison = []
    for key in 'ABCD':
        x, y = full[key], dedup[key]
        comparison.extend([{'set': 'full_1256', **x}, {'set': 'dedup_1203', **y}])
    write_csv(OUT / 'comparison.csv', comparison)
    write_csv(OUT / 'dedup_details.csv', dedup_details)
    write_csv(OUT / 'full_details_recomputed.csv', full_details)
    print('\n제외 전후 정책 비교 (기준: 원본 코드 일치 TP, 정상 이미지 위험 예측 FP)')
    print(f"{'정책':4s} {'세트':12s} {'Cov':>7s} {'F1(확정)':>10s} {'전체위험R':>10s} {'TP':>5s} {'FP':>5s} {'Review':>7s}")
    for key in 'ABCD':
        for label, src in [('전체 1256', full), ('제외 1203', dedup)]:
            r = src[key]
            print(f"{key:4s} {label:12s} {r['coverage']:7.2%} {r['f1_covered']:10.2%} {r['total_hazard_recall']:10.2%} {r['tp_exact']:5d} {r['fp_normal']:5d} {r['review']:7d}")
    # Reference baseline numbers previously published from project's original evaluation.
    expected = {'A': (465, 16, 110), 'B': (380, 13, 84), 'C': (465, 16, 88), 'D': (322, 3, 33)}
    for model, (tp, fp, fn) in expected.items():
        actual = full[model]
        if (actual['tp_exact'], actual['fp_normal'], actual['fn_covered']) != (tp, fp, fn):
            print(f"주의: 원본 스크립트의 {model} TP/FP/FN와 차이: "
                  f"현재 {(actual['tp_exact'],actual['fp_normal'],actual['fn_covered'])}, 기준 {(tp,fp,fn)}. "
                  "이 경우 comparison.csv 를 논문에 직접 사용하기 전에 정책 차이를 점검하세요.")
    print('\n저장:', OUT / 'comparison.csv')
    print('주의: 유사 이미지 제외는 선정된 53장에 대한 민감도 분석이며, 남은 1203장이 완전 독립임을 증명하지는 않습니다.')


if __name__ == '__main__':
    main()
