"""Safely integrate unverified law RAG retrieval into Capstone1-AI/main.py.

Run at repo root: python apply_law_rag.py
Creates a timestamped backup; no endpoint input/output schema changes.
Do not claim legal citations verified until separately checked against official law.
"""
from pathlib import Path
from datetime import datetime
import ast
import shutil

MAIN = Path('main.py')
LAW = Path('law_rag.py')
DATA = Path('data/law_articles.json')
if not all(p.exists() for p in (MAIN, LAW, DATA)):
    raise SystemExit('Missing main.py, law_rag.py or data/law_articles.json; run at project root.')

original = MAIN.read_text(encoding='utf-8')
if 'retrieved_law_candidates' in original:
    raise SystemExit('main.py already contains law integration. No changes made.')

search_anchor = '        scene = {\n'
retrieval = '''        # Unverified law excerpts support retrieval/audit only, never legal conclusions.
        from law_rag import search_law_articles
        law_candidates = [
            {
                "id": law.get("id"),
                "law_name": law.get("law_name"),
                "article_number": law.get("article_number"),
                "article_title": law.get("article_title"),
                "article_excerpt": (law.get("article_text") or "")[:600],
                "source_pages": law.get("source_pages", []),
                "score": law.get("score"),
                "verification_status": law.get("verification_status", "unverified"),
                "official_source_url": law.get("official_source_url"),
                "effective_date": law.get("effective_date"),
            }
            for law in search_law_articles(query, top_k=3)
        ]

'''
criteria_anchor = '            "retrieved_criteria": search_results,\n'
criteria_insert = '            "retrieved_law_candidates": law_candidates,\n'
old_gate = '''        if unverified_knowledge:
            report["법적사항"] = "관련 법령 추가 확인 필요"
'''
new_gate = '''        # Deterministic legal provenance gate: LLM output is never enough.
        # Verified law must have an official reference and applicability date.
        verified_laws = [
            law for law in law_candidates
            if law.get("verification_status") == "verified"
            and law.get("official_source_url")
            and law.get("effective_date")
        ]
        verified_code_basis = [
            r for r in search_results
            if isinstance(r, dict)
            and r.get("retrieval_method") == "hazard_code_exact"
            and r.get("verification_status") == "verified"
            and r.get("legal_basis")
        ]
        if verified_laws:
            report["법적사항"] = "관련 법령(현장 적용·위반 여부 별도 확인): " + "; ".join(
                f'{law["law_name"]} {law["article_number"]} ({law["article_title"]})'
                for law in verified_laws
            )
        elif verified_code_basis:
            report["법적사항"] = "관련 법령(현장 적용·위반 여부 별도 확인): " + "; ".join(
                str(r["legal_basis"]) for r in verified_code_basis
            )
        else:
            report["법적사항"] = "관련 법령 추가 확인 필요"
'''
checks = [
    (search_anchor, 1, 'scene assignment'),
    (criteria_anchor, 1, 'retrieved_criteria field'),
    (old_gate, 1, 'old legal gate'),
]
for anchor, count, label in checks:
    found = original.count(anchor)
    if found != count:
        raise SystemExit(f'Unexpected {label}: expected {count}, got {found}. No changes made.')

modified = original.replace(search_anchor, retrieval + search_anchor, 1)
modified = modified.replace(criteria_anchor, criteria_anchor + criteria_insert, 1)
modified = modified.replace(old_gate, new_gate, 1)
try:
    ast.parse(modified, filename='main.py')
except SyntaxError as exc:
    raise SystemExit(f'Patched main.py fails syntax check: {exc}. No changes made.')

backup = MAIN.with_name(f'main.py.bak_law_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
shutil.copy2(MAIN, backup)
MAIN.write_text(modified, encoding='utf-8')
print('Law retrieval added to scene: retrieved_law_candidates')
print('Legal statement guarded by verified provenance (otherwise requires review)')
print('Backup:', backup)
print('Updated:', MAIN)
