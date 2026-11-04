
from risk_pipeline import evaluate_hazard_risk
from rag_test import search_criteria, generate_assessment


def run_assessment(
    hazard_type,
    scene,
    risk_evidence,
    query
):
    """
    규칙 판정 → 위험도 산정 → RAG 검색 → LLM 보고서 생성
    """

    # 1. 규칙 기반 위반 판정 및 위험도 산정
    risk_result = evaluate_hazard_risk(
        hazard_type,
        scene,
        risk_evidence
    )

    # 2. 관련 안전 기준 RAG 검색
    search_results = search_criteria(
        query,
        top_k=5,
        method="char"
    )

    # 3. 규칙 결과와 검색 근거를 함께 전달
    llm_scene = {
        "observations": scene,
        "risk_assessment": risk_result,
        "retrieved_criteria": search_results
    }

    # 4. Qwen3:4B 위험성평가 보고서 생성
    report = generate_assessment(
        scene=scene,
        rule_result=risk_result["rule_result"],
        risk_assessment=risk_result["risk_assessment"],
        search_results=search_results,
        return_json=True
    )

    return {
        "rule_result": risk_result["rule_result"],
        "risk_assessment": risk_result["risk_assessment"],
        "retrieved_criteria": search_results,
        "report": report
    }
