from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "risk_data_clean.csv"

df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

search_colums = [
    "대분류", "위험요소", "판단항목", "정상", "위반", "예상 위험", "개선내용", "안전대책", "법적사항"
]

missing = [col for col in search_colums if col not in df.columns]
if missing:
    raise ValueError(f"csv에 없는 column: {missing}")

df[search_colums] = df[search_colums].fillna("").astype(str)
documents = df[search_colums].agg(" ".join, axis=1).tolist()

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
)

document_vectors = vectorizer.fit_transform(documents)

def search_criteria(query, top_k=5):
    """현장 상황과 관련 있는 기준을 상위 top_k개 검색한다."""
    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector, document_vectors
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue

        row = df.iloc[idx]

        results.append({
            "기준_행": int(idx),
            "유사도": round(float(scores[idx]), 4),
            **{col: row[col] for col in search_colums},
        })

    return results

if __name__ == "__main__":
    scene = (
        "일자가 지난 위험성평가서가 다수 배치되어 있다."
    )

    results = search_criteria(scene)

    print(f"전체 기준 수: {len(df)}")
    print(f"검색 결과 수: {len(results)}")

    for i, item in enumerate(results, start=1):
        print(f"\n검색 결과 {i}")   
        print(f"유사도: {item['유사도']}")

        for col in search_colums:
            print(f"{col}: {item[col]}")