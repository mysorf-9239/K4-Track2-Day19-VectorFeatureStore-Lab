"""
HybridMemoryAgent: episodic memory (Qdrant) + user profile (Feast).

Ý tưởng chính (chi tiết xem bonus/ARCHITECTURE.md):
  - remember(): cắt text theo câu (tối đa ~60 từ/chunk), embed, rồi upsert vào
    1 collection Qdrant chung, mỗi point có payload user_id.
  - recall(): lấy profile + hoạt động gần đây từ Feast, search hybrid
    (BM25 + vector, RRF k=60, rank bắt đầu từ 1) chỉ trong memory của user đó,
    rồi ghép lại thành 1 đoạn context. Không gọi LLM.
"""
from __future__ import annotations

import re
import sys
import time
import unicodedata
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qdrant_client import QdrantClient, models  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

from app.embeddings import Embedder  # noqa: E402

COLLECTION = "episodic_memory"
MAX_WORDS = 60
RRF_K = 60
PROFILE_FEATURES = [
    "user_profile_features:reading_speed_wpm",
    "user_profile_features:preferred_language",
    "user_profile_features:topic_affinity",
    "query_velocity_features:queries_last_hour",
    "query_velocity_features:distinct_topics_24h",
]


def strip_accents(text):
    # "tự động" -> "tu dong", để user gõ không dấu vẫn match được
    nfd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn").replace("đ", "d")


def tokenize(text):
    # index cả bản có dấu lẫn không dấu của mỗi token
    toks = re.findall(r"\w+", unicodedata.normalize("NFC", text).lower())
    return toks + [strip_accents(t) for t in toks]


def chunk(text, max_words=MAX_WORDS):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    chunks, cur = [], []
    for s in sentences:
        if cur and len(" ".join(cur + [s]).split()) > max_words:
            chunks.append(" ".join(cur))
            cur = []
        cur.append(s)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


class HybridMemoryAgent:
    def __init__(self, feast_repo=ROOT / "app" / "feast_repo", top_k=3):
        self.embedder = Embedder()
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=self.embedder.dim, distance=models.Distance.COSINE),
        )
        self.top_k = top_k
        self.texts = {}  # user_id -> list of (point_id, text), dùng cho BM25
        self.store = None
        if feast_repo is not None and (feast_repo / "registry.db").exists():
            try:
                from feast import FeatureStore
                self.store = FeatureStore(repo_path=str(feast_repo))
            except Exception as e:
                print(f"[agent] không load được Feast, chạy không có profile: {e}")

    def remember(self, text, user_id="u_001"):
        pieces = chunk(text)
        vectors = list(self.embedder.embed(pieces))
        points = []
        for piece, vec in zip(pieces, vectors):
            pid = str(uuid.uuid4())
            points.append(models.PointStruct(
                id=pid,
                vector=vec.tolist(),
                payload={"user_id": user_id, "text": piece, "ts": time.time()},
            ))
            self.texts.setdefault(user_id, []).append((pid, piece))
        self.client.upsert(COLLECTION, points=points)

    def get_profile(self, user_id):
        if self.store is None:
            return {}
        try:
            raw = self.store.get_online_features(
                features=PROFILE_FEATURES, entity_rows=[{"user_id": user_id}]
            ).to_dict()
        except Exception as e:
            print(f"[agent] lookup Feast lỗi: {e}")
            return {}
        return {k: v[0] for k, v in raw.items() if k != "user_id"}

    def search(self, query, user_id):
        mine = self.texts.get(user_id, [])
        if not mine:
            return []
        depth = max(self.top_k * 5, 20)

        # vector search, filter theo user_id (đây là ranh giới privacy)
        qv = next(self.embedder.embed([query])).tolist()
        user_filter = models.Filter(must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
        ])
        hits = self.client.query_points(COLLECTION, query=qv, limit=depth, query_filter=user_filter).points
        sem_ids = [str(p.id) for p in hits]

        # BM25 chỉ trên memory của user này
        scores = BM25Okapi([tokenize(t) for _, t in mine]).get_scores(tokenize(query))
        order = sorted(range(len(mine)), key=lambda i: -scores[i])[:depth]
        kw_ids = [mine[i][0] for i in order if scores[i] > 0]

        # RRF, rank bắt đầu từ 1
        rrf = {}
        for ranked in (kw_ids, sem_ids):
            for rank, pid in enumerate(ranked, start=1):
                rrf[pid] = rrf.get(pid, 0.0) + 1.0 / (RRF_K + rank)
        text_of = dict(mine)
        best = sorted(rrf.items(), key=lambda kv: -kv[1])[:self.top_k]
        return [text_of[pid] for pid, _ in best]

    def recall(self, query, user_id="u_001"):
        p = self.get_profile(user_id)
        memories = self.search(query, user_id)

        lines = [f"[user {user_id}] query: {query}"]
        if p:
            lines.append(f"Profile: thích chủ đề '{p.get('topic_affinity')}', ngôn ngữ "
                         f"'{p.get('preferred_language')}', đọc khoảng {p.get('reading_speed_wpm')} wpm.")
            lines.append(f"Recent activity: {p.get('queries_last_hour')} query trong 1 giờ qua, "
                         f"{p.get('distinct_topics_24h')} chủ đề trong 24h.")
        else:
            lines.append("Profile: (chưa có, cần chạy NB4 để materialize Feast)")

        if memories:
            lines.append("Top memories:")
            lines += [f"  {i}. {m}" for i, m in enumerate(memories, 1)]
        else:
            lines.append("Top memories: (không tìm thấy)")
        return "\n".join(lines)
