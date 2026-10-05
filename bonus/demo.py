"""
Demo 5 query cho HybridMemoryAgent.

Chạy từ thư mục gốc repo (nên chạy NB4 trước để Feast có dữ liệu):
    .venv/bin/python bonus/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent import HybridMemoryAgent  # noqa: E402

USER = "u_001"

MEMORIES = [
    "Hôm nay tôi đọc tài liệu về Kubernetes. Pod là đơn vị triển khai nhỏ nhất. "
    "Deployment quản lý ReplicaSet và hỗ trợ rolling update.",
    "Ghi chú: Horizontal Pod Autoscaler tự động mở rộng số pod theo CPU và lưu lượng "
    "người dùng. Cluster Autoscaler thêm node khi pod bị pending.",
    "Tôi đã đọc bài về cloud security: nguyên tắc least privilege với IAM, mã hoá dữ liệu "
    "at rest bằng KMS, và bật audit log cho mọi tài khoản.",
    "Hỏi trợ lý cách tối ưu chi phí hạ tầng cloud: dùng spot instance cho batch job, "
    "reserved instance cho workload ổn định.",
    "Đọc xong chương về vector database: Qdrant dùng HNSW, hybrid search kết hợp BM25 "
    "và vector bằng Reciprocal Rank Fusion.",
    "Mình đang tìm hiểu feature store, Feast làm point-in-time join để tránh data leakage.",
]

# memory của user khác, không được phép hiện ra khi u_001 hỏi
OTHER_USER = "u_002"
OTHER_MEMORY = "Ghi chú riêng của u_002: lịch họp review Kubernetes với khách hàng."

QUERIES = [
    ("chỉ cần vector", "Tôi đã đọc gì về Kubernetes?"),
    ("cần profile", "Recommend đọc gì tiếp"),
    ("cần recent activity", "Tôi đang quan tâm gì gần đây?"),
    ("paraphrase", "Tài liệu về tự động mở rộng hạ tầng?"),
    ("mixed", "Cho tôi summary cloud security"),
]


def main():
    agent = HybridMemoryAgent()
    for m in MEMORIES:
        agent.remember(m, user_id=USER)
    agent.remember(OTHER_MEMORY, user_id=OTHER_USER)

    for i, (kind, q) in enumerate(QUERIES, 1):
        print(f"\n=== Query {i} ({kind}) ===")
        ctx = agent.recall(q, user_id=USER)
        print(ctx)
        if OTHER_MEMORY in ctx:
            raise SystemExit("lỗi: lộ memory của user khác")

    print("\nXong 5 query, không có memory nào của u_002 bị lộ.")


if __name__ == "__main__":
    main()
