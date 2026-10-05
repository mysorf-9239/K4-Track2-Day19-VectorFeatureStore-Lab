# Reflection: Lab 19

**Tên:** Nguyễn Đức Danh
**MSHV:** 2A202602722
**Cohort:** A20-K4
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

Trung bình Precision@10: hybrid 78.6%, BM25 77.8%, vector 73.2%.

- **exact:** BM25 và hybrid hoà nhau ở 96.7%, vector 88.7%. Query chứa đúng từ khoá trong tài liệu nên BM25 gần như không sai, RRF giữ được thứ hạng đó.
- **mixed:** hybrid thắng tuyệt đối 100%, vector 98.5%, BM25 97%. Mỗi retriever bắt một nửa câu hỏi, RRF cộng điểm cho doc đứng cao ở cả hai list.
- **paraphrase:** khác với kỳ vọng, vector không thắng (24%), BM25 33.3%, hybrid 32%. Lý do là bge-small-en-v1.5 là model tiếng Anh, câu diễn đạt lại thuần Việt bị embed kém. Muốn vector thắng ở đây phải đổi sang bge-m3 hoặc multilingual-e5.

Khi nào không dùng hybrid: tra mã lỗi, SKU, tên hàm, log ID thì dùng BM25 thuần, vừa chính xác vừa nhanh (P99 khoảng 2ms so với khoảng 40ms của hybrid). Corpus đa ngôn ngữ hoặc query toàn ngôn ngữ tự nhiên, cộng với model embedding tốt cho ngôn ngữ đó, thì vector thuần là đủ và đỡ phải duy trì 2 index.

---

## Điều ngạc nhiên nhất khi làm lab này

Ở NB8, join "latest value" thay vì point-in-time làm AUC tăng ảo +0.12 với 98% dòng bị rò, nhìn số rất đẹp nhưng không dùng được lúc serving.

---

## Bonus challenge

- [x] Đã làm bonus (xem `bonus/`)
- [ ] Pair work: không, làm một mình
