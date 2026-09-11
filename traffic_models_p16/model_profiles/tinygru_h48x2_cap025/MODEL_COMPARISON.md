# So sánh TinyGRU P16 cũ và H48x2 cap-0.25 mới

Model cũ là `tinygru-p16-v1`; model mới là
`tinygru-p16-h48x2-cap025-v1`. Random Forest trong `libtraffic_ai.so` không
đổi. Bảng này chỉ so sánh nhánh TinyGRU/P16.

| Hạng mục | P16 cũ | H48x2 cap-0.25 mới | Ý nghĩa triển khai |
| --- | --- | --- | --- |
| Input application mỗi lần inference | 90 packet | 90 packet | **Không đổi API app:** truyền timestamp, length, direction cho 90 packet và một remote IPv4 global dùng chung. |
| Numeric features | `log1p(length)`, `log1p(IAT)`, `direction` | Giữ nguyên | SDK tự tính IAT, transform và normalize; app không tự normalize. |
| P16 | Remote IPv4 `/16`, ID `0` là UNK | Giữ nguyên; vocabulary 142 ID | IP không hợp lệ/chưa gặp vẫn chạy qua UNK. |
| Kiến trúc kết hợp IP | **Early fusion**: embedding IP ở từng timestep đi vào GRU | **Late fusion**: GRU chỉ xử lý traffic; IP chỉ hiệu chỉnh logits cuối | Model mới ít phụ thuộc vào IP hơn. |
| P16 embedding | 8 chiều | 2 chiều, average-pool trên window | Giảm footprint và chi phí tính toán. |
| GRU | 2 layer, hidden 64; input 11D | 2 layer, hidden 48; input 3D | Nhánh traffic của model mới nhẹ hơn. |
| Ảnh hưởng IP | Học trực tiếp trong trạng thái GRU | `0.25 × tanh(...)`, giới hạn ±0.25 mỗi logit | IP không thể lấn át tín hiệu traffic. |
| Inference parameters | khoảng 45.5k | 24,731 | Giảm gần 46% số tham số. |
| GRU compute | khoảng 3.5M MACs/window | khoảng 1.905M MACs/window | Model mới phù hợp hơn với CPU nhúng. |
| Direction khi train | private/public heuristic | Ưu tiên inferred device endpoint, rồi fallback private/public | Khi deploy, ưu tiên app truyền direction router-authoritative: `+1` LAN→WAN, `-1` WAN→LAN. |
| Training balance | Weighted loss; session dài có thể đóng góp nhiều window | Cân bằng class, giới hạn 512 unique window/session/epoch, không replacement trong epoch | Giảm bias từ class hoặc session dài. |
| Output/class order | logits `[1,5]`: Background, Game, RTVideo, Voice, VStream | Giữ nguyên | Không đổi mapping label P16. |
| Confidence policy | Model luôn trả prediction | Model luôn trả prediction; SDK đánh dấu `accepted` khi confidence ≥ 0.98 | `0.98` là policy SDK/app, không nằm bên trong model. |

## Ý nghĩa của late fusion

```text
P16 cũ: numeric(3) + P16 embedding(8) → GRU H64×2 → 5 logits
Mới:    numeric(3) → GRU H48×2 → traffic logits
         P16 IDs → embedding(2) → average pool → capped IP correction
         traffic logits + IP correction → 5 logits
```

Với model mới, nếu remote IP lạ hoặc không có thì P16 map sang `UNK=0` và
phần hiệu chỉnh IP suy giảm; nhánh GRU dựa trên traffic vẫn hoạt động độc lập.

## Normalization và đóng gói

Không copy mean/std của model cũ sang model mới. Release chứa đúng vocabulary
và normalization của H48x2 trong `traffic_p16_default_config()`. Luôn deploy
đồng bộ `libtraffic_ai.so` và `libtinygru_h48x2_cap025.so` từ cùng archive.

## Đánh giá và phạm vi checkpoint deploy

Kiến trúc H48×2 cap-0.25 được chọn qua các thí nghiệm session-disjoint trước
khi chốt kiến trúc. Checkpoint trong release là checkpoint **final**, train lại
trên toàn bộ dataset sau khi kiến trúc và preprocessing đã freeze. Vì vậy,
không gán held-out accuracy/F1 từ experiment cũ cho checkpoint này; các file
`TRAIN_ONLY_*` chỉ là sanity check. Xác thực cuối cùng cần dùng fresh captures
chưa tham gia train hoặc model selection.
