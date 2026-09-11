# So sánh TinyGRU P16 cũ và H48x2 cap-0.25 mới

Model cũ là `tinygru-p16-v1`; model mới là
`tinygru-p16-h48x2-cap025-v1`. Random Forest trong `libtraffic_ai.so` không
đổi. Bảng này chỉ nói về nhánh TinyGRU/P16.

| Hạng mục | P16 cũ | H48x2 cap-0.25 mới | Ảnh hưởng tích hợp |
| --- | --- | --- | --- |
| Model ID | `tinygru-p16-v1` | `tinygru-p16-h48x2-cap025-v1` | Log ID để tránh nạp nhầm release. |
| Số input / ABI | 2: `[1,90,3] float32`, `[1,90] int32` | Không đổi | App vẫn truyền 90 packet và một remote IPv4. |
| Numeric features | `log1p(length), log1p(IAT), direction`; z-score cả 3 | Không đổi về ý nghĩa/thứ tự | App không tự normalize. Mean/std được SDK mới thay đúng theo model. |
| Direction | `+1` LAN→WAN, `-1` WAN→LAN | Không đổi | App tiếp tục truyền direction theo quy ước này. |
| P16 | global-unicast remote IPv4 `/16`; ID 0 là UNK | Không đổi | Một remote chung cho 90 packet vẫn hợp lệ. |
| GRU input | Numeric 3 + P16 embedding 8 = 11; early fusion | Numeric 3; late fusion | Không thay input app, nhưng không được dùng TFLite cũ với SDK/profile mới. |
| GRU | hidden 64, 2 layers | hidden 48, 2 layers | Model mới nhỏ hơn. |
| P16 embedding | dim 8, đưa vào từng timestep GRU | dim 2, average-pool các ID khác UNK | Tín hiệu IP mới chỉ điều chỉnh logits cuối. |
| IP contribution | Học chung bên trong GRU | `0.25 * tanh(ip_to_logits(mean_embedding))` | Ảnh hưởng IP bị giới hạn biên ±0.25 logit. |
| Inference parameters | khoảng 45.5k | 24,731 | Giảm gần 46% parameter. |
| Window | 90 packet, skip 10, stride 90 | 90 packet, skip 0, stride 90 | Đây là thay đổi deployment quan trọng nhất: window đầu mới là packet 1–90. |
| Output | raw logits `[1,5]` | raw logits `[1,5]` | Class order vẫn: Background, Game, RTVideo, Voice, VStream. |
| Confidence policy | không có ngưỡng accept trong profile cũ | softmax temperature 1.0; accept nếu confidence ≥ 0.98 | Dùng `result.accepted` trước khi tự động áp nhãn mới. |

## Giá trị normalization

| Channel | P16 cũ mean / std | H48x2 mới mean / std |
| --- | --- | --- |
| `log1p_packet_length` | `6.0075965 / 1.3676786` | `6.0075564 / 1.3676575` |
| `log1p_nonnegative_iat` | `0.0020584022 / 0.021586655` | `0.0020678083 / 0.021813218` |
| `direction` | `-0.16327207 / 0.9865821` | `-0.16352206 / 0.9865407` |

Không copy mean/std cũ sang model mới. `traffic_p16_default_config()` trong
release đã chứa đúng constants của model H48x2.

## API cần dùng

App của anh Kiên tiếp tục gọi:

```c
traffic_p16_predict_directional_packets(
    packets, TRAFFIC_P16_WINDOW_SIZE, remote_global_ipv4,
    traffic_p16_default_config(), &result);
```

Sau khi gọi thành công, kiểm tra `result.label`, `result.confidence`, và
`result.accepted`. Không cần đổi format 90×3; chỉ cần đổi policy lấy window từ
skip-10 sang không skip, đồng thời deploy đồng bộ `libtraffic_ai.so` và
`libtinygru_h48x2_cap025.so` của cùng archive này.
