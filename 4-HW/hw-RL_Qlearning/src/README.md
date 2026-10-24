# Đồ án: Robot tự học đường đi bằng Q-learning (Python + Pygame)

Môi trường GridWorld 8x8 có tường (`#`), xuất phát (`S`) và đích (`G`).
Robot dùng **tabular Q-learning** để học đường đi tới G trong môi trường xác định.

## Cấu trúc

```text
qlearning_robot_project/
├── gridworld.py       # Bản đồ, hành động, reward, trạng thái tiếp theo
├── qlearning.py       # Q-table, epsilon-greedy, cập nhật Q, train
├── main.py            # Giao diện Pygame, nút Train/Play/Reset
├── plot_results.py    # Vẽ reward, success rate ra PNG
├── verify_training.py # Kiểm chứng không cần Pygame
├── requirements.txt
└── README.md
```

## Cài và chạy

Yêu cầu Python 3.10+.

```bash
cd qlearning_robot_project
python -m pip install -r requirements.txt
python main.py
```

Nếu dùng macOS/Linux và lệnh `python` không hoạt động, thử `python3`.

Trên GUI:
- **TRAIN (T):** train 5.000 episode; nhấn lần nữa để tạm dừng/tiếp tục.
- **PLAY (P):** robot chạy theo Q-value lớn nhất, đường đi tô vàng.
- **RESET (R):** xóa Q-table và học lại.
- **ESC:** thoát.

Khi train xong, file `training_log.csv` sẽ được tạo ở thư mục bạn chạy `python main.py`.
Vẽ kết quả:

```bash
python plot_results.py
```

Hai ảnh `reward_curve.png`, `success_curve.png` được lưu cùng thư mục.

Kiểm tra thuật toán (không cần mở GUI):

```bash
python verify_training.py
```

## Cách tính Q-value

`Q[s, a]` chứa ước lượng giá trị dài hạn khi robot ở trạng thái `s` và chọn hành động `a`.

```text
target = reward + gamma * max(Q[next_state])  # nếu chưa tới đích
Q[s, a] = Q[s, a] + alpha * (target - Q[s, a])
```

Nếu `next_state` là trạng thái đích, `target = reward` (không cộng phần tương lai).

## Thiết kế

- State: `s = row * 8 + col`, từ `0` tới `63` (bao gồm chỉ số của các ô tường, tuy nhiên robot không đi được vào đó).
- Action: `0=UP`, `1=RIGHT`, `2=DOWN`, `3=LEFT`.
- Reward: tới đích `+100`; bước hợp lệ `-1`; đi vào tường/ngoài bản đồ `-10` và đứng yên.
- Learning rate: `alpha=0.15`.
- Discount: `gamma=0.95`.
- Khám phá: epsilon-greedy, bắt đầu `1.0`, giảm nhân `0.998`/episode tới sàn `0.05`.
- Mỗi episode: chọn ngẫu nhiên một ô tự do, không phải đích; dừng khi tới đích hoặc sau 160 bước.
- Train `5000` episode; mỗi frame xử lý tối đa `14` episode để UI vẫn nhận tương tác.
- Khi đánh giá: **không** khám phá; chọn hành động có Q lớn nhất.

Đường đi ngắn nhất theo BFS chỉ dùng để **đánh giá**, không dùng để điều khiển robot hoặc huấn luyện.

## Gợi ý mở rộng

1. Thêm lựa chọn thay đổi tham số alpha/gamma/epsilon trên giao diện.
2. Cho người dùng vẽ/xóa tường bằng chuột rồi train lại.
3. Thử map 10x10 và so sánh số episode hội tụ.
4. So sánh learned path với BFS hoặc A* về số bước và thời gian tính.
5. Trộn thêm nhiễu chuyển động, ví dụ robot có 10% xác suất trượt sang ô khác.

## Lưu ý

Môi trường là bản đồ tĩnh, tất định; Q-learning vẫn sử dụng được trong môi trường này.
Một trận học có thể không tối ưu với mọi random seed; thử train thêm hoặc điều chỉnh tham số.
