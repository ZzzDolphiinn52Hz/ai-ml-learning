#!/usr/bin/env python3
"""
PLA từng bước — ví dụ 4 điểm đã tính tay.

Chạy:
    python pla_step_demo.py

Nút:
    Bước tiếp  — sửa đúng 1 điểm sai (điểm có chỉ số nhỏ nhất)
    Reset      — về w = 0
    Chạy hết   — lặp đến khi hội tụ
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.widgets import Button
from matplotlib.patches import FancyBboxPatch

# Ưu tiên font có dấu tiếng Việt
for _name in ("Noto Sans", "DejaVu Sans", "Arial", "Segoe UI"):
    try:
        font_manager.findfont(_name, fallback_to_default=False)
        plt.rcParams["font.family"] = _name
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

# ----- data -----
X = np.array([
    [1.0, 2.0, 4.0, 5.0],
    [1.0, 3.0, 1.0, 2.0],
], dtype=float)
y = np.array([1.0, 1.0, -1.0, -1.0])
NAMES = ["x1 (1,1) +", "x2 (2,3) +", "x3 (4,1) −", "x4 (5,2) −"]

N = X.shape[1]
X_bar = np.vstack([np.ones(N), X])  # (3, N)


def predict(w):
    s = w @ X_bar
    pred = np.sign(s)
    pred[pred == 0] = 1.0
    return s, pred


def misclassified(w):
    _, pred = predict(w)
    return np.where(pred != y)[0]


class PLAStepper:
    def __init__(self):
        self.reset_state()

        plt.close("all")
        self.fig = plt.figure(figsize=(11.2, 6.4), facecolor="#f7f7f4")
        self.ax = self.fig.add_axes([0.07, 0.16, 0.50, 0.76])
        self.ax_info = self.fig.add_axes([0.60, 0.16, 0.37, 0.76])
        self.ax_info.set_axis_off()

        self.ax_btn_step = self.fig.add_axes([0.10, 0.035, 0.16, 0.07])
        self.ax_btn_all = self.fig.add_axes([0.28, 0.035, 0.16, 0.07])
        self.ax_btn_reset = self.fig.add_axes([0.46, 0.035, 0.16, 0.07])

        self.btn_step = Button(self.ax_btn_step, "Bước tiếp", color="#dbeafe", hovercolor="#93c5fd")
        self.btn_all = Button(self.ax_btn_all, "Chạy hết", color="#dcfce7", hovercolor="#86efac")
        self.btn_reset = Button(self.ax_btn_reset, "Reset", color="#fee2e2", hovercolor="#fca5a5")
        self.btn_step.on_clicked(self.on_step)
        self.btn_all.on_clicked(self.on_run_all)
        self.btn_reset.on_clicked(self.on_reset)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.suptitle(
            "PLA tung buoc — Space: buoc tiep, A: chay het, R: reset",
            fontsize=13,
            fontweight="bold",
            y=0.97,
        )
        self.draw()

    def reset_state(self):
        self.w = np.zeros(3)
        self.t = 0
        self.last_fixed = None
        self.done = False
        self.log = ["t = 0 | w = [0, 0, 0] | moi diem score = 0 nen pred = +1"]

    def do_one_step(self):
        if self.done:
            return
        mis = misclassified(self.w)
        if len(mis) == 0:
            self.done = True
            self.last_fixed = None
            self.log.append("Hội tụ: không còn điểm sai.")
            return
        i = int(mis[0])
        before = self.w.copy()
        self.w = self.w + y[i] * X_bar[:, i]
        self.last_fixed = i
        self.t += 1
        sign = "+" if y[i] > 0 else "−"
        self.log.append(
            f"t = {self.t} | sua diem {NAMES[i]} | "
            f"w := [{before[0]:.0f}, {before[1]:.0f}, {before[2]:.0f}] "
            f"{sign} [{X_bar[0, i]:.0f}, {X_bar[1, i]:.0f}, {X_bar[2, i]:.0f}] "
            f"= [{self.w[0]:.0f}, {self.w[1]:.0f}, {self.w[2]:.0f}]"
        )
        if len(misclassified(self.w)) == 0:
            self.done = True
            self.log.append("Hội tụ: không còn điểm sai.")

    def on_step(self, _event=None):
        self.do_one_step()
        self.draw()

    def on_run_all(self, _event=None):
        guard = 0
        while not self.done and guard < 50:
            self.do_one_step()
            guard += 1
        self.draw()

    def on_reset(self, _event=None):
        self.reset_state()
        self.draw()

    def on_key(self, event):
        if event.key in (" ", "right", "n"):
            self.on_step()
        elif event.key in ("r", "R"):
            self.on_reset()
        elif event.key in ("a", "A"):
            self.on_run_all()

    def boundary_points(self, w, x_min, x_max, y_min, y_max):
        w0, w1, w2 = w
        if abs(w1) < 1e-12 and abs(w2) < 1e-12:
            return None
        if abs(w2) >= 1e-12:
            xs = np.linspace(x_min, x_max, 200)
            ys = -(w0 + w1 * xs) / w2
            return xs, ys
        x_vert = -w0 / w1
        return np.array([x_vert, x_vert]), np.array([y_min, y_max])

    def draw(self):
        ax = self.ax
        ax.clear()
        ax.set_xlim(0.2, 5.8)
        ax.set_ylim(0.2, 3.8)
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.28)
        ax.set_facecolor("#ffffff")

        scores, pred = predict(self.w)
        mis = set(misclassified(self.w).tolist())

        # nửa mặt dương / âm nếu có boundary
        xx, yy = np.meshgrid(np.linspace(0.2, 5.8, 200), np.linspace(0.2, 3.8, 160))
        zz = self.w[0] + self.w[1] * xx + self.w[2] * yy
        if not (abs(self.w[1]) < 1e-12 and abs(self.w[2]) < 1e-12):
            ax.contourf(xx, yy, zz, levels=[-1e9, 0, 1e9], colors=["#fee2e2", "#dbeafe"], alpha=0.45)
            line = self.boundary_points(self.w, 0.2, 5.8, 0.2, 3.8)
            if line is not None:
                ax.plot(line[0], line[1], color="#111827", lw=2.4, label=r"$w^\top \bar{x} = 0$")

        # điểm
        for i in range(N):
            is_pos = y[i] > 0
            color = "#2563eb" if is_pos else "#dc2626"
            marker = "s" if is_pos else "o"
            ax.scatter(
                X[0, i], X[1, i],
                c=color, marker=marker, s=160,
                zorder=5, edgecolors="white", linewidths=1.2,
            )
            if i in mis:
                ax.scatter(
                    X[0, i], X[1, i],
                    facecolors="none", edgecolors="#f59e0b",
                    s=340, linewidths=2.4, zorder=6,
                )
            if self.last_fixed == i:
                ax.scatter(
                    X[0, i], X[1, i],
                    facecolors="none", edgecolors="#111827",
                    s=420, linewidths=2.0, zorder=7,
                )
            ax.annotate(
                f"x{i+1}\n{int(scores[i]) if float(scores[i]).is_integer() else f'{scores[i]:.1f}'}",
                (X[0, i], X[1, i]),
                textcoords="offset points",
                xytext=(8, 8),
                fontsize=8,
                color="#111827",
            )

        title = f"t = {self.t}   w = [{self.w[0]:.0f}, {self.w[1]:.0f}, {self.w[2]:.0f}]"
        if self.done:
            title += "   | hoi tu"
        elif self.t == 0:
            title += "   (chưa có đường; mọi pred = +1)"
        ax.set_title(title, loc="left", fontsize=11)
        if abs(self.w[1]) > 1e-12 or abs(self.w[2]) > 1e-12:
            ax.legend(loc="upper left", framealpha=0.92, fontsize=8)

        # panel phải
        info = self.ax_info
        info.clear()
        info.set_xlim(0, 1)
        info.set_ylim(0, 1)
        info.set_axis_off()
        box = FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor="#ffffff", edgecolor="#d1d5db", linewidth=1.0,
        )
        info.add_patch(box)

        lines = [
            "Bảng điểm hiện tại",
            "",
        ]
        header = f"{'i':<4}{'điểm':<12}{'y':>4}{'score':>8}{'pred':>7}{'':>6}"
        lines.append(header)
        lines.append("-" * 40)
        for i in range(N):
            mark = "SAI" if i in mis else "ok"
            lines.append(
                f"{i+1:<4}({X[0, i]:.0f},{X[1, i]:.0f})     "
                f"{int(y[i]):+d}{scores[i]:8.1f}{int(pred[i]):+7d}   {mark}"
            )
        lines += [
            "",
            f"Số điểm sai: {len(mis)}",
            "",
            "Nhật ký cập nhật",
            "",
        ]
        # show last few log lines
        shown = self.log[-8:]
        lines.extend(shown)
        if self.done:
            lines += [
                "",
                "Đường:  1 - 2 x1 + x2 = 0",
                "hay     x2 = 2 x1 - 1",
            ]

        text = "\n".join(lines)
        info.text(
            0.07, 0.95, text,
            va="top", ha="left",
            family="Noto Sans",
            fontsize=8.4,
            color="#111827",
            transform=info.transAxes,
        )
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    app = PLAStepper()
    plt.show()
