import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn import datasets, neighbors
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def myweight1(distances):
    return 1 / (distances + 1e-8)


def myweight2(distances):
    sigma2 = 0.4
    return np.exp(-(distances**2) / sigma2)


# Phần lõi: tải dữ liệu Iris và chia thành tập train/test.
np.random.seed(7)
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    iris_X,
    iris_y,
    test_size=130,
    random_state=7,
)


WEIGHT_OPTIONS = {
    "uniform": "uniform",
    "distance": "distance",
    "myweight1": myweight1,
    "myweight2": myweight2,
}

CLASS_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]
REGION_COLORS = ListedColormap(["#f9d6d2", "#d8f3df", "#d9e8fb"])


class KNNIrisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("KNN - Iris Flower Classification")
        self.root.geometry("1180x760")
        self.root.minsize(980, 650)

        self.model = None
        self.y_pred = None
        self.animation_job = None
        self.neighbor_artists = []
        self.is_running = False
        self.plot_x_index = 0
        self.plot_y_index = 1

        self.k_var = tk.IntVar(value=7)
        self.weight_var = tk.StringVar(value="uniform")
        self.x_feature_var = tk.StringVar(value=iris.feature_names[0])
        self.y_feature_var = tk.StringVar(value=iris.feature_names[1])
        self.speed_var = tk.IntVar(value=250)
        self.status_var = tk.StringVar(
            value="Chọn tham số rồi nhấn 'Chạy phân loại'."
        )
        self.accuracy_var = tk.StringVar(value="Accuracy: --")
        self.progress_var = tk.StringVar(
            value=f"Train: {len(X_train)} mẫu | Test: {len(X_test)} mẫu"
        )
        self.neighbor_var = tk.StringVar(value="Khoảng cách lân cận: --")

        self._build_interface()
        self._draw_dataset_preview()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_interface(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        control_frame = ttk.Frame(self.root, padding=18)
        control_frame.grid(row=0, column=0, sticky="nsew")
        control_frame.columnconfigure(0, weight=1)

        ttk.Label(
            control_frame,
            text="KNN CONTROL",
            font=("Segoe UI", 16, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 18))

        ttk.Label(control_frame, text="Số láng giềng K:").grid(
            row=1, column=0, sticky="w"
        )
        self.k_spinbox = ttk.Spinbox(
            control_frame,
            from_=1,
            to=len(X_train),
            textvariable=self.k_var,
            width=24,
        )
        self.k_spinbox.grid(row=2, column=0, sticky="ew", pady=(4, 14))

        ttk.Label(control_frame, text="Kiểu trọng số (weights):").grid(
            row=3, column=0, sticky="w"
        )
        self.weight_combo = ttk.Combobox(
            control_frame,
            textvariable=self.weight_var,
            values=list(WEIGHT_OPTIONS),
            state="readonly",
            width=22,
        )
        self.weight_combo.grid(row=4, column=0, sticky="ew", pady=(4, 14))

        ttk.Separator(control_frame).grid(
            row=5, column=0, sticky="ew", pady=(2, 14)
        )

        ttk.Label(control_frame, text="Trục X:").grid(row=6, column=0, sticky="w")
        self.x_feature_combo = ttk.Combobox(
            control_frame,
            textvariable=self.x_feature_var,
            values=iris.feature_names,
            state="readonly",
            width=22,
        )
        self.x_feature_combo.grid(row=7, column=0, sticky="ew", pady=(4, 12))

        ttk.Label(control_frame, text="Trục Y:").grid(row=8, column=0, sticky="w")
        self.y_feature_combo = ttk.Combobox(
            control_frame,
            textvariable=self.y_feature_var,
            values=iris.feature_names,
            state="readonly",
            width=22,
        )
        self.y_feature_combo.grid(row=9, column=0, sticky="ew", pady=(4, 14))

        self.x_feature_combo.bind("<<ComboboxSelected>>", self._feature_changed)
        self.y_feature_combo.bind("<<ComboboxSelected>>", self._feature_changed)

        ttk.Label(control_frame, text="Tốc độ hiển thị:").grid(
            row=10, column=0, sticky="w"
        )
        ttk.Scale(
            control_frame,
            from_=100,
            to=3000,
            variable=self.speed_var,
            orient="horizontal",
        ).grid(row=11, column=0, sticky="ew", pady=(4, 14))
        ttk.Label(
            control_frame,
            text="Nhanh  ←                         →  Chậm",
            foreground="#666666",
            font=("Segoe UI", 8),
        ).grid(row=12, column=0, sticky="ew", pady=(0, 14))

        self.run_button = ttk.Button(
            control_frame,
            text="▶ Chạy phân loại",
            command=self.run_classification,
        )
        self.run_button.grid(row=13, column=0, sticky="ew", pady=4)

        self.stop_button = ttk.Button(
            control_frame,
            text="■ Dừng",
            command=self.stop_animation,
            state="disabled",
        )
        self.stop_button.grid(row=14, column=0, sticky="ew", pady=4)

        ttk.Button(
            control_frame,
            text="↻ Đặt lại",
            command=self.reset_view,
        ).grid(row=15, column=0, sticky="ew", pady=4)

        ttk.Separator(control_frame).grid(
            row=16, column=0, sticky="ew", pady=(16, 12)
        )
        ttk.Label(
            control_frame,
            textvariable=self.accuracy_var,
            font=("Segoe UI", 14, "bold"),
            foreground="#175a9c",
        ).grid(row=17, column=0, sticky="w")
        ttk.Label(control_frame, textvariable=self.progress_var).grid(
            row=18, column=0, sticky="w", pady=(6, 4)
        )
        ttk.Label(
            control_frame,
            textvariable=self.status_var,
            wraplength=235,
            justify="left",
        ).grid(row=19, column=0, sticky="nw", pady=(6, 0))

        ttk.Label(
            control_frame,
            textvariable=self.neighbor_var,
            wraplength=235,
            justify="left",
            foreground="#555555",
        ).grid(row=20, column=0, sticky="nw", pady=(8, 0))

        ttk.Label(
            control_frame,
            text=(
                "Model học bằng đủ 4 đặc trưng. Nền màu là lát cắt "
                "2D. Đường nối được chiếu từ khoảng cách 4D; đường "
                "dày hơn có trọng số lớn hơn."
            ),
            wraplength=235,
            justify="left",
            foreground="#666666",
            font=("Segoe UI", 9, "italic"),
        ).grid(row=21, column=0, sticky="sw", pady=(18, 0))

        chart_frame = ttk.Frame(self.root, padding=(0, 12, 12, 12))
        chart_frame.grid(row=0, column=1, sticky="nsew")
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar_frame = ttk.Frame(chart_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew")
        NavigationToolbar2Tk(self.canvas, toolbar_frame).update()

    def _selected_feature_indices(self):
        x_index = iris.feature_names.index(self.x_feature_var.get())
        y_index = iris.feature_names.index(self.y_feature_var.get())
        if x_index == y_index:
            raise ValueError("Trục X và trục Y phải là hai đặc trưng khác nhau.")
        return x_index, y_index

    def _feature_changed(self, _event=None):
        if self.is_running:
            return
        try:
            self._selected_feature_indices()
        except ValueError:
            return
        self.model = None
        self.accuracy_var.set("Accuracy: --")
        self.status_var.set("Đã đổi trục. Nhấn 'Chạy phân loại' để chạy lại.")
        self._draw_dataset_preview()

    def _format_axes(self, x_index, y_index, title):
        self.ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
        self.ax.set_xlabel(iris.feature_names[x_index])
        self.ax.set_ylabel(iris.feature_names[y_index])
        self.ax.grid(alpha=0.2)

    def _draw_dataset_preview(self):
        try:
            x_index, y_index = self._selected_feature_indices()
        except ValueError:
            return

        self.ax.clear()
        self.neighbor_artists = []
        for class_id, class_name in enumerate(iris.target_names):
            mask = iris_y == class_id
            self.ax.scatter(
                iris_X[mask, x_index],
                iris_X[mask, y_index],
                c=CLASS_COLORS[class_id],
                label=class_name,
                s=48,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.7,
            )

        self._format_axes(x_index, y_index, "Dữ liệu Iris theo lớp thực tế")
        self.ax.legend(title="Loài hoa")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _draw_decision_view(self):
        self.ax.clear()
        self.neighbor_artists = []
        x_index = self.plot_x_index
        y_index = self.plot_y_index

        x_values = iris_X[:, x_index]
        y_values = iris_X[:, y_index]
        x_margin = max((x_values.max() - x_values.min()) * 0.08, 0.2)
        y_margin = max((y_values.max() - y_values.min()) * 0.08, 0.2)

        xx, yy = np.meshgrid(
            np.linspace(x_values.min() - x_margin, x_values.max() + x_margin, 180),
            np.linspace(y_values.min() - y_margin, y_values.max() + y_margin, 180),
        )

        # Tạo lát cắt 2D; hai đặc trưng bị ẩn được giữ ở giá trị trung bình.
        grid_data = np.tile(X_train.mean(axis=0), (xx.size, 1))
        grid_data[:, x_index] = xx.ravel()
        grid_data[:, y_index] = yy.ravel()
        grid_prediction = self.model.predict(grid_data).reshape(xx.shape)

        self.ax.contourf(
            xx,
            yy,
            grid_prediction,
            levels=[-0.5, 0.5, 1.5, 2.5],
            cmap=REGION_COLORS,
            alpha=0.65,
        )

        for class_id, class_name in enumerate(iris.target_names):
            mask = y_train == class_id
            self.ax.scatter(
                X_train[mask, x_index],
                X_train[mask, y_index],
                c=CLASS_COLORS[class_id],
                label=f"Train: {class_name}",
                marker="o",
                s=70,
                edgecolors="#222222",
                linewidths=0.8,
                zorder=3,
            )

        self.ax.scatter(
            X_test[:, x_index],
            X_test[:, y_index],
            c="#aeb6bf",
            label="Test: chưa phân loại",
            marker="x",
            s=32,
            alpha=0.75,
            zorder=2,
        )

        wrong_handle = Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#e00034",
            markeredgewidth=2,
            label="Viền đỏ: dự đoán sai",
        )
        neighbor_handle = Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=2,
            label="Đường nối K láng giềng",
        )
        handles, labels = self.ax.get_legend_handles_labels()
        handles.append(wrong_handle)
        labels.append("Viền đỏ: dự đoán sai")
        handles.append(neighbor_handle)
        labels.append("Đường nối K láng giềng")
        self.ax.legend(handles, labels, title="Kết quả", loc="best", fontsize=8)

        self._format_axes(
            x_index,
            y_index,
            f"Decision regions - {self.k_var.get()}KNN / {self.weight_var.get()}",
        )
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _clear_neighbor_connections(self):
        for artist in self.neighbor_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self.neighbor_artists = []

    def _neighbor_weights_for_display(self, distances):
        selected_weight = self.weight_var.get()
        if selected_weight == "uniform":
            raw_weights = np.ones_like(distances)
        elif selected_weight in ("distance", "myweight1"):
            raw_weights = 1 / (distances + 1e-8)
        else:
            raw_weights = myweight2(distances)

        max_weight = raw_weights.max()
        relative_weights = raw_weights / max_weight if max_weight > 0 else raw_weights
        return raw_weights, relative_weights

    def _draw_neighbor_connections(self, test_index):
        self._clear_neighbor_connections()

        distances, neighbor_indices = self.model.kneighbors(
            X_test[test_index : test_index + 1],
            return_distance=True,
        )
        distances = distances[0]
        neighbor_indices = neighbor_indices[0]
        raw_weights, relative_weights = self._neighbor_weights_for_display(distances)

        test_x = X_test[test_index, self.plot_x_index]
        test_y = X_test[test_index, self.plot_y_index]

        for distance, train_index, relative_weight in zip(
            distances, neighbor_indices, relative_weights
        ):
            neighbor_class = int(y_train[train_index])
            neighbor_x = X_train[train_index, self.plot_x_index]
            neighbor_y = X_train[train_index, self.plot_y_index]

            line = self.ax.plot(
                [test_x, neighbor_x],
                [test_y, neighbor_y],
                color=CLASS_COLORS[neighbor_class],
                linewidth=0.7 + 3.3 * float(relative_weight),
                alpha=0.2 + 0.75 * float(relative_weight),
                zorder=5,
            )[0]
            neighbor_ring = self.ax.scatter(
                neighbor_x,
                neighbor_y,
                facecolors="none",
                edgecolors=CLASS_COLORS[neighbor_class],
                s=90 + 45 * float(relative_weight),
                linewidths=1.2 + 1.4 * float(relative_weight),
                zorder=6,
            )
            self.neighbor_artists.extend([line, neighbor_ring])

        nearest_details = []
        for distance, train_index, weight in zip(
            distances[:3], neighbor_indices[:3], raw_weights[:3]
        ):
            class_name = iris.target_names[int(y_train[train_index])]
            nearest_details.append(
                f"{class_name}: d={distance:.2f}, w={weight:.3g}"
            )
        self.neighbor_var.set("3 láng giềng gần nhất: " + " | ".join(nearest_details))

    def run_classification(self):
        self.stop_animation(update_status=False)

        try:
            k = int(self.k_var.get())
            if not 1 <= k <= len(X_train):
                raise ValueError(f"K phải nằm trong khoảng 1 đến {len(X_train)}.")
            self.plot_x_index, self.plot_y_index = self._selected_feature_indices()
        except (ValueError, tk.TclError) as error:
            messagebox.showerror("Tham số không hợp lệ", str(error))
            return

        selected_weight = self.weight_var.get()
        weight_function = WEIGHT_OPTIONS[selected_weight]

        # Phần lõi: tạo model, huấn luyện và dự đoán bằng KNN.
        self.model = neighbors.KNeighborsClassifier(
            n_neighbors=k,
            p=2,
            weights=weight_function,
        )
        self.model.fit(X_train, y_train)
        self.y_pred = self.model.predict(X_test)

        self.accuracy_var.set("Accuracy: đang tính...")
        self.progress_var.set(f"Đã phân loại: 0/{len(X_test)}")
        self.neighbor_var.set("Khoảng cách lân cận: đang tính...")
        self.status_var.set("Đang phân loại từng mẫu test...")
        self._draw_decision_view()

        self.is_running = True
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._animate_prediction(0)

    def _animate_prediction(self, index):
        if not self.is_running:
            return

        predicted_class = int(self.y_pred[index])
        actual_class = int(y_test[index])
        is_correct = predicted_class == actual_class

        self._draw_neighbor_connections(index)

        self.ax.scatter(
            X_test[index, self.plot_x_index],
            X_test[index, self.plot_y_index],
            c=CLASS_COLORS[predicted_class],
            marker="s",
            s=54,
            edgecolors="#222222" if is_correct else "#e00034",
            linewidths=0.8 if is_correct else 2.2,
            zorder=4,
        )

        processed = index + 1
        running_accuracy = accuracy_score(
            y_test[:processed], self.y_pred[:processed]
        )
        self.progress_var.set(f"Đã phân loại: {processed}/{len(X_test)}")
        self.accuracy_var.set(f"Accuracy hiện tại: {running_accuracy * 100:.2f}%")
        self.status_var.set(
            f"Mẫu {processed}: dự đoán {iris.target_names[predicted_class]} | "
            f"thực tế {iris.target_names[actual_class]} | "
            f"{'đúng' if is_correct else 'sai'}"
        )
        self.canvas.draw_idle()

        if processed < len(X_test):
            self.animation_job = self.root.after(
                int(self.speed_var.get()),
                self._animate_prediction,
                processed,
            )
        else:
            final_accuracy = accuracy_score(y_test, self.y_pred)
            self.is_running = False
            self.animation_job = None
            self.run_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            self.accuracy_var.set(f"Accuracy: {final_accuracy * 100:.2f}%")
            self.status_var.set(
                f"Hoàn tất với {self.k_var.get()}KNN, "
                f"weights={self.weight_var.get()}."
            )

    def stop_animation(self, update_status=True):
        if self.animation_job is not None:
            self.root.after_cancel(self.animation_job)
            self.animation_job = None
        was_running = self.is_running
        self.is_running = False
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        if update_status and was_running:
            self.status_var.set("Đã dừng mô phỏng. Nhấn chạy để bắt đầu lại.")

    def reset_view(self):
        self.stop_animation(update_status=False)
        self.model = None
        self.y_pred = None
        self.accuracy_var.set("Accuracy: --")
        self.progress_var.set(
            f"Train: {len(X_train)} mẫu | Test: {len(X_test)} mẫu"
        )
        self.neighbor_var.set("Khoảng cách lân cận: --")
        self.status_var.set("Chọn tham số rồi nhấn 'Chạy phân loại'.")
        self._draw_dataset_preview()

    def _on_close(self):
        self.stop_animation(update_status=False)
        self.root.destroy()


if __name__ == "__main__":
    app_root = tk.Tk()
    KNNIrisGUI(app_root)
    app_root.mainloop()
