import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# 1. create 2 classes of 2D points
# ============================================================
np.random.seed(2)

# center values for 2 classes
mean0 = [2, 2]   # +1
mean1 = [4, 2]   # -1

# co-variance
cov = [[0.3, 0.2], 
       [0.2, 0.3]]

N = 5  # number of points per class

# create random points for each class following Gaussian distribution
# N rows, 2 columns (each row is a point x, y)
class0 = np.random.multivariate_normal(mean0, cov, N)
class1 = np.random.multivariate_normal(mean1, cov, N)

# convert to shape (2 rows, N columns) for easier processing
X0 = class0.T
X1 = class1.T

# combine the two classes into one dataset
# X_points shape = (2, 40)
X_points = np.concatenate((X0, X1), axis=1)

# labels for the two classes: +1 for class0, -1 for class1
# y shape = (1, 40)
y_pos = np.ones((1, N))
y_neg = -1 * np.ones((1, N))
y = np.concatenate((y_pos, y_neg), axis=1)

# add a row of ones to X_points to create X_bar for bias term
# X shape = (3, 40): row 0 = 1, row 1 = x, row 2 = y
ones_row = np.ones((1, 2 * N))
X = np.concatenate((ones_row, X_points), axis=0)


# ============================================================
# 2. functions for PLA
# ============================================================
def sign_of(value):
    """Giong np.sign: > 0 -> 1, < 0 -> -1, = 0 -> 0."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def predict_one(w, x):
    """
    Du doan nhan cua 1 diem.
    w: vector trong so, shape (3, 1)
    x: 1 diem [1, x, y], shape (3, 1)

    Tinh tich vo huong: w0*1 + w1*x + w2*y
    roi lay dau.
    """
    s = 0.0
    for k in range(len(x)):
        s = s + w[k, 0] * x[k, 0]
    return sign_of(s)


def predict_all(w, X_data):
    """Du doan nhan tat ca cac diem."""
    n_points = X_data.shape[1]
    preds = []
    for i in range(n_points):
        xi = X_data[:, i].reshape(3, 1)
        preds.append(predict_one(w, xi))
    return preds


def has_converged(X_data, y_data, w):
    """True neu moi diem deu duoc phan loai dung."""
    preds = predict_all(w, X_data)
    n_points = X_data.shape[1]
    for i in range(n_points):
        if preds[i] != y_data[0, i]:
            return False
    return True


def perceptron(X_data, y_data, w_init):
    """
    Perceptron Learning Algorithm (PLA).

    Moi vong:
    - xao tron thu tu cac diem
    - neu diem bi sai: w_new = w_old + y * x
    Lap den khi khong con diem sai.
    """
    w_history = [w_init]
    mis_points = []
    n_points = X_data.shape[1]

    while True:
        # xao tron thu tu diem
        order = np.random.permutation(n_points)

        for i in range(n_points):
            idx = order[i]
            xi = X_data[:, idx].reshape(3, 1)
            yi = y_data[0, idx]

            pred = predict_one(w_history[-1], xi)
            if pred != yi:
                # diem nay dang bi phan loai sai -> cap nhat w
                mis_points.append(idx)
                w_new = w_history[-1] + yi * xi
                w_history.append(w_new)

        if has_converged(X_data, y_data, w_history[-1]):
            break

    return w_history, mis_points


# ============================================================
# 3. Chay PLA
# ============================================================
d = X.shape[0]  # = 3
w_init = np.random.randn(d, 1)
w_history, mis_points = perceptron(X, y, w_init)
print(mis_points)


# ============================================================
# 4. Ve duong thang phan tach
# ============================================================
def draw_line(w):
    """Ve doan thang w0 + w1*x + w2*y = 0 trong cua so do thi."""
    w0, w1, w2 = [float(v) for v in w.ravel()]

    if w2 != 0:
        x_left = 0.0
        x_right = 6.0
        y_left = -(w1 * x_left + w0) / w2
        y_right = -(w1 * x_right + w0) / w2
        return plt.plot([x_left, x_right], [y_left, y_right], "k-")

    # duong dung neu w2 = 0
    x_vert = -w0 / w1
    return plt.plot([x_vert, x_vert], [-2.0, 4.0], "k-")


# ============================================================
# 5. Animation: moi frame la 1 lan cap nhat w
# ============================================================
def viz_alg_1d_2(w_list):
    n_iter = len(w_list)
    plt.figure()

    def update(i):
        plt.cla()
        plt.plot(X0[0, :], X0[1, :], "b^", label="+1")
        plt.plot(X1[0, :], X1[1, :], "ro", label="-1")
        plt.xlim(0, 6)
        plt.ylim(-2, 4)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.legend()

        draw_line(w_list[i])

        # khoanh diem dang bi sua, tru frame cuoi
        if i < n_iter - 1:
            px = float(X[1, mis_points[i]])
            py = float(X[2, mis_points[i]])
            plt.plot(px, py, "ko", markersize=12, fillstyle="none")

        plt.title("PLA: buoc %d/%d" % (i, n_iter - 1))

    anim = FuncAnimation(
        plt.gcf(), update, frames=n_iter, interval=800, repeat=False
    )
    anim.save("pla_vis.gif", writer="pillow")
    plt.show()


viz_alg_1d_2(w_history)
