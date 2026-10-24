"""Giai va ve quy dao da thuc bac 5."""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. KHAI BAO DE BAI
# ============================================================
# Thoi gian bat dau va ket thuc
t0 = 0.0
tf = 1.0

# Vi tri q tai luc dau va luc cuoi (don vi: do)
q0 = 10.0
qf = 45.0

# Van toc q' tai luc dau va luc cuoi
q_dot_0 = 0.0
q_dot_f = 0.0

# Gia toc q'' tai luc dau va luc cuoi
q_2dot_0 = 0.0
q_2dot_f = 0.0


# ============================================================
# 2. LAP HE PHUONG TRINH DE TIM a0, a1, ..., a5
# ============================================================
# Da thuc bac 5:
# q(t)   = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
#
# Dao ham bac 1:
# q'(t)  = a1 + 2*a2*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4
#
# Dao ham bac 2:
# q''(t) = 2*a2 + 6*a3*t + 12*a4*t^2 + 20*a5*t^3

# Ma tran A gom 6 phuong trinh:
# q(t0), q'(t0), q''(t0), q(tf), q'(tf), q''(tf)
A = np.array(
    [
        [1, t0, t0**2, t0**3, t0**4, t0**5],
        [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],
        [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],
        [1, tf, tf**2, tf**3, tf**4, tf**5],
        [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],
        [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3],
    ],
    dtype=float,
)

# Cot ket qua b cua 6 phuong trinh
b = np.array(
    [q0, q_dot_0, q_2dot_0, qf, q_dot_f, q_2dot_f],
    dtype=float,
)

# Giai he A * a = b de tim 6 he so
a = np.linalg.solve(A, b)

# May tinh co the cho ra 349.9999999999994 thay vi 350.
# Lam tron 10 chu so sau dau phay de ket qua dep va de doc hon.
a = np.round(a, 10)
a[np.abs(a) < 1e-10] = 0

a0 = a[0]
a1 = a[1]
a2 = a[2]
a3 = a[3]
a4 = a[4]
a5 = a[5]

print("Cac he so tim duoc:")
print("a0 =", a0)
print("a1 =", a1)
print("a2 =", a2)
print("a3 =", a3)
print("a4 =", a4)
print("a5 =", a5)
print()
print("q(t) = ", a0, " + ", a1, "*t + ", a2, "*t^2 + ", a3, "*t^3 + ", a4, "*t^4 + ", a5, "*t^5")


# ============================================================
# 3. TINH q, q', q'', q''' at many times
# ============================================================
# Tao 501 gia tri t tu 0 den 1
t = np.linspace(t0, tf, 501)

# z - position
q = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5

# v - velocity
q_dot = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4

# a - acceleration
q_2dot = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3

# Jerk
q_3dot = 6*a3 + 24*a4*t + 60*a5*t**2


# ============================================================
# 4. VE BIEU DO
# ============================================================
plt.figure(figsize=(11, 7))

# Bieu do q(t)
plt.subplot(2, 2, 1)
plt.plot(t, q, color="red", label="q(t)")
plt.title("Vi tri q(t)")
plt.ylabel("q [deg]")
plt.grid(True)
plt.legend()

# Bieu do q'(t)
plt.subplot(2, 2, 2)
plt.plot(t, q_dot, color="green", label="q'(t)")
plt.title("Van toc q'(t)")
plt.ylabel("q' [deg/s]")
plt.grid(True)
plt.legend()

# Bieu do q''(t)
plt.subplot(2, 2, 3)
plt.plot(t, q_2dot, color="blue", label="q''(t)")
plt.title("Gia toc q''(t)")
plt.xlabel("Thoi gian t [s]")
plt.ylabel("q'' [deg/s^2]")
plt.grid(True)
plt.legend()

# Bieu do q'''(t)
plt.subplot(2, 2, 4)
plt.plot(t, q_3dot, color="magenta", label="q'''(t)")
plt.title("Jerk q'''(t)")
plt.xlabel("Thoi gian t [s]")
plt.ylabel("q''' [deg/s^3]")
plt.grid(True)
plt.legend()

plt.suptitle("Quy dao da thuc bac 5")
plt.tight_layout()

# Luu bieu do thanh anh trong thu muc hien tai
plt.savefig("quintic_trajectory.png", dpi=160)

# Hien thi bieu do
plt.show()
