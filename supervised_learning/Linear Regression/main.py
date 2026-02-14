import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = np.random.rand(50, 1)

y = 3.5 * x + np.random.randn(50, 1) * 20

model = LinearRegression()
model.fit(x, y)

y_pred = model.predict(x)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, color="blue", label="Data Points")
plt.plot(x, y_pred, color="red", linewidth=2, label="Regression Line")
plt.title("Linear Regression on Random Dataset")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

print("Slope (Coefficient):", model.coef_[0][0])
print("Intercept:", model.intercept_[0])
