import numpy as np

def normalize_data(x):
    range_x = np.max(x) - np.min(x)
    if range_x == 0:
        return np.zeros_like(x)
    return (x - np.min(x)) / range_x

def gradient_descent(x1, x2, x3, y, iterations=2000, learning_rate=0.01):
    a1_curr = 0
    a2_curr = 0
    a3_curr = 0
    b_curr = 0
    n = len(x1)
    
    for i in range(iterations):
        y_predicted = a1_curr * x1 + a2_curr * x2 + a3_curr * x3 + b_curr

        error = np.sum((y - y_predicted)**2)
        
        a1d = -(2/n) * np.sum(x1 * (y - y_predicted))
        a2d = -(2/n) * np.sum(x2 * (y - y_predicted))
        a3d = -(2/n) * np.sum(x3 * (y - y_predicted))
        bd = -(2/n) * np.sum(y - y_predicted)

        a1_curr = a1_curr - learning_rate * a1d
        a2_curr = a2_curr - learning_rate * a2d
        a3_curr = a3_curr - learning_rate * a3d
        b_curr = b_curr - learning_rate * bd
        
        if i % 100 == 0:
            print(f"Iteration {i}: a1={a1_curr:.4f}, a2={a2_curr:.4f}, a3={a3_curr:.4f}, b={b_curr:.4f}, error={error:.4f}")
            
    return a1_curr, a2_curr, a3_curr, b_curr


#Data Asli
#x1 x2 x3  y
#30 9  90 92%
#20 9  80 70%
#25 3  90 85%
#27 6  85 ...

#Final Result: y = -0.0211x1 + 0.0843x2 + 0.2441x3 + 0.6146


x1 = np.array([30, 20, 25])
x2 = np.array([9, 9, 3])
x3 = np.array([90, 80, 90])
y = np.array([0.92, 0.70, 0.85])

#Normalisasi Data
x1 = normalize_data(x1)
x2 = normalize_data(x2)
x3 = normalize_data(x3)

a1, a2, a3, b = gradient_descent(x1, x2, x3, y)
print(f"\nFinal Result: y = {a1:.4f}x1 + {a2:.4f}x2 + {a3:.4f}x3 + {b:.4f}")

result = a1 * x1 + a2 * x2 + a3 * x3 + b
print(f"Predicted values: {result}")
print(f"Actual values: {y}")
print(f"Akurasi: {100 - np.mean(np.abs(result - y) / y) * 100:.2f}%")

print(f"data baris ke-4: x1=27, x2=6, x3=85")
x1_new = np.array([30, 20, 25, 27])
x2_new = np.array([9, 9, 3, 6])
x3_new = np.array([90, 80, 90, 85])
x1_new = normalize_data(x1_new)
x2_new = normalize_data(x2_new)
x3_new = normalize_data(x3_new)
result_new = a1 * x1_new + a2 * x2_new + a3 * x3_new + b
print(f"Predicted value: {result_new[-1]}")