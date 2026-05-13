import numpy as np

def normalize_data(x):
    range_x = np.max(x) - np.min(x)
    if range_x == 0:
        return np.zeros_like(x)
    return (x - np.min(x)) / range_x

def gradient_descent_optimal_lr(X, y, iterations=10):
    n, p = X.shape
    d = np.zeros(p)
    
    XTX = np.dot(X.T, X)
    
    for i in range(iterations):
        prediction = np.dot(X, d)
        error = prediction - y
        
        gradient = (2/n) * np.dot(X.T, error)
        
        num = np.dot(gradient.T, gradient)
        den = (2/n) * np.dot(gradient.T, np.dot(XTX, gradient))
        
        if den == 0:
            break
            
        learning_rate = num / den
        
        d = d - learning_rate * gradient
        
        cost = np.mean(error**2)
        
        if i % 100 == 0:
            print(f"Iteration {i}: Cost={cost:.10f}, Optimal Learning Rate={learning_rate:.4f}")
            
    return d

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

X = np.column_stack([
    normalize_data(x1), 
    normalize_data(x2), 
    normalize_data(x3), 
    np.ones(len(x1))
])

d = gradient_descent_optimal_lr(X, y)

print("\nFinal Result:")
print(f"y = {d[0]:.4f}x1 + {d[1]:.4f}x2 + {d[2]:.4f}x3 + {d[3]:.4f}")

x1_norm = normalize_data(x1)
x2_norm = normalize_data(x2)
x3_norm = normalize_data(x3)

result = d[0] * x1_norm + d[1] * x2_norm + d[2] * x3_norm + d[3]
print(f"Predicted values: {result}")
print(f"Actual values: {y}")
print(f"Akurasi: {100 - np.mean(np.abs(result - y) / y) * 100:.2f}%")

print(f"data baris ke-4: x1=27, x2=6, x3=85")
x1_new = np.array([30, 9, 25, 27])
x2_new = np.array([9, 9, 3, 6])
x3_new = np.array([90, 80, 90, 85])
x1_new = normalize_data(x1_new)
x2_new = normalize_data(x2_new)
x3_new = normalize_data(x3_new)

result_new = d[0] * x1_new + d[1] * x2_new + d[2] * x3_new + d[3]
print(f"Predicted value: {result_new[-1]}")
