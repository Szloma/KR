import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


x = np.linspace(0, 100, 1000)
y = np.cos(x)*np.sin(x/2)
n_steps = 5

X = []
Y = []
for i in range(len(y) - n_steps):
    X.append(y[i:i + n_steps])# x value
    Y.append(y[i + n_steps]) ## sin value
    #X.append([y[i]])
    #Y.append(y[i + 1])

X = np.array(X)
Y = np.array(Y)

X = X.reshape((X.shape[0], n_steps, 1)) ##shape, time steps, features

print("X shape:", X.shape)
print("Y shape:", Y.shape)

model = Sequential([
    LSTM(32, input_shape=(n_steps, 1)),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

model.fit(X, Y, epochs=25, batch_size=32)

future_steps = 200

last_sequence = y[len(y)-n_steps:]
print("last sequence:", last_sequence, "")
last_sequence = np.expand_dims(last_sequence, axis=0)  # (1, n_steps)
last_window = np.expand_dims(last_sequence, axis=2)#
future_preds = []

for _ in range(future_steps):
    next_value = model.predict(last_window)[0, 0]
    future_preds.append(next_value)
    last_window = np.roll(last_window, -1, axis=1)
    last_window[0, -1, 0] = next_value

plt.figure()

plt.plot(x, y, label="True sin(x)")
dx = x[1] - x[0]
start_x = x[-1] + dx
future_x = start_x + dx * np.arange(future_steps)

plt.plot(future_x, future_preds, "--", label="Predicted future")

plt.legend()
plt.title("LSTM Future Prediction of sin(x)")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()
