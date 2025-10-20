import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
from model import StockLSTM

ticker = "FPT.VN"
df = yf.download(ticker, start="2018-01-01", end="2024-12-31")
df.to_csv(f"data/{ticker}_raw.csv", index=True)

df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()

delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
df['RSI_14'] = 100 - (100 / (1 + rs))

ema12 = df['Close'].ewm(span=12, adjust=False).mean()
ema26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

df.dropna(inplace=True)

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'EMA_14', 'RSI_14', 'MACD', 'Signal_Line']
target = ['Close']

data = df[features].values
labels = df[target].values

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
data_scaled = scaler_X.fit_transform(data)
labels_scaled = scaler_y.fit_transform(labels)

def create_sequences(X, y, seq_length=60):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(xs), np.array(ys)

seq_length = 60
X, y = create_sequences(data_scaled, labels_scaled)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

model = StockLSTM(input_size=len(features)).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}")

model.eval()
with torch.no_grad():
    preds = model(X_test).cpu().numpy()

y_true_inv = scaler_y.inverse_transform(y_test.cpu().numpy())
y_pred_inv = scaler_y.inverse_transform(preds)

direction_acc = np.mean(np.sign(np.diff(y_pred_inv.squeeze())) == np.sign(np.diff(y_true_inv.squeeze())))
print(f"Directional Accuracy: {direction_acc*100:.2f}%")

plt.figure(figsize=(10,5))
plt.plot(y_true_inv, label="Actual")
plt.plot(y_pred_inv, label="Predicted")
plt.title(f"{ticker} Stock Forecast (LSTM + Indicators)")
plt.legend()
plt.tight_layout()
plt.savefig("results/prediction_plot.png")
plt.show()
