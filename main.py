import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from model import StockLSTM

# ====== STEP 1. Download Data ======
ticker = "VNM.VN"  # Có thể đổi sang "FPT.VN" hoặc "VNINDEX"
df = yf.download(ticker, start="2020-01-01", end="2024-12-31")

# Dùng cột Close
data = df[['Close']].values

# ====== STEP 2. Scale data ======
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# ====== STEP 3. Create dataset ======
def create_sequences(data, seq_length=60):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ====== STEP 4. Convert to torch tensors ======
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ====== STEP 5. Model, Loss, Optimizer ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StockLSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ====== STEP 6. Training ======
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        output = model(batch_x)
        loss = criterion(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.6f}")

# ====== STEP 7. Evaluation ======
model.eval()
with torch.no_grad():
    preds = model(X_test.to(device)).cpu().numpy()

# Chuyển ngược lại giá trị gốc
preds_inv = scaler.inverse_transform(preds)
y_test_inv = scaler.inverse_transform(y_test)

# ====== STEP 8. Visualization ======
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label="Actual Price")
plt.plot(preds_inv, label="Predicted Price")
plt.title(f"{ticker} Stock Price Prediction (LSTM)")
plt.xlabel("Days")
plt.ylabel("Price (VND)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/prediction_plot.png")
plt.show()

print("✅ Done! Plot saved to results/prediction_plot.png")
