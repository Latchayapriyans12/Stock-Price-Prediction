# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
Predict future stock prices using an RNN model based on historical closing prices from trainset.csv and testset.csv, with data normalized using MinMaxScaler.

## Design Steps

### Step 1:
Import necessary libraries.

### Step 2:
Load and preprocess the data.

### Step 3:
Create input-output sequences.

### Step 4: 
Convert data to PyTorch tensors.

### Step 5:
Define the RNN model.

### Step 6:
Train the model using the training data.

### Step 7:
Evaluate the model and plot predictions.

## Program
#### Name:latchaya priyan S
#### Register Number:212224230139
Include your code here
```Python 
# Define RNN Model
class RNNModel(nn.Module):

    def __init__(self):
        super(RNNModel, self).__init__()

        self.hidden_size = 64
        self.num_layers = 2

        # RNN Layer
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_size, 1)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers,
                         x.size(0),
                         self.hidden_size).to(x.device)

        out, _ = self.rnn(x, h0)

        # last time step
        out = out[:, -1, :]

        out = self.fc(out)
        return out



model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Train the Model
num_epochs = 20
train_losses = []

model.train()

for epoch in range(num_epochs):

    epoch_loss = 0.0

    for x_batch, y_batch in train_loader:

        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.6f}")





# Plot training loss
print('Name: latchaya priyan  S                ')
print('Register Number:  212224230139   ')
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

## Step 4: Make Predictions on Test Set
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
print('Name: latchaya priyan  S                ')
print('Register Number:  212224230139   ')
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')



```

## Output

### True Stock Price, Predicted Stock Price vs time and Predictions 

<img width="831" height="548" alt="image" src="https://github.com/user-attachments/assets/faae3b7d-a258-4f2a-856c-29a38a86f2c0" />





## Result

The RNN model successfully predicts future stock prices based on historical closing prices. The predicted prices closely follow the actual prices, demonstrating the model's ability to capture temporal patterns. The performance of the model is evaluated by comparing the predicted and actual prices through visual plots.
