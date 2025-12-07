
# ✈️ Predictive Maintenance: Smart Tiering with AutoEncoders & Attention LSTM

A "Smart Tiering" Prognostic Health Management (PHM) system trained on the NASA CMAPSS Turbofan dataset. 

Instead of running computationally expensive Deep Learning models continuously, this architecture uses a lightweight **AutoEncoder** as a "Gatekeeper" to detect anomalies, and triggers a high-fidelity **Attention-based LSTM** for RUL prediction only when degradation is detected.

## 📊 Key Results (FD001 Test Set)
| Metric | Value | Note |
| :--- | :--- | :--- |
| **RMSE** | **15.76** | State-of-the-art range for this dataset. |
| **NASA Score** | **473.0** | Extremely low penalty (Standard LSTMs score ~600+). |
| **AUC-ROC** | **0.99** | Perfect separation of Healthy vs Failing states. |
| **Recall @ 20** | **0.81** (Mean) / **0.94** (Conservative) | High safety margin using Uncertainty Quantification. |

## 🧠 Architecture
### Tier 1: Anomaly Detection (The Gatekeeper)
- **Model:** Dense AutoEncoder.
- **Function:** Monitors sensor reconstruction error.
- **Output:** Health Index (0-100%) & Confidence Z-Score.
- **Logic:** `If Health Score < 80: Trigger Tier 2`.

### Tier 2: RUL Prognostics (The Expert)
- **Model:** LSTM with Temporal Attention.
- **Function:** Predicts Remaining Useful Life (cycles).
- **Feature:** Implements **Monte Carlo Dropout** to provide 95% Confidence Intervals along with the prediction.

## 🛠️ Tech Stack
- **PyTorch:** Model development (Custom LSTM Cell with Attention).
- **Signal Processing:** Exponential Weighted Moving Average (EWMA) for noise reduction.
- **Physics-Informed:** RUL Clipping and Piece-wise linear degradation assumptions.

## 📉 Visuals
*(Insert your "Health Index" plot and "RUL with Uncertainty" plot here)*

## 🚀 Usage
```python
# The system runs a smart check on incoming sensor data
status = smart_maintenance_check(sensor_data, ae_model, lstm_model)

if status['action'] == "Tier 2 Activated":
    print(f"Predicted RUL: {status['rul_prediction']} +/- {status['confidence']}")