# 📊 Stock Surge Prediction Model Comparison: v1.0 vs v2.0

## 1. Executive Summary
This report analyzes the performance improvements achieved by transitioning from a univariate price model (**v1.0 Chronos**) to a multivariate price-volume model (**v2.0 PatchTST**). The inclusion of trading volume as a predictor significantly improved the model's ability to filter out false signals, leading to higher precision and profitability.

| Metric | v1.0 (Chronos) | v2.0 (PatchTST) | Improvement |
| :--- | :--- | :--- | :--- |
| **Input Data** | Close Price Only | **Close + Volume** | Multivariate Analysis |
| **Algorithm** | T5 Transformer (LoRA) | **PatchTST (From Scratch)** | SOTA Architecture |
| **Signals Generated** | 10,425 | **5,813** | -44% (Noise Reduction) |
| **Precision** | 11.94% | **13.88%** | +1.94%p |
| **Avg Return (5-Day)** | +5.42% | **+6.26%** | **+15.5% Relative Increase** |

---

## 2. v1.0 Model: Chronos (Univariate)
### Architecture
- **Base Model**: `amazon/chronos-t5-small` (Pre-trained on 10M+ time series).
- **Fine-tuning**: LoRA (Low-Rank Adaptation) method.
- **Input**: 60-day context window of **Close Price** only.
- **Objective**: Predict the probability distribution of future prices.

### Performance Analysis
- **Strengths**: Robust pattern recognition due to massive pre-training. Good at identifying general upward trends.
- **Weaknesses**: Cannot distinguish between low-volume (weak) increases and high-volume (strong) breakouts. Prone to generating too many signals ("Over-active").

---

## 3. v2.0 Model: PatchTST (Multivariate)
### Architecture
- **Model**: Patch Time Series Transformer (PatchTST).
- **Key Feature**: **Channel Independence** (Processes Price and Volume separately, then fuses information).
- **Configuration**:
  - Context Length: 96 days
  - Prediction Length: 5 days
  - Patch Length: 16 (Stride 8)
  - Layers: 3 / Heads: 4 / Model Dim: 128
  - Dropout: 0.2

### Training Process
- **Dataset**: ~40,000 balanced samples (50% Surge / 50% Non-Surge).
- **Preprocessing**: 
  - Price: Normalized.
  - Volume: Log-transformed (`log1p`) to handle extreme values.
- **Training**: Trained from scratch for 15 epochs using AdamW optimizer.
- **Loss Function**: MSE (Mean Squared Error) on forecast values.

### Results Interpretation
- **Signal Refining**: The model effectively learned that *price increases without volume support* are less likely to be sustained surges. This is why the total number of signals dropped from ~10k to ~5.8k.
- **Quality over Quantity**: The remaining signals proved to be of higher quality, raising the average return per trade from 5.4% to 6.3%.

---

## 4. Conclusion & Recommendation
The **v2.0 PatchTST model** is superior for real-world trading due to its higher precision and risk-adjusted return.

### Recommendation
- **Primary Model**: Use **v2.0** for generating buy signals.
- **Strategy**: 
  - Buy when `Surge Probability >= 0.6` (high confidence).
  - Hold for 5 trading days.
  - Stop-loss: -5% (suggested based on volatility).

### Future Work (v3.0)
- **Macro Factors**: Incorporate exchange rates (USD/KRW) or interest rates as static covariates.
- **Sentiment Analysis**: Add news sentiment scores as an additional input channel.
