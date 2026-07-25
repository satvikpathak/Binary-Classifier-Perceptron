# 🧠 Binary Pattern Classifier

A single-layer perceptron trained on the OR gate, visualized through an interactive Streamlit dashboard with a dark glassmorphism UI.

## Features

- **Live Inference** — toggle binary inputs and see the sigmoid probability in real time
- **3D Decision Boundary** — interactive Plotly surface showing the learned probability plane
- **Training Loss Curve** — convergence visualization across 5 000 epochs
- **Learned Parameters** — view the exact weights and bias

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the perceptron (creates model_weights.pth + training_history.json)
python train.py

# 3. Launch the dashboard
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo → set **Main file path** to `app.py`
4. Click **Deploy** — your live URL is ready in ~60 seconds

## Tech Stack

| Layer      | Tool           |
|------------|----------------|
| Model      | PyTorch        |
| Frontend   | Streamlit      |
| Charts     | Plotly         |
| Numerical  | NumPy          |
