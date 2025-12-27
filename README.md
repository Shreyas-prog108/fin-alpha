# fin-alpha

Financial analysis platform with AI insights, risk scoring, and price prediction.

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env
echo "API_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" > .env
echo "GEMINI_API_KEY=your_key_here" >> .env
echo "BACKEND_URL=http://localhost:8000" >> .env

# Run
uvicorn backend.app:app --reload

# Or run the agent CLI
python3 agents/run.py
```

## API Usage

All endpoints (except `/api/health`) require: `Authorization: Bearer YOUR_API_KEY`

```bash
# Health check (no auth)
curl http://localhost:8000/api/health

# Risk analysis
curl -X POST http://localhost:8000/api/analyze-risk \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","data":[{"close":150,"volume":5000000}]}'

# Price prediction
curl -X POST http://localhost:8000/api/predict-price \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","data":[{"close":150},{"close":148}]}'

# Market maker quote
curl -X POST http://localhost:8000/api/market-maker/quote \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"mid_price":100,"volatility":0.2,"risk_aversion":0.1,"time_horizon":1,"inventory":0,"kappa":1.5}'
```

## Project Structure

```
fin-alpha/
├── backend/              # API server
│   ├── app.py           # Main API
│   ├── config.py        # Configuration
│   ├── models.py        # Request models
│   ├── gemini_helper.py # Gemini integration
│   ├── risk_analysis.py
│   ├── price_prediction.py
│   └── market_maker.py
├── agents/              # Agent system
│   ├── agent.py
│   ├── config.py
│   ├── tools.py
│   ├── clients/         # Data clients
│   └── prompts/
├── requirements.txt
├── .env                 # Create this
└── README.md
```

## Environment Variables

```
API_KEY                 # Bearer token (generate one)
GEMINI_API_KEY          # Get from https://ai.google.dev/
BACKEND_URL             # http://localhost:8000
DEBUG=false             # Set to true for dev
```

## Endpoints

| Method | Path                        | Auth | Description      |
| ------ | --------------------------- | ---- | ---------------- |
| GET    | `/`                       | No   | Root             |
| GET    | `/api/health`             | No   | Health check     |
| POST   | `/api/analyze-risk`       | Yes  | Risk analysis    |
| POST   | `/api/predict-price`      | Yes  | Price prediction |
| POST   | `/api/market-maker/quote` | Yes  | Market maker     |
| GET    | `/api/gemini-models`      | Yes  | List models      |

## Features

- Risk analysis (volatility, anomalies)
- Price prediction (EMA, linear regression)
- Market making (Avellaneda-Stoikov)
- AI analysis (Gemini)
- Secure API (Bearer tokens)
- Input validation
- Security headers

## Security

✅ API keys in headers (not URLs)
✅ Bearer token authentication
✅ Input validation on all requests
✅ Security headers added
✅ CORS configured
✅ Error logging

## Production

```bash
# Set these
DEBUG=false
REQUIRE_HTTPS=true

# Use HTTPS with reverse proxy
# Keep API keys in secure vault
# Run security scans: bandit, pip-audit
```

## Issues

**Port in use?** `uvicorn backend.app:app --port 8001`

**Missing API key?** Add to `.env`:

```
GEMINI_API_KEY=your_key
API_KEY=your_bearer_token
```

## Documentation

- `COMPLETION_REPORT.md` - What was fixed
- `GET_STARTED.md` - Setup guide
- `SECURITY.md` - Security details
