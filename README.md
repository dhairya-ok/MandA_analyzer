# M&A Deal Screening & Valuation System

End-to-end system for automated M&A target discovery, pairwise compatibility scoring, multi-model valuation, and Deal Brief PDF export.

## Stack
- Backend: FastAPI (Python), Pandas/NumPy, SQLAlchemy, LightGBM (optional), Redis cache
- DB: PostgreSQL
- Frontend: React + TypeScript (Vite)
- Deployment: Render (render.yaml) and local Docker Compose

## Local Development

1) Copy env example and set values
```bash
cp .env.example .env
```

2) Start Docker services
```bash
docker compose up -d --build
```
- API: http://localhost:8000
- Frontend: http://localhost:5173
- Postgres: localhost:5432
- Redis: localhost:6379

3) API docs
- Swagger: http://localhost:8000/docs
- Health: http://localhost:8000/health

## Deploy to Render

- Ensure your repo includes `render.yaml`
- Create a new Blueprint on Render from this repo
- Provision Managed PostgreSQL and Redis via the Blueprint
- Backend and Frontend services will be created automatically

## Project Structure
```
ma-deal-system/
├── backend/
│   ├── api/
│   │   └── main.py
│   ├── models/
│   │   └── models.py
│   ├── db.py
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── index.html
│   └── package.json
├── docker-compose.yml
├── render.yaml
├── .cursorrules
└── README.md
```

## Next Steps
- Implement data ingestion (yfinance, Finnhub, FRED)
- Implement pairwise scoring engine
- Implement DCF, Comps, and ensemble logic
- PDF export for Deal Briefs
