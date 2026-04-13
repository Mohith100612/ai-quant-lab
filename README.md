# AI Quant Lab - Zerodha Kite Connect

A comprehensive suite of tools for historical OHLCV data fetching and real-time tick collection using the Zerodha Kite Connect API.

## Features

- **Historical Data**: Fetch and store OHLCV data with automatic range chunking to stay within API limits.
- **Real-time Ticker**: 1-second polling of stock quotes for a watch list of symbols, stored in MySQL.
- **FastAPI Wrapper**: Modern REST API to manage authentication, watch lists, and data retrieval.
- **Database Integrated**: Persistent storage using MySQL for both historical candles and real-time ticks.
- **Automated Scheduling**: Cron-based market-open (09:15) and market-close (15:30) polling.

## Project Structure

- `main.py`: CLI entry point for fetching historical data.
- `api.py`: FastAPI server for managing the system.
- `ticker.py`: Real-time quote polling and database insertion logic.
- `fetcher.py`: Historical data fetching utilities and Kite API interaction.
- `auth.py`: Authentication and session management.
- `database.py`: MySQL connectivity and schema management.

## Setup

1. **Environment Variables**: Create a `.env` file based on your Zerodha credentials:
   ```env
   KITE_API_KEY=your_api_key
   KITE_API_SECRET=your_api_secret
   KITE_ACCESS_TOKEN=your_access_token (updated via auth)
   MYSQL_URL=mysql://user:pass@host:port/dbname
   ```

2. **Installation**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Running the API**:
   ```bash
   uvicorn api:app --reload
   ```

## Usage

### CLI Historical Fetching
```bash
python main.py --symbol RELIANCE --from 2024-01-01 --to 2024-12-31 --interval day
```

### API Endpoints
- `GET /auth/login-url`: Get the Kite login URL.
- `POST /auth/session`: Exchange request token for access token.
- `GET /candles`: Fetch candles (DB-first).
- `POST /ticker/watch`: Add a symbol to the real-time watch list.
