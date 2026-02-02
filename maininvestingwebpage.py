import logging
import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Query

# Try importing requests_cache; if missing, log a warning
try:
    import requests_cache

    # ---------------- FIX YFINANCE BLOCKING ----------------
    session = requests_cache.CachedSession(
        cache_name="yahoo_cache",
        backend="sqlite",
        expire_after=3600  # 1 hour cache
    )
    yf.set_tz_cache_location("yahoo_cache")

except ModuleNotFoundError:
    session = None
    print("Warning: requests_cache not installed. yfinance will work but without caching.")

# ---------------- APP SETUP ----------------
app = FastAPI(title="Portfolio Tracker API")
logger = logging.getLogger("portfolio_app")

ACTIVE_PORTFOLIO_FILE = "portfolio.csv"

# ---------------- UTILITIES ----------------
def sanitize_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "symbol" not in df.columns:
        return pd.DataFrame(columns=["symbol", "shares", "price"])

    df = df.dropna(subset=["symbol"])
    df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    df["shares"] = pd.to_numeric(df.get("shares", 0), errors="coerce")
    df["price"] = pd.to_numeric(df.get("price", 0), errors="coerce")
    df = df[df["shares"].notna() & (df["shares"] > 0) & df["price"].notna()]
    return df

def load_and_sanitize_portfolio() -> pd.DataFrame:
    if not os.path.exists(ACTIVE_PORTFOLIO_FILE):
        return pd.DataFrame(columns=["symbol", "shares", "price"])
    try:
        df = pd.read_csv(ACTIVE_PORTFOLIO_FILE)
    except Exception as e:
        logger.error(f"Error reading portfolio file: {e}")
        return pd.DataFrame(columns=["symbol", "shares", "price"])
    return sanitize_portfolio(df)

# ---------------- CSV UPLOAD ----------------
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global ACTIVE_PORTFOLIO_FILE

    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{file.filename}"

    content = await file.read()
    with open(path, "wb") as f:
        f.write(content)

    ACTIVE_PORTFOLIO_FILE = path
    df = sanitize_portfolio(pd.read_csv(path))

    return {
        "message": "CSV uploaded successfully",
        "active_file": path,
        "rows_loaded": len(df),
        "symbols": df["symbol"].unique().tolist()
    }

# ---------------- MARKET DATA ----------------
@app.get("/market/{symbol}")
def market_data(symbol: str):
    symbol = symbol.strip().upper()
    try:
        stock = yf.Ticker(symbol, session=session)
        hist = stock.history(period="5y")
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    if hist.empty:
        raise HTTPException(status_code=404, detail="No market data returned")
    return hist.reset_index().to_dict(orient="records")

# ---------------- PORTFOLIO SUMMARY ----------------
@app.get("/portfolio/{symbol}")
def portfolio_data(symbol: str):
    portfolio = load_and_sanitize_portfolio()
    symbol = symbol.strip().upper()
    trades = portfolio[portfolio["symbol"] == symbol]
    if trades.empty:
        return {"error": "symbol not found in portfolio"}

    total_shares = trades["shares"].sum()
    if total_shares == 0:
        return {"error": "symbol has zero shares"}

    avg_cost = (trades["shares"] * trades["price"]).sum() / total_shares
    stock = yf.Ticker(symbol, session=session)
    price_series = stock.history(period="1d")["Close"]
    if price_series.empty:
        raise HTTPException(status_code=503, detail="Price lookup failed")

    price = float(price_series.iloc[-1])
    equity = total_shares * price
    pnl = (price - avg_cost) * total_shares
    roi = (price - avg_cost) / avg_cost * 100

    return {
        "symbol": symbol,
        "total_shares": int(total_shares),
        "avg_cost": round(avg_cost, 2),
        "current_price": round(price, 2),
        "equity": round(equity, 2),
        "pnl": round(pnl, 2),
        "roi_percent": round(roi, 2)
    }

# ---------------- TIMELINE CHART ENGINE ----------------
@app.get("/portfolio_timeseries/{symbol}")
def portfolio_timeseries(
    symbol: str,
    range: str = Query("1y", enum=["1d", "1w", "1m", "3m", "1y", "5y"])
):
    portfolio = load_and_sanitize_portfolio()
    symbol = symbol.strip().upper()
    trades = portfolio[portfolio["symbol"] == symbol]
    if trades.empty:
        return {"error": "symbol not found in portfolio"}

    total_shares = trades["shares"].sum()
    if total_shares == 0:
        return {"error": "symbol has zero shares"}

    avg_cost = (trades["shares"] * trades["price"]).sum() / total_shares
    period_map = {
        "1d": "1d",
        "1w": "7d",
        "1m": "1mo",
        "3m": "3mo",
        "1y": "1y",
        "5y": "5y"
    }

    stock = yf.Ticker(symbol, session=session)
    hist = stock.history(period=period_map[range])
    if hist.empty:
        raise HTTPException(status_code=503, detail="No historical data")

    hist = hist.reset_index()
    hist["equity"] = hist["Close"] * total_shares
    hist["pnl"] = (hist["Close"] - avg_cost) * total_shares

    return hist[["Date", "Close", "equity", "pnl"]].to_dict(orient="records")

# ---------------- TOTAL PORTFOLIO ----------------
@app.get("/portfolio_total")
def portfolio_total(range: str = "1y"):
    portfolio = load_and_sanitize_portfolio()
    if portfolio.empty:
        return []

    symbols = portfolio["symbol"].unique()
    combined = None

    for sym in symbols:
        shares = portfolio[portfolio["symbol"] == sym]["shares"].sum()
        if shares == 0:
            continue

        stock = yf.Ticker(sym, session=session)
        hist = stock.history(period=range)
        if hist.empty:
            continue

        hist["value"] = hist["Close"] * shares
        if combined is None:
            combined = hist["value"]
        else:
            combined += hist["value"]

    if combined is None:
        return []

    combined = combined.reset_index()
    return combined.to_dict(orient="records")