
import pandas as pd

def ensure_datetime(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    return out

def heikin_ashi(df_ohlc: pd.DataFrame) -> pd.DataFrame:
    # Verwacht kolommen: date, Open, High, Low, Close
    df = df_ohlc.copy().sort_values("date")
    ha_close = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4

    ha_open = [df["Open"].iloc[0]]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha_close.iloc[i-1]) / 2)

    import pandas as pd
    ha_open = pd.Series(ha_open, index=df.index)
    ha_high = pd.concat([df["High"], ha_open, ha_close], axis=1).max(axis=1)
    ha_low  = pd.concat([df["Low"],  ha_open, ha_close], axis=1).min(axis=1)

    out = df[["date"]].copy()
    out["HA_Open"]  = ha_open
    out["HA_High"]  = ha_high
    out["HA_Low"]   = ha_low
    out["HA_Close"] = ha_close
    return out
