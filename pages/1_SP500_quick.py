import streamlit as st, pandas as pd, numpy as np
from datetime import date, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils.bq import run_query, bq_ping

st.set_page_config(page_title="⚡ SP500 Quick", layout="wide")
st.title("⚡ SP500 Quick")

VIEW = st.secrets.get("tables", {}).get("spx_view","nth-pier-468314-p7.marketdata.spx_with_vix_v")
if not bq_ping(): st.error("Geen BigQuery-verbinding."); st.stop()

@st.cache_data(ttl=1800)
def load(): return run_query(f"SELECT * FROM `{VIEW}` ORDER BY date")
df = load(); df["date"]=pd.to_datetime(df["date"]).dt.date
for c in ["open","high","low","close","vix_close","delta_pct"]: df[c]=pd.to_numeric(df[c], errors="coerce")

def heikin_ashi(d):
    hc=(d["open"]+d["high"]+d["low"]+d["close"])/4.0
    ho=[d["open"].iloc[0]]; [ho.append((ho[-1]+hc.iloc[i-1])/2.0) for i in range(1,len(d))]
    ho=pd.Series(ho,index=d.index); hh=pd.concat([d["high"],ho,hc],axis=1).max(axis=1)
    hl=pd.concat([d["low"], ho,hc],axis=1).min(axis=1)
    return pd.DataFrame({"ha_open":ho,"ha_high":hh,"ha_low":hl,"ha_close":hc}, index=d.index)
def atr_rma(h,l,c,n): 
    pc=c.shift(1); tr=pd.concat([(h-l).abs(),(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()
def st_on_ha(ha,n=10,m=1.0):
    h,l,c=ha["ha_high"],ha["ha_low"],ha["ha_close"]; atr=atr_rma(h,l,c,n); hl2=(h+l)/2.0
    ub=hl2+m*atr; lb=hl2-m*atr
    fu=np.full(len(ha),np.nan); fl=np.full(len(ha),np.nan); trd=np.ones(len(ha),int)
    for i in range(len(ha)):
        if i==0: fu[i]=ub.iloc[i]; fl[i]=lb.iloc[i]; trd[i]=1; continue
        fu[i]=ub.iloc[i] if (ub.iloc[i]<fu[i-1]) or (c.iloc[i-1]>fu[i-1]) else fu[i-1]
        fl[i]=lb.iloc[i] if (lb.iloc[i]>fl[i-1]) or (c.iloc[i-1]<fl[i-1]) else fl[i-1]
        trd[i]=1 if c.iloc[i]>fu[i-1] else (-1 if c.iloc[i]<fl[i-1] else trd[i-1])
    line=pd.Series(np.where(trd==1,fl,fu),index=ha.index); trend=pd.Series(trd,index=ha.index)
    return line,trend
def rsi(s,n=14):
    d=s.diff(); up=pd.Series(np.where(d>0,d,0),index=s.index).rolling(n).mean()
    dn=pd.Series(np.where(d<0,-d,0),index=s.index).rolling(n).mean(); rs=up/dn
    return 100-(100/(1+rs))

min_d,max_d=df["date"].min(),df["date"].max()
start,end=st.slider("Periode",min_value=min_d,max_value=max_d,
                    value=(max(max_d-timedelta(days=365),min_d),max_d),format="YYYY-MM-DD")
d=df[(df["date"]>=start)&(df["date"]<=end)].reset_index(drop=True)
d["delta_pct"]=d["close"].pct_change()*100.0 if "delta_pct" not in d else d["delta_pct"]
ha=heikin_ashi(d); st_line,trend=st_on_ha(ha,10,1.0)

fig=make_subplots(rows=3, cols=1, shared_xaxes=True,
                  subplot_titles=["HA + Supertrend(10,1) + Donchian + EMA(20/50/200)",
                                  "VIX & Realized Vol (20/60)","Momentum: RSI(14) & CCI(20)"],
                  row_heights=[0.55,0.25,0.20], vertical_spacing=0.06)

# paneel 1
fig.add_trace(go.Candlestick(x=d["date"],open=ha["ha_open"],high=ha["ha_high"],low=ha["ha_low"],close=ha["ha_close"],name="HA"),1,1)
fig.add_trace(go.Scatter(x=d["date"],y=d["close"].ewm(span=20,adjust=False).mean(),name="EMA20"),1,1)
fig.add_trace(go.Scatter(x=d["date"],y=d["close"].ewm(span=50,adjust=False).mean(),name="EMA50"),1,1)
fig.add_trace(go.Scatter(x=d["date"],y=d["close"].ewm(span=200,adjust=False).mean(),name="EMA200"),1,1)
up=st_line.where(trend==1); dn=st_line.where(trend==-1)
fig.add_trace(go.Scatter(x=d["date"],y=up,mode="lines",line=dict(width=2,color="green"),name="ST ↑"),1,1)
fig.add_trace(go.Scatter(x=d["date"],y=dn,mode="lines",line=dict(width=2,color="red"),name="ST ↓"),1,1)

# paneel 2: VIX + RV
ret=(d["delta_pct"]/100).fillna(0)
rv20=ret.rolling(20).std()*np.sqrt(252)*100; rv60=ret.rolling(60).std()*np.sqrt(252)*100
if "vix_close" in d and d["vix_close"].notna().any():
    fig.add_trace(go.Scatter(x=d["date"],y=d["vix_close"],name="VIX"),2,1)
fig.add_trace(go.Scatter(x=d["date"],y=rv20,name="RV 20d"),2,1)
fig.add_trace(go.Scatter(x=d["date"],y=rv60,name="RV 60d"),2,1)

# paneel 3: RSI & CCI
from numpy import where
def cci(df_,n=20):
    tp=(df_["high"]+df_["low"]+df_["close"])/3; sma=tp.rolling(n).mean()
    mad=(tp-sma).abs().rolling(n).mean(); return (tp-sma)/(0.015*mad)
fig.add_trace(go.Scatter(x=d["date"],y=rsi(d["close"],14),name="RSI(14)"),3,1)
fig.add_trace(go.Scatter(x=d["date"],y=cci(d,20),name="CCI(20)"),3,1)
fig.update_layout(height=900, margin=dict(l=20,r=20,t=60,b=20), legend_orientation="h")
fig.update_layout(xaxis_rangeslider_visible=False); fig.update_xaxes(rangeslider_visible=False)
st.plotly_chart(fig, use_container_width=True)
