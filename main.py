"""기온 예측기 — 서울 연평균기온 회귀 직선으로 연도별 기온을 예상하는 스트림릿 앱

실행:  streamlit run temp_predictor.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ───────────────────────── 설정 ─────────────────────────
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업의 기준 기간: 2025년까지
MIN_DAYS = 300        # 관측일이 이보다 적은 해는 제외
SLIDER_MIN, SLIDER_MAX = 1900, 2100

# 따뜻한 색 팔레트
CREAM = "#FFF8EF"
BROWN = "#5A3E2B"
ORANGE = "#E8743B"
RED = "#C8412B"
SAND = "#F3E3CD"
GOLD = "#F2A541"

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {CREAM}; }}
    h1, h2, h3, p, label, span, div {{ color: {BROWN}; }}
    .big-card {{
        background: linear-gradient(135deg, {SAND}, #FFE2BF);
        border: 2px solid {GOLD};
        border-radius: 20px;
        padding: 28px 16px;
        text-align: center;
        margin: 12px 0 8px 0;
    }}
    .big-card .label {{ font-size: 1.1rem; color: {BROWN}; }}
    .big-card .value {{ font-size: 4.2rem; font-weight: 800; color: {RED}; line-height: 1.2; }}
    .big-card .sub {{ font-size: 0.95rem; color: #8A6A52; }}
    div[data-testid="stMetric"] {{
        background-color: {SAND};
        border-radius: 14px;
        padding: 12px 16px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ───────────────────────── 데이터 처리 ─────────────────────────
@st.cache_data(show_spinner="서울 기온 데이터를 불러오는 중…")
def load_raw(url: str) -> pd.DataFrame:
    # utf-8-sig: 파일 맨 앞에 BOM이 붙어 있어도 첫 열 이름이 깨지지 않는다.
    df = pd.read_csv(url, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """연도별 평균기온과 관측일 수를 구하고, 기준 기간·관측일 조건으로 걸러 낸다."""
    d = df[["날짜", "평균기온"]].copy()
    d["날짜"] = pd.to_datetime(d["날짜"].astype(str).str.strip(), errors="coerce")
    d["평균기온"] = pd.to_numeric(d["평균기온"], errors="coerce")
    d = d.dropna(subset=["날짜", "평균기온"])      # 기온이 비어 있는 날은 관측일에서 제외
    d["연도"] = d["날짜"].dt.year
    d = d[d["연도"] <= LAST_YEAR]                  # 2025년 이후 데이터 제외

    g = d.groupby("연도")["평균기온"].agg(연평균기온="mean", 관측일수="count").reset_index()
    return g[g["관측일수"] >= MIN_DAYS].reset_index(drop=True)


def fit_line(years: np.ndarray, temps: np.ndarray):
    slope, intercept = np.polyfit(years, temps, 1)
    r = float(np.corrcoef(years, temps)[0, 1])
    return float(slope), float(intercept), r


# ───────────────────────── 화면 ─────────────────────────
st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온에 직선을 맞추고, 고른 해의 기온을 예상해 봅니다.")

try:
    raw = load_raw(DATA_URL)
    yearly = yearly_summary(raw)
except Exception as e:  # 네트워크 오류, 열 이름 불일치 등
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

if len(yearly) < 3:
    st.error("조건을 만족하는 연도가 너무 적어 직선을 만들 수 없어요.")
    st.stop()

years = yearly["연도"].to_numpy(dtype=float)
temps = yearly["연평균기온"].to_numpy(dtype=float)
slope, intercept, r = fit_line(years, temps)

n_years = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# 직선을 만든 해 정보
c1, c2, c3 = st.columns(3)
c1.metric("직선을 만든 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
st.caption(
    f"{LAST_YEAR}년까지의 자료 중 관측일이 {MIN_DAYS}일 이상인 해만 사용했어요."
)

# 슬라이더 + 예상 기온
st.subheader("연도를 골라 보세요")
sel_year = st.slider("연도", SLIDER_MIN, SLIDER_MAX, 2050, step=1)
pred = slope * sel_year + intercept

note = ""
if sel_year > end_year:
    note = f"자료({end_year}년) 이후를 직선으로 늘려 본 값이에요"
elif sel_year < start_year:
    note = f"자료({start_year}년) 이전을 직선으로 늘려 본 값이에요"
else:
    note = "자료가 있는 기간 안의 연도예요"

st.markdown(
    f"""
    <div class="big-card">
        <div class="label">{sel_year}년의 예상 연평균기온</div>
        <div class="value">{pred:.1f} °C</div>
        <div class="sub">{note}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 상관계수
st.subheader("상관계수와 회귀 직선")
m1, m2 = st.columns(2)
m1.metric("상관계수 r", f"{r:.3f}")
m2.metric("직선의 식", f"기온 = {slope:.4f} × 연도 {intercept:+.2f}")

# 그래프 (plotly)
x_lo = min(start_year, sel_year)
x_hi = max(end_year, sel_year)
line_x = np.array([x_lo, x_hi], dtype=float)
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=8, color=ORANGE, line=dict(width=1, color=BROWN)),
        hovertemplate="%{x}년<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(color=RED, width=3),
        hovertemplate="%{x}년<br>직선 값 %{y:.2f} °C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=[sel_year],
        y=[pred],
        mode="markers",
        name=f"{sel_year}년 예상",
        marker=dict(size=16, color=GOLD, symbol="star", line=dict(width=2, color=BROWN)),
        hovertemplate=f"{sel_year}년 예상<br>{pred:.2f} °C<extra></extra>",
    )
)
fig.update_layout(
    title=f"서울 연평균기온 (r = {r:.3f})",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    paper_bgcolor=CREAM,
    plot_bgcolor="#FFFDF9",
    font=dict(color=BROWN),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=70, b=10),
)
fig.update_xaxes(gridcolor=SAND)
fig.update_yaxes(gridcolor=SAND)
st.plotly_chart(fig, width="stretch")

if sel_year > end_year or sel_year < start_year:
    st.caption("⚠️ 직선을 자료 밖으로 늘린 값은 참고용이에요. 실제 기온은 이 값과 다를 수 있어요.")
