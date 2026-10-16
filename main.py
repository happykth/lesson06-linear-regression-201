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
RECENT_N = 20         # '최근 N년' 비교에 쓸 해의 개수

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
    .slope-card {{
        background: linear-gradient(135deg, #FFE2BF, {SAND});
        border: 2px solid {ORANGE};
        border-radius: 20px;
        padding: 24px 16px;
        text-align: center;
        margin: 8px 0 12px 0;
    }}
    .slope-card .label {{ font-size: 1.05rem; color: {BROWN}; }}
    .slope-card .value {{ font-size: 3.6rem; font-weight: 800; color: {RED}; line-height: 1.25; }}
    .slope-card .sub {{ font-size: 0.9rem; color: #8A6A52; }}
    .cmp-card {{
        background-color: {SAND};
        border: 1.5px solid {GOLD};
        border-radius: 16px;
        padding: 18px 10px;
        text-align: center;
        height: 100%;
    }}
    .cmp-card.recent {{ background-color: #FFE2BF; border-color: {ORANGE}; }}
    .cmp-card .label {{ font-size: 1rem; font-weight: 700; color: {BROWN}; }}
    .cmp-card .period {{ font-size: 0.85rem; color: #8A6A52; margin-bottom: 4px; }}
    .cmp-card .value {{ font-size: 2.4rem; font-weight: 800; color: {RED}; line-height: 1.3; }}
    .cmp-card .sub {{ font-size: 0.85rem; color: #8A6A52; }}
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

# 최근 N년(남은 해 중 가장 최근 N개)만으로 구한 직선
recent = yearly.tail(RECENT_N)
recent_years = recent["연도"].to_numpy(dtype=float)
recent_slope, recent_intercept, recent_r = fit_line(
    recent_years, recent["연평균기온"].to_numpy(dtype=float)
)
recent_start, recent_end = int(recent["연도"].min()), int(recent["연도"].max())
per100 = slope * 100                 # 100년에 오르는 온도(전체 기간)
recent_per100 = recent_slope * 100   # 100년에 오르는 온도(최근 N년)

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

# 기울기: 100년에 몇 도 오르는가
st.subheader("100년에 몇 도 오를까?")
direction = "오르는" if per100 >= 0 else "내리는"
st.markdown(
    f"""
    <div class="slope-card">
        <div class="label">직선의 기울기를 100년으로 바꾸면</div>
        <div class="value">100년에 {per100:+.2f} °C</div>
        <div class="sub">{start_year}~{end_year}년 전체 자료로 구한 직선 기준 · 해마다 {slope:+.4f} °C씩 {direction} 속도예요</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 전체 기간 vs 최근 N년 비교
st.markdown(f"##### 전체 기간 vs 최근 {RECENT_N}년")
col_all, col_recent = st.columns(2)
col_all.markdown(
    f"""
    <div class="cmp-card">
        <div class="label">전체 기간</div>
        <div class="period">{start_year}~{end_year}년 · {n_years}개 해</div>
        <div class="value">{per100:+.2f} °C</div>
        <div class="sub">100년당 · 상관계수 r = {r:.3f}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
col_recent.markdown(
    f"""
    <div class="cmp-card recent">
        <div class="label">최근 {RECENT_N}년</div>
        <div class="period">{recent_start}~{recent_end}년 · {len(recent)}개 해</div>
        <div class="value">{recent_per100:+.2f} °C</div>
        <div class="sub">100년당 · 상관계수 r = {recent_r:.3f}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

ratio_text = ""
if per100 > 0 and recent_per100 > 0:
    ratio_text = f" 최근 {RECENT_N}년의 기울기는 전체 기간의 약 {recent_per100 / per100:.1f}배예요."
st.caption(
    f"두 값의 차이는 {recent_per100 - per100:+.2f} °C/100년이에요.{ratio_text} "
    f"다만 최근 {RECENT_N}년은 점이 {len(recent)}개뿐이라 해마다의 들쭉날쭉한 변화에 기울기가 크게 흔들릴 수 있어요."
)

# 상관계수
st.subheader("상관계수와 회귀 직선")
m1, m2 = st.columns(2)
m1.metric("상관계수 r (전체 기간)", f"{r:.3f}")
m2.metric("직선의 식 (전체 기간)", f"기온 = {slope:.4f} × 연도 {intercept:+.2f}")

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
        name="회귀 직선(전체 기간)",
        line=dict(color=RED, width=3),
        hovertemplate="%{x}년<br>직선 값 %{y:.2f} °C<extra></extra>",
    )
)
recent_line_x = np.array([recent_start, recent_end], dtype=float)
fig.add_trace(
    go.Scatter(
        x=recent_line_x,
        y=recent_slope * recent_line_x + recent_intercept,
        mode="lines",
        name=f"직선(최근 {RECENT_N}년)",
        line=dict(color=BROWN, width=3, dash="dash"),
        hovertemplate="%{x}년<br>최근 직선 값 %{y:.2f} °C<extra></extra>",
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
