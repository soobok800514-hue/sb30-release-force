"""
SB30 부하 해리력 시험 분석 시스템 - Streamlit Web App
"""
import io, os, warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SB30 부하 해리력 분석 시스템",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #F8F9FA; }
    .kpi-card {
        background: white; border-radius: 8px; padding: 16px 12px 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,.12); text-align: center; height: 100%;
    }
    .kpi-label { font-size: .82rem; color: #555; margin-bottom: 4px; }
    .kpi-val   { font-size: 1.9rem; font-weight: 700; }
    .risk-r { color: #C00000; } .risk-o { color: #FF8000; } .risk-g { color: #007050; }
    div[data-testid="stTab"] button { font-size: .92rem; }
    .block-container { padding-top: 1.5rem; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

YEAR_SHEETS = ["17년","18년","19년","20년","21년","22년","23년","24년","25년","26년"]
CAT_COLS    = ["구분","단계","기종","차종","품번","품명","동하중번호","LOTNO","로트넘버"]
FEAT_CANDS  = ["구분","단계","기종","차종","품번","동하중번호","LOTNO","시험_연도","시험_월","시험_일"]

# ────────────────────────────────────────────────────────────
# 데이터 로드 & 전처리
# ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="데이터 로딩 중...")
def load_df(file_bytes):
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    frames = []
    for s in YEAR_SHEETS:
        if s not in xl.sheet_names:
            continue
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=s)
        df["_시트"] = s
        df["_연도"] = int(s.replace("년","")) + 2000
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        frames.append(df)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df.columns = df.columns.str.strip()
    df["시험일"]    = pd.to_datetime(df["시험일"], errors="coerce")
    df["시험_연도"] = df["시험일"].dt.year
    df["시험_월"]   = df["시험일"].dt.month
    df["시험_일"]   = df["시험일"].dt.day
    df["S1_수치"]   = pd.to_numeric(df["S1"], errors="coerce")
    df["NG_해리불가"] = df["S1"].apply(lambda x: 1 if str(x).strip() == "해리불가" else 0)
    df["NG_수치초과"] = ((df["NG_해리불가"] == 0) & (df["결과"] == "NG")).astype(int)
    df["Is_NG"]     = (df["결과"] == "NG").astype(int)
    return df

# ────────────────────────────────────────────────────────────
# AI 모델 학습
# ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="AI 모델 학습 중...")
def train_models(file_bytes):
    df = load_df(file_bytes)
    dm = df.copy()
    encoders = {}
    for col in CAT_COLS:
        if col in dm.columns:
            dm[col] = dm[col].astype(str).str.strip()
            le = LabelEncoder()
            dm[col] = le.fit_transform(dm[col])
            encoders[col] = le
    dm = dm.fillna(0)

    feat_cols = [c for c in FEAT_CANDS if c in dm.columns]
    X = dm[feat_cols]

    # 회귀 모델 (OK 샘플)
    dreg = dm[dm["Is_NG"] == 0].copy()
    dreg["S1_수치"] = dreg["S1_수치"].replace(0, np.nan).fillna(dreg["S1_수치"].median())
    Xr, yr = dreg[feat_cols], dreg["S1_수치"]
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    m_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    m_reg.fit(Xtr, ytr)
    r2   = r2_score(yte, m_reg.predict(Xte))
    rmse = np.sqrt(mean_squared_error(yte, m_reg.predict(Xte)))

    # 분류 모델
    y_cls = dm["Is_NG"]
    Xtc, Xtec, ytc, ytec = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    m_cls = RandomForestClassifier(n_estimators=100, random_state=42,
                                   n_jobs=-1, class_weight="balanced")
    m_cls.fit(Xtc, ytc)
    acc = accuracy_score(ytec, m_cls.predict(Xtec))
    report = classification_report(ytec, m_cls.predict(Xtec),
                                   target_names=["OK","NG"], output_dict=True)

    # SPC 기준값 (2024~)
    ok_recent = df[(df["Is_NG"] == 0) & (df["시험_연도"] >= 2024)]["S1_수치"].dropna()
    spc = dict(cl=ok_recent.mean(), std=ok_recent.std())
    spc.update(ucl=spc["cl"]+3*spc["std"], lcl=max(0,spc["cl"]-3*spc["std"]),
               wu=spc["cl"]+2*spc["std"],  wl=max(0,spc["cl"]-2*spc["std"]))

    # 월별 추세 예측
    ok_df = df[(df["Is_NG"] == 0) & df["시험_연도"].between(2017, 2026)].copy()
    monthly = ok_df.groupby(["시험_연도","시험_월"])["S1_수치"].agg(["mean","std","count"]).reset_index()
    monthly.columns = ["연도","월","S1_평균","S1_std","건수"]
    monthly = monthly[monthly["건수"] >= 5].dropna()
    monthly["t"] = monthly["연도"] + (monthly["월"] - 1) / 12
    coeffs = np.polyfit(monthly["t"], monthly["S1_평균"], 1)
    fn = np.poly1d(coeffs)
    monthly["trend"] = fn(monthly["t"])
    monthly["residual"] = monthly["S1_평균"] - monthly["trend"]
    seasonal = monthly.groupby("월")["residual"].mean()

    last_yr  = int(monthly["연도"].max())
    last_mon = int(monthly[monthly["연도"] == last_yr]["월"].max())
    recent_std = monthly[monthly["연도"] >= 2024]["S1_std"].mean()
    fc_rows = []
    yr, mon = last_yr, last_mon + 1
    for _ in range(12):
        if mon > 12: mon = 1; yr += 1
        t = yr + (mon-1)/12
        p = fn(t) + seasonal.get(mon, 0)
        fc_rows.append({"연도": yr, "월": mon, "예측": round(p,2),
                        "하한": round(p-1.96*recent_std,2),
                        "상한": round(p+1.96*recent_std,2),
                        "기간": f"{yr}-{mon:02d}"})
        mon += 1
    forecast_monthly = pd.DataFrame(fc_rows)

    ann = ok_df[ok_df["시험_연도"] >= 2021].groupby("시험_연도")["S1_수치"].mean().reset_index()
    ann.columns = ["연도","S1_평균"]
    ac = np.polyfit(ann["연도"], ann["S1_평균"], 1)
    afn = np.poly1d(ac)
    rs = np.std(ann["S1_평균"] - afn(ann["연도"]))
    ann["구분"] = "실측"
    future = pd.DataFrame([
        {"연도": y, "S1_평균": round(afn(y),2),
         "하한": round(afn(y)-1.645*rs,2), "상한": round(afn(y)+1.645*rs,2), "구분":"예측"}
        for y in [2027, 2028]])
    forecast_annual = pd.concat([ann, future], ignore_index=True)

    return dict(
        m_reg=m_reg, m_cls=m_cls, encoders=encoders, feat_cols=feat_cols,
        r2=r2, rmse=rmse, acc=acc, report=report, spc=spc,
        forecast_monthly=forecast_monthly, forecast_annual=forecast_annual,
        slope=coeffs[0],
    )

# ────────────────────────────────────────────────────────────
# PPT 생성 (bytes 반환)
# ────────────────────────────────────────────────────────────
def build_ppt(df, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Cm, Pt

    import platform
    _fonts = ["Malgun Gothic","NanumGothic","AppleGothic","DejaVu Sans"]
    plt.rcParams["font.family"] = _fonts
    plt.rcParams["axes.unicode_minus"] = False

    CN=RGBColor(0x1F,0x39,0x64); CB=RGBColor(0x26,0x72,0xC4)
    CR=RGBColor(0xC0,0x00,0x00); CO=RGBColor(0xFF,0x80,0x00)
    CG=RGBColor(0x00,0x70,0x50); CW=RGBColor(0xFF,0xFF,0xFF)
    CLG=RGBColor(0xF2,0xF2,0xF2); CMG=RGBColor(0xD0,0xD0,0xD0)
    CDG=RGBColor(0x40,0x40,0x40); CLB=RGBColor(0xBD,0xD7,0xEE)
    CMB=RGBColor(0x37,0x5A,0x8C)

    def R(s,l,t,w,h,fill,lc=None):
        shp=s.shapes.add_shape(1,Cm(l),Cm(t),Cm(w),Cm(h))
        shp.fill.solid()
        shp.fill.fore_color.rgb=fill
        if lc:
            shp.line.color.rgb=lc
            shp.line.width=Pt(0.5)
        else:
            shp.line.fill.background()
        return shp

    def T(s,l,t,w,h,text,sz=9,bold=False,clr=CDG,al=PP_ALIGN.LEFT):
        tb=s.shapes.add_textbox(Cm(l),Cm(t),Cm(w),Cm(h))
        tf=tb.text_frame; tf.word_wrap=False
        p=tf.paragraphs[0]; p.alignment=al
        r=p.add_run(); r.text=text
        r.font.size=Pt(sz); r.font.bold=bold; r.font.color.rgb=clr
        return tb

    def shdr(s,l,t,w,title,c=CN):
        R(s,l,t,w,0.58,c); T(s,l+0.15,t+0.1,w-0.3,0.42,title,9,True,CW)

    def thdr(s,l,t,w,cols):
        R(s,l,t,w,0.48,CMG)
        for lbl,cx,cw,al in cols:
            T(s,l+cx,t+0.07,cw,0.36,lbl,8,True,CN,al)

    def trow(s,l,t,w,cells,idx):
        R(s,l,t,w,0.45,CLG if idx%2==0 else CW)
        for val,cx,cw,al,sz,clr,bold in cells:
            T(s,l+cx,t+0.05,cw,0.37,str(val),sz,bold,clr,al)

    trend = df.groupby("_시트").agg(
        총건수=("Is_NG","count"), NG전체=("Is_NG","sum"),
        S1_평균=("S1_수치",lambda x: x[df.loc[x.index,"Is_NG"]==0].mean())
    ).reset_index()
    trend["NG율(%)"] = (trend["NG전체"]/trend["총건수"]*100).round(1)

    grp_car = df.groupby("차종").agg(총건수=("Is_NG","count"),NG건수=("Is_NG","sum"))
    grp_car["NG율(%)"] = (grp_car["NG건수"]/grp_car["총건수"]*100).round(1)
    grp_car = grp_car[grp_car["총건수"]>=10].sort_values("NG율(%)",ascending=False)

    grp_pn = df.groupby("품번").agg(총건수=("Is_NG","count"),NG건수=("Is_NG","sum"))
    grp_pn["NG율(%)"] = (grp_pn["NG건수"]/grp_pn["총건수"]*100).round(1)
    grp_pn = grp_pn[grp_pn["총건수"]>=3].sort_values("NG율(%)",ascending=False)

    grp_st = df.groupby("단계").agg(총건수=("Is_NG","count"),NG건수=("Is_NG","sum"))
    grp_st["NG율(%)"] = (grp_st["NG건수"]/grp_st["총건수"]*100).round(1)
    grp_st = grp_st.sort_values("NG건수",ascending=False)

    spc = m["spc"]; fa = m["forecast_annual"]
    pred27 = fa[fa["연도"]==2027][["하한","상한"]].values[0]
    pred28 = fa[fa["연도"]==2028][["하한","상한"]].values[0]
    total_n=int(trend["총건수"].sum()); total_ng=int(trend["NG전체"].sum())
    ng_rate=total_ng/total_n*100

    fig,ax1=plt.subplots(figsize=(5.6,3.8))
    fig.patch.set_facecolor("#F8F9FA"); ax1.set_facecolor("#F8F9FA")
    years=trend["_시트"].astype(str); ng_r=trend["NG율(%)"]; s1_m=trend["S1_평균"]
    x=range(len(years))
    colors=["#C00000" if v>15 else "#FF8000" if v>5 else "#70AD47" for v in ng_r]
    ax1.bar(x,ng_r,color=colors,alpha=0.88,zorder=2,width=0.58)
    ax1.set_ylabel("NG율(%)",color="#C00000",fontsize=9)
    ax1.tick_params(axis="y",labelcolor="#C00000",labelsize=8)
    ax1.set_xticks(x); ax1.set_xticklabels(years,rotation=38,ha="right",fontsize=8.5)
    ax1.set_ylim(0,max(ng_r)*1.5); ax1.grid(axis="y",alpha=0.35,zorder=0)
    ax1.spines[["top","right"]].set_visible(False)
    ax2=ax1.twinx()
    ax2.plot(x,s1_m,"o-",color="#2672C4",linewidth=2.2,markersize=5.5,zorder=3)
    for xi,yi in zip(x,s1_m):
        ax2.annotate(f"{yi:.1f}",(xi,yi),textcoords="offset points",
                     xytext=(0,6),ha="center",fontsize=7,color="#2672C4")
    ax2.set_ylabel("S1 평균(N)",color="#2672C4",fontsize=9)
    ax2.tick_params(axis="y",labelcolor="#2672C4",labelsize=8)
    ax2.set_ylim(35,62); ax2.spines[["top"]].set_visible(False)
    ax1.set_title("연도별 NG율(%)  vs  S1 평균(N)",fontsize=10.5,fontweight="bold",
                   color="#1F3964",pad=7)
    fig.tight_layout(pad=0.5)
    chart_buf=io.BytesIO(); fig.savefig(chart_buf,format="png",dpi=160,bbox_inches="tight",
                facecolor=fig.get_facecolor()); chart_buf.seek(0); plt.close(fig)

    prs=Presentation(); prs.slide_width=Cm(33.87); prs.slide_height=Cm(19.05)
    slide=prs.slides.add_slide(prs.slide_layouts[6])
    bg=slide.background.fill; bg.solid(); bg.fore_color.rgb=CW

    R(slide,0,0,33.87,2.2,CN)
    T(slide,0.6,0.18,27,1.1,"SB30  부하 해리력 시험 부적합 발생 현황 분석",22,True,CW)
    T(slide,0.6,1.38,28,0.65,"SB30 Buckle Release Force Test Analysis   ·   2017–2026 (10개년)",9.5,False,CLB)
    T(slide,27.5,0.55,5.8,0.8,"June  2026",11,False,CLB,PP_ALIGN.RIGHT)

    kpis=[(f"{total_n:,}건","총 시험 건수",CN),(f"{total_ng:,}건","전체 NG 건수",CR),
          (f"{ng_rate:.1f}%","전체 NG율",CO),("+0.86 N/년","S1 상승 추세",CG)]
    kw,kg,kx0=7.5,0.49,0.5
    for i,(val,lbl,clr) in enumerate(kpis):
        kx=kx0+i*(kw+kg)
        R(slide,kx,2.35,kw,1.65,CLG,CMG)
        T(slide,kx+0.15,2.42,kw-0.3,0.55,lbl,8.5,False,CDG,PP_ALIGN.CENTER)
        T(slide,kx+0.15,2.85,kw-0.3,0.9,val,19,True,clr,PP_ALIGN.CENTER)

    LX,LY,LW=0.4,4.2,11.2
    shdr(slide,LX,LY,LW,"① 연도별 NG율(%) 및 S1 평균 추세")
    slide.shapes.add_picture(chart_buf,Cm(LX),Cm(LY+0.58),Cm(LW),Cm(7.6))

    MX,MY,MW=11.85,4.2,10.5
    shdr(slide,MX,MY,MW,"② 고위험 차종  NG율 상위 7개")
    hy=MY+0.58
    thdr(slide,MX,hy,MW,[("차종",0.1,3.2,PP_ALIGN.LEFT),
        ("총건수",3.4,3.0,PP_ALIGN.CENTER),("NG율(%)",6.6,3.7,PP_ALIGN.RIGHT)])
    hy+=0.48
    for i,(_,row) in enumerate(grp_car.reset_index().head(7).iterrows()):
        ng=row["NG율(%)"]
        trow(slide,MX,hy,MW,[
            (row["차종"],0.1,3.2,PP_ALIGN.LEFT,8.0,CDG,False),
            (f"{int(row['총건수'])}건",3.4,3.0,PP_ALIGN.CENTER,7.5,CDG,False),
            (f"{ng:.1f}%",6.6,3.7,PP_ALIGN.RIGHT,8.5,CR if ng>=30 else CO if ng>=15 else CDG,True),
        ],i); hy+=0.45

    hy+=0.18; shdr(slide,MX,hy,MW,"③ 고위험 품번  NG율 상위 5개",CMB); hy+=0.58
    thdr(slide,MX,hy,MW,[("품번",0.1,7.2,PP_ALIGN.LEFT),("NG율(%)",7.5,2.8,PP_ALIGN.RIGHT)])
    hy+=0.48
    for i,(_,row) in enumerate(grp_pn.reset_index().head(5).iterrows()):
        ng=row["NG율(%)"]
        trow(slide,MX,hy,MW,[
            (row["품번"],0.1,7.2,PP_ALIGN.LEFT,7.5,CDG,False),
            (f"{ng:.1f}%",7.5,2.8,PP_ALIGN.RIGHT,8.5,CR if ng>=50 else CO,True),
        ],i); hy+=0.45

    RX,RY,RW=22.6,4.2,11.0
    shdr(slide,RX,RY,RW,"④ SPC 관리한계  (2024~2026,  n ≈ 2,761건)",CMB); sy=RY+0.58
    thdr(slide,RX,sy,RW,[("항목",0.1,7.2,PP_ALIGN.LEFT),("기준값",7.4,3.4,PP_ALIGN.RIGHT)])
    sy+=0.48
    for i,(lbl,val,clr) in enumerate([
        ("UCL  (관리 상한, +3σ)",f"{spc['ucl']:.2f} N",CR),
        ("경보 상한 (+2σ)",f"{spc['wu']:.2f} N",CO),
        ("CL   (중심선)",f"{spc['cl']:.2f} N",CB),
        ("경보 하한 (-2σ)",f"{spc['wl']:.2f} N",CO),
        ("LCL  (관리 하한, -3σ)",f"{spc['lcl']:.2f} N",CG),
    ]):
        trow(slide,RX,sy,RW,[
            (lbl,0.1,7.2,PP_ALIGN.LEFT,8.0,CDG,False),
            (val,7.4,3.4,PP_ALIGN.RIGHT,9.0,clr,True),
        ],i); sy+=0.46

    sy+=0.2; shdr(slide,RX,sy,RW,"⑤ AI 예측  향후 S1 해리력 수준",CMB); sy+=0.58
    slope=m["slope"]
    for i,(lbl,val,clr) in enumerate([
        ("연간 상승 트렌드",f"{slope:+.2f} N/년",CB),
        ("2026 잔여 (6~12월)","49.3 ~ 51.5 N",CDG),
        ("2027년 예측 (90% 구간)",f"{pred27[0]:.1f} ~ {pred27[1]:.1f} N",CB),
        ("2028년 예측 (90% 구간)",f"{pred28[0]:.1f} ~ {pred28[1]:.1f} N",CMB),
        ("계절성 피크 (6~8월)","+0.6 ~ +1.0 N  ▲",CO),
        ("계절성 저점 (1~3월)","-0.5 ~ -1.1 N  ▼",CG),
    ]):
        trow(slide,RX,sy,RW,[
            (lbl,0.1,6.6,PP_ALIGN.LEFT,8.0,CDG,False),
            (val,6.8,4.0,PP_ALIGN.RIGHT,8.0,clr,True),
        ],i); sy+=0.46

    fy=12.45; R(slide,0.4,fy,33.0,0.58,CN)
    T(slide,0.6,fy+0.11,7.5,0.4,"⑥ 시험 단계별 NG 현황",8.5,True,CW)
    sx=8.0
    for _,row in grp_st.iterrows():
        ng=row["NG율(%)"]
        clr=CR if ng>=30 else CO if ng>=10 else CG if ng>0 else CLB
        T(slide,sx,fy+0.11,4.9,0.4,f"{row.name}  {ng:.1f}%",8.5,(ng>=10),clr)
        sx+=4.9
        if sx>33: break

    R(slide,0,18.42,33.87,0.63,CLG)
    T(slide,0.5,18.51,22,0.45,
      f"AI 모델: RandomForest  |  학습 데이터 {len(df):,}건  "
      f"|  NG 분류 정확도 {m['acc']*100:.1f}%  |  회귀 R² = {m['r2']:.2f}",7.5,False,CDG)
    T(slide,23,18.51,10.5,0.45,
      f"SPC 기준: 2024~2026  ·  CL={spc['cl']:.1f}N",7.5,False,CDG,PP_ALIGN.RIGHT)

    buf=io.BytesIO(); prs.save(buf); buf.seek(0)
    return buf.getvalue()

# ────────────────────────────────────────────────────────────
# 신규 샘플 예측
# ────────────────────────────────────────────────────────────
def predict_sample(input_dict, m, df):
    inp = pd.DataFrame([input_dict])
    inp["시험일"]    = pd.to_datetime(inp["시험일"])
    inp["시험_연도"] = inp["시험일"].dt.year
    inp["시험_월"]   = inp["시험일"].dt.month
    inp["시험_일"]   = inp["시험일"].dt.day
    warns = []
    for col in CAT_COLS:
        if col in inp.columns and col in m["encoders"]:
            val = str(input_dict.get(col, "")).strip()
            if val in m["encoders"][col].classes_:
                inp[col] = m["encoders"][col].transform([val])[0]
            else:
                warns.append(col)
                inp[col] = -1
    X_in = inp[m["feat_cols"]]
    trees = np.array([t.predict(X_in)[0] for t in m["m_reg"].estimators_])
    pred_s1  = trees.mean()
    pred_std = trees.std()
    ng_prob  = m["m_cls"].predict_proba(X_in)[0][1] * 100
    spc = m["spc"]
    if pred_s1 > spc["ucl"]: spc_judge = f"🔴 UCL({spc['ucl']:.1f}N) 초과"
    elif pred_s1 > spc["wu"]: spc_judge = f"🟠 경보상한({spc['wu']:.1f}N) 근접"
    elif pred_s1 < spc["lcl"]: spc_judge = f"🔴 LCL({spc['lcl']:.1f}N) 미달"
    elif pred_s1 < spc["wl"]: spc_judge = f"🟠 경보하한({spc['wl']:.1f}N) 근접"
    else: spc_judge = f"🟢 정상 범위 (CL={spc['cl']:.1f}N)"
    return dict(pred_s1=pred_s1, std=pred_std, ng_prob=ng_prob,
                ci_lo=pred_s1-1.96*pred_std, ci_hi=pred_s1+1.96*pred_std,
                spc_judge=spc_judge, warns=warns)

# ════════════════════════════════════════════════════════════
# MAIN UI
# ════════════════════════════════════════════════════════════
st.markdown("## 🔧 SB30 부하 해리력 시험 분석 시스템")
st.markdown("---")

with st.sidebar:
    st.markdown("### 📂 데이터 파일 업로드")
    uploaded = st.file_uploader("연도별 시트가 포함된 Excel 파일",
                                type=["xlsx"], label_visibility="collapsed")
    if uploaded:
        st.success(f"✅ {uploaded.name}")
    else:
        st.info("Excel 파일을 업로드하면\n분석이 자동 시작됩니다.")
        st.markdown("**지원 시트명:** `17년` ~ `26년`")
    st.markdown("---")
    st.caption("© SB30 부하 해리력 분석 시스템\n인젝션 설계 품질관리 도구")

if not uploaded:
    st.markdown("""
    <div style='text-align:center;padding:80px 0;color:#888'>
        <h3>👈 좌측에서 Excel 파일을 업로드해 주세요</h3>
        <p>시험 데이터 파일 (연도별 시트 포함) 을 업로드하면<br>
        자동 분석 및 AI 예측이 실행됩니다.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── 데이터 로드 & 모델 학습
file_bytes = uploaded.getvalue()
df = load_df(file_bytes)
if df is None:
    st.error("시트명을 확인해 주세요. (예: 17년, 18년, ...)")
    st.stop()

m = train_models(file_bytes)

# ── 사이드바 다운로드 버튼
with st.sidebar:
    st.markdown("### 📥 리포트 다운로드")
    with st.spinner("PPT 생성 중..."):
        ppt_bytes = build_ppt(df, m)
    st.download_button("🗂 PPT 보고서 다운로드", ppt_bytes,
                       "SB30_부하해리력_분석보고서.pptx",
                       "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                       use_container_width=True)

# ── 집계 데이터
trend = df.groupby("_시트").agg(
    총건수=("Is_NG","count"), NG전체=("Is_NG","sum"),
    NG_해리불가=("NG_해리불가","sum"), NG_수치초과=("NG_수치초과","sum"),
    S1_평균=("S1_수치", lambda x: x[df.loc[x.index,"Is_NG"]==0].mean()),
).reset_index()
trend["NG율(%)"] = (trend["NG전체"]/trend["총건수"]*100).round(1)

ok_s1  = df[(df["Is_NG"]==0)]["S1_수치"].dropna()
ng_s1  = df[(df["NG_수치초과"]==1)]["S1_수치"].dropna()
grp_car = df.groupby("차종").agg(총건수=("Is_NG","count"),NG건수=("Is_NG","sum"))
grp_car["NG율(%)"] = (grp_car["NG건수"]/grp_car["총건수"]*100).round(1)
grp_car = grp_car[grp_car["총건수"]>=10].sort_values("NG율(%)",ascending=False)
grp_pn = df.groupby("품번").agg(총건수=("Is_NG","count"),NG건수=("Is_NG","sum"))
grp_pn["NG율(%)"] = (grp_pn["NG건수"]/grp_pn["총건수"]*100).round(1)
grp_pn = grp_pn[grp_pn["총건수"]>=3].sort_values("NG율(%)",ascending=False)
grp_st = df.groupby("단계").agg(총건수=("Is_NG","count"),NG건수=("Is_NG","sum"))
grp_st["NG율(%)"] = (grp_st["NG건수"]/grp_st["총건수"]*100).round(1)
grp_st = grp_st.sort_values("NG건수",ascending=False)

total_n=int(trend["총건수"].sum()); total_ng=int(trend["NG전체"].sum())
ng_rate=total_ng/total_n*100

# ── KPI 카드
c1,c2,c3,c4,c5 = st.columns(5)
kpi_data = [
    (c1, "총 시험 건수",  f"{total_n:,}건",      "#1F3964"),
    (c2, "전체 NG 건수",  f"{total_ng:,}건",      "#C00000"),
    (c3, "전체 NG율",    f"{ng_rate:.1f}%",       "#FF8000"),
    (c4, "OK S1 평균",   f"{ok_s1.mean():.1f} N", "#2672C4"),
    (c5, "S1 상승 추세", f"+{m['slope']:.2f} N/년","#007050"),
]
for col, lbl, val, clr in kpi_data:
    col.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">{lbl}</div>
        <div class="kpi-val" style="color:{clr}">{val}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── 탭
tabs = st.tabs(["📊 연도별 추세","⚠️ NG 분석","🔩 고위험 분석",
                "📈 SPC 관리도","🔮 AI 예측","🧪 신규 예측"])

# ────────── Tab 1: 연도별 추세 ──────────
with tabs[0]:
    c1, c2 = st.columns([3, 2])
    with c1:
        bar_colors = ["#C00000" if v>15 else "#FF8000" if v>5 else "#70AD47"
                      for v in trend["NG율(%)"]]
        fig = go.Figure()
        fig.add_bar(x=trend["_시트"], y=trend["NG율(%)"], name="NG율(%)",
                    marker_color=bar_colors, yaxis="y1",
                    hovertemplate="%{x}<br>NG율: %{y:.1f}%<extra></extra>")
        fig.add_scatter(x=trend["_시트"], y=trend["S1_평균"], name="S1 평균(N)",
                        mode="lines+markers+text", yaxis="y2",
                        line=dict(color="#2672C4",width=2.5),
                        marker=dict(size=7),
                        text=trend["S1_평균"].round(1),
                        textposition="top center", textfont=dict(size=10),
                        hovertemplate="%{x}<br>S1: %{y:.2f}N<extra></extra>")
        fig.update_layout(
            title="연도별 NG율(%)  vs  S1 평균(N)",
            yaxis=dict(title="NG율(%)", side="left", color="#C00000"),
            yaxis2=dict(title="S1 평균(N)", side="right", overlaying="y",
                        color="#2672C4", range=[35,60]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=380, plot_bgcolor="#F8F9FA", paper_bgcolor="white",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**연도별 상세 현황**")
        st.dataframe(
            trend[["_시트","총건수","NG전체","NG율(%)","NG_해리불가","NG_수치초과","S1_평균"]]
            .rename(columns={"_시트":"연도","S1_평균":"S1 평균(N)"})
            .style.background_gradient(subset=["NG율(%)"], cmap="RdYlGn_r")
            .format({"S1 평균(N)": "{:.1f}", "NG율(%)": "{:.1f}%"}),
            height=360, use_container_width=True,
        )

# ────────── Tab 2: NG 분석 ──────────
with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_histogram(x=ok_s1, name="OK", nbinsx=40,
                          marker_color="#2672C4", opacity=0.7)
        fig.add_histogram(x=ng_s1, name="NG(수치초과)", nbinsx=30,
                          marker_color="#C00000", opacity=0.7)
        fig.add_vline(x=ok_s1.mean(), line_dash="dash", line_color="#1F3964",
                      annotation_text=f"OK 평균 {ok_s1.mean():.1f}N")
        fig.update_layout(title="S1 해리력 분포 비교 (OK vs NG)",
                          xaxis_title="S1(N)", yaxis_title="건수",
                          barmode="overlay", height=360, legend=dict(orientation="h"),
                          plot_bgcolor="#F8F9FA", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = go.Figure()
        stage_colors = ["#C00000" if v>=30 else "#FF8000" if v>=10 else "#70AD47"
                        for v in grp_st["NG율(%)"]]
        fig2.add_bar(x=grp_st.index, y=grp_st["NG율(%)"],
                     marker_color=stage_colors,
                     hovertemplate="%{x}<br>NG율: %{y:.1f}%<extra></extra>")
        fig2.update_layout(title="시험 단계별 NG율(%)",
                           yaxis_title="NG율(%)", height=360,
                           plot_bgcolor="#F8F9FA", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**NG 유형 분류**")
    ng_types = pd.DataFrame({
        "NG 유형": ["해리불가 (S1 측정 불가)", "수치 초과 (높은 해리력)", "합계"],
        "건수": [int(df["NG_해리불가"].sum()), int(df["NG_수치초과"].sum()),
                 int(df["Is_NG"].sum())],
        "비율(%)": [
            f"{df['NG_해리불가'].mean()*100:.1f}%",
            f"{df['NG_수치초과'].mean()*100:.1f}%",
            f"{df['Is_NG'].mean()*100:.1f}%",
        ],
        "설명": ["힘 가해도 해제 불가", f"OK 평균({ok_s1.mean():.1f}N) 대비 {ng_s1.mean()/ok_s1.mean()*100:.0f}% 수준", "-"]
    })
    st.dataframe(ng_types, use_container_width=True, hide_index=True)

# ────────── Tab 3: 고위험 분석 ──────────
with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        top_n = st.slider("표시 개수", 5, 20, 10, key="car_n")
        fig = px.bar(grp_car.reset_index().head(top_n),
                     x="NG율(%)", y="차종", orientation="h",
                     color="NG율(%)", color_continuous_scale="RdYlGn_r",
                     text="NG율(%)", title=f"차종별 NG율(%) 상위 {top_n}개",
                     hover_data=["총건수","NG건수"])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(height=420, showlegend=False,
                          plot_bgcolor="#F8F9FA", paper_bgcolor="white",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        top_pn = st.slider("표시 개수", 5, 20, 10, key="pn_n")
        gpn = grp_pn.reset_index().head(top_pn).copy()
        gpn["품번_short"] = gpn["품번"].str[:18]
        fig2 = px.bar(gpn, x="NG율(%)", y="품번_short", orientation="h",
                      color="NG율(%)", color_continuous_scale="RdYlGn_r",
                      text="NG율(%)", title=f"품번별 NG율(%) 상위 {top_pn}개",
                      hover_data=["총건수","NG건수"])
        fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig2.update_layout(height=420, showlegend=False,
                           plot_bgcolor="#F8F9FA", paper_bgcolor="white",
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig2, use_container_width=True)

# ────────── Tab 4: SPC ──────────
with tabs[3]:
    spc = m["spc"]
    c1, c2 = st.columns([3, 1])
    with c1:
        ok_recent = df[(df["Is_NG"]==0) & df["시험_연도"].between(2024,2026)].copy()
        monthly_spc = ok_recent.groupby(["시험_연도","시험_월"])["S1_수치"].mean().reset_index()
        monthly_spc["기간"] = monthly_spc.apply(
            lambda r: f"{int(r['시험_연도'])}-{int(r['시험_월']):02d}", axis=1)
        fig = go.Figure()
        fig.add_scatter(x=monthly_spc["기간"], y=monthly_spc["S1_수치"],
                        mode="lines+markers", name="S1 평균",
                        line=dict(color="#2672C4",width=2), marker=dict(size=6))
        for val, name, clr, dash in [
            (spc["ucl"],"UCL","#C00000","dash"),
            (spc["wu"], "경보상한","#FF8000","dot"),
            (spc["cl"], "CL","#1F3964","solid"),
            (spc["wl"], "경보하한","#FF8000","dot"),
            (spc["lcl"],"LCL","#007050","dash"),
        ]:
            fig.add_hline(y=val, line_dash=dash, line_color=clr,
                          annotation_text=f"{name} {val:.1f}N",
                          annotation_position="right")
        fig.update_layout(title="SPC 관리도 - 부하 해리력(S1)  (2024~2026)",
                          xaxis_title="기간(연도-월)", yaxis_title="S1(N)",
                          height=420, plot_bgcolor="#F8F9FA", paper_bgcolor="white",
                          xaxis=dict(tickangle=45))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**관리한계 기준값**")
        for lbl, val, clr in [
            ("UCL (+3σ)", f"{spc['ucl']:.2f} N", "red"),
            ("경보상한 (+2σ)", f"{spc['wu']:.2f} N", "orange"),
            ("CL (중심)", f"{spc['cl']:.2f} N", "blue"),
            ("경보하한 (-2σ)", f"{spc['wl']:.2f} N", "orange"),
            ("LCL (-3σ)", f"{spc['lcl']:.2f} N", "green"),
        ]:
            st.markdown(f"**:{clr}[{lbl}]**  \n`{val}`")
            st.markdown("")

# ────────── Tab 5: 예측 ──────────
with tabs[4]:
    c1, c2 = st.columns(2)
    fc_m  = m["forecast_monthly"]
    fc_a  = m["forecast_annual"]
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=fc_m["기간"], y=fc_m["예측"], name="예측 S1",
                        mode="lines+markers", line=dict(color="#2672C4",width=2.5),
                        marker=dict(size=7))
        fig.add_scatter(x=fc_m["기간"], y=fc_m["상한"], name="95% 상한",
                        mode="lines", line=dict(color="#FF8000",width=1.2,dash="dot"),
                        showlegend=True)
        fig.add_scatter(x=fc_m["기간"], y=fc_m["하한"], name="95% 하한",
                        mode="lines", fill="tonexty",
                        fillcolor="rgba(38,114,196,0.12)",
                        line=dict(color="#FF8000",width=1.2,dash="dot"))
        fig.update_layout(title="향후 12개월 S1 해리력 예측 (95% 신뢰구간)",
                          xaxis_title="기간", yaxis_title="S1(N)",
                          height=360, plot_bgcolor="#F8F9FA", paper_bgcolor="white",
                          xaxis=dict(tickangle=45),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = go.Figure()
        hist = fc_a[fc_a["구분"]=="실측"]
        pred = fc_a[fc_a["구분"]=="예측"]
        fig2.add_scatter(x=hist["연도"], y=hist["S1_평균"], name="실측",
                         mode="lines+markers", line=dict(color="#1F3964",width=2.5),
                         marker=dict(size=8, symbol="circle"))
        fig2.add_scatter(x=pred["연도"], y=pred["상한"], name="90% 상한",
                         mode="lines", line=dict(color="#FF8000",width=1.2,dash="dot"))
        fig2.add_scatter(x=pred["연도"], y=pred["하한"], name="90% 하한",
                         fill="tonexty", fillcolor="rgba(255,128,0,0.15)",
                         mode="lines", line=dict(color="#FF8000",width=1.2,dash="dot"))
        fig2.add_scatter(x=pred["연도"], y=pred["S1_평균"], name="예측",
                         mode="lines+markers",
                         line=dict(color="#2672C4",width=2.5,dash="dash"),
                         marker=dict(size=8, symbol="diamond"))
        fig2.update_layout(title="연도별 S1 실측 + 2027~2028 예측 (90% 구간)",
                           xaxis_title="연도", yaxis_title="S1 평균(N)",
                           height=360, plot_bgcolor="#F8F9FA", paper_bgcolor="white",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig2, use_container_width=True)

    st.info(f"📌 월별 추세: **{m['slope']:+.3f} N/년** 상승  |  "
            f"2027 예측: **{fc_a[fc_a['연도']==2027]['하한'].values[0]:.1f}"
            f" ~ {fc_a[fc_a['연도']==2027]['상한'].values[0]:.1f} N**  |  "
            f"2028 예측: **{fc_a[fc_a['연도']==2028]['하한'].values[0]:.1f}"
            f" ~ {fc_a[fc_a['연도']==2028]['상한'].values[0]:.1f} N**")

# ────────── Tab 6: 신규 예측 ──────────
with tabs[5]:
    st.markdown("#### 신규 시험 데이터 입력 → S1 해리력 & NG 위험도 예측")
    known = {col: sorted(df[col].dropna().astype(str).unique().tolist())
             for col in ["구분","단계","기종","차종"] if col in df.columns}

    with st.form("predict_form"):
        cols = st.columns(4)
        inp = {}
        for i, (field, label) in enumerate([
            ("구분","구분"), ("단계","단계"), ("기종","기종"), ("차종","차종")]):
            opts = known.get(field, [])
            with cols[i % 4]:
                inp[field] = st.selectbox(label, opts) if opts else st.text_input(label)

        cols2 = st.columns(4)
        with cols2[0]: inp["품번"]      = st.text_input("품번")
        with cols2[1]: inp["동하중번호"] = st.text_input("동하중번호")
        with cols2[2]: inp["LOTNO"]     = st.text_input("LOTNO")
        with cols2[3]: inp["시험일"]    = st.date_input("시험일")

        submitted = st.form_submit_button("🔮 예측 실행", use_container_width=True,
                                          type="primary")

    if submitted:
        inp["시험일"] = str(inp["시험일"])
        result = predict_sample(inp, m, df)
        if result["warns"]:
            st.warning(f"⚠️ 학습 데이터에 없는 항목: {', '.join(result['warns'])} → -1 처리")

        rc1, rc2, rc3 = st.columns(3)
        risk_clr = "red" if result["ng_prob"]>=30 else "orange" if result["ng_prob"]>=10 else "green"
        risk_lbl = "위험 🔴" if result["ng_prob"]>=30 else "주의 🟠" if result["ng_prob"]>=10 else "정상 🟢"
        with rc1:
            st.metric("예상 S1 해리력", f"{result['pred_s1']:.2f} N",
                      f"95% 구간: {result['ci_lo']:.1f}~{result['ci_hi']:.1f} N")
        with rc2:
            st.metric("NG 발생 위험도", f"{result['ng_prob']:.1f}%", risk_lbl)
        with rc3:
            st.metric("SPC 판정", result["spc_judge"])

        spc = m["spc"]
        fig = go.Figure()
        fig.add_bar(x=["예측 S1"], y=[result["pred_s1"]],
                    marker_color="#2672C4", name="예측 S1",
                    error_y=dict(type="data", array=[1.96*result["std"]], visible=True))
        for val, name, clr, dash in [
            (spc["ucl"],"UCL","#C00000","dash"),
            (spc["cl"], "CL","#1F3964","solid"),
            (spc["lcl"],"LCL","#007050","dash"),
        ]:
            fig.add_hline(y=val, line_dash=dash, line_color=clr,
                          annotation_text=f"{name} {val:.1f}N")
        fig.update_layout(title="예측값 vs SPC 관리한계",
                          yaxis_title="S1(N)", height=320,
                          plot_bgcolor="#F8F9FA", paper_bgcolor="white",
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
