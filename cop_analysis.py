"""
SB30 부하 해리력 시험 부적합 발생 분석 + AI 예측
"""
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from openpyxl.chart import BarChart, LineChart, Reference

# ============================================================
# 설정
# ============================================================
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(BASE_DIR, "SB30 부하 해리력 시험 부적합 발생 현황(260527)의 워크시트.xlsx")
YEAR_SHEETS = ['17년','18년','19년','20년','21년','22년','23년','24년','25년','26년']
REPORT_FILE = os.path.join(BASE_DIR, "분석결과_리포트.xlsx")

# ============================================================
# 1. 데이터 로드
# ============================================================
all_sheets = []
xl = pd.ExcelFile(DATA_FILE)
for s in YEAR_SHEETS:
    if s in xl.sheet_names:
        df = pd.read_excel(DATA_FILE, sheet_name=s)
        df['_시트'] = s
        df['_연도'] = int(s.replace('년', '')) + 2000
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
        all_sheets.append(df)

df = pd.concat(all_sheets, ignore_index=True)
df.columns = df.columns.str.strip()
print(f"통합 데이터: {len(df)}행 × {len(df.columns)}열\n")


# ============================================================
# 2. 전처리
# ============================================================
df['시험일'] = pd.to_datetime(df['시험일'], errors='coerce')
df['시험_연도'] = df['시험일'].dt.year
df['시험_월']  = df['시험일'].dt.month
df['시험_일']  = df['시험일'].dt.day

df['S1_수치'] = pd.to_numeric(df['S1'], errors='coerce')

# NG 유형 분류
# - 유형1: S1 = '해리불가' (힘을 가해도 해제 불가)
# - 유형2: S1이 수치지만 결과='NG' (해리력 규격 초과)
# - 전체 NG: 결과 컬럼 기준 (두 유형 모두 포함)
df['NG_해리불가'] = df['S1'].apply(lambda x: 1 if str(x).strip() == '해리불가' else 0)
df['NG_수치초과'] = ((df['NG_해리불가'] == 0) & (df['결과'] == 'NG')).astype(int)
df['Is_NG']     = (df['결과'] == 'NG').astype(int)   # 전체 NG 플래그


# ============================================================
# 3. 연도별 품질 추세 분석
# ============================================================
print("=" * 60)
print("[1] 연도별 품질 추세")
print("=" * 60)
trend = df.groupby('_시트').agg(
    총건수    = ('Is_NG', 'count'),
    NG전체    = ('Is_NG', 'sum'),
    NG_해리불가 = ('NG_해리불가', 'sum'),
    NG_수치초과 = ('NG_수치초과', 'sum'),
    S1_평균   = ('S1_수치', lambda x: x[df.loc[x.index,'Is_NG']==0].mean()),
    S1_std    = ('S1_수치', lambda x: x[df.loc[x.index,'Is_NG']==0].std()),
).reset_index()
trend['NG율(%)'] = (trend['NG전체'] / trend['총건수'] * 100).round(1)
trend.index = range(len(trend))
print(trend[['_시트','총건수','NG전체','NG율(%)','NG_해리불가','NG_수치초과','S1_평균','S1_std']].to_string(index=False))

print(f"\n  전체 NG: {df['Is_NG'].sum()}건 / {len(df)}건 ({df['Is_NG'].mean()*100:.1f}%)")
print(f"    ├ 해리불가  : {df['NG_해리불가'].sum()}건")
print(f"    └ 수치초과  : {df['NG_수치초과'].sum()}건")


# ============================================================
# 4. S1 수치 분포 분석 (OK vs NG 비교)
# ============================================================
print("\n" + "=" * 60)
print("[2] S1 해리력 분포 비교")
print("=" * 60)
ok_vals = df[(df['Is_NG'] == 0)]['S1_수치'].dropna()
ng_vals = df[(df['NG_수치초과'] == 1)]['S1_수치'].dropna()  # 수치 있는 NG만

print(f"  {'구분':<12} {'평균':>7} {'중앙값':>7} {'std':>7} {'min':>7} {'max':>7} {'n':>5}")
print(f"  {'OK':12} {ok_vals.mean():7.2f} {ok_vals.median():7.2f} {ok_vals.std():7.2f} "
      f"{ok_vals.min():7.2f} {ok_vals.max():7.2f} {len(ok_vals):5}")
print(f"  {'NG(수치초과)':12} {ng_vals.mean():7.2f} {ng_vals.median():7.2f} {ng_vals.std():7.2f} "
      f"{ng_vals.min():7.2f} {ng_vals.max():7.2f} {len(ng_vals):5}")

# 경계 케이스: OK이지만 고위험 구간 (OK 평균 + 2σ 초과)
upper_threshold = ok_vals.mean() + 2 * ok_vals.std()
marginal_ok = df[(df['Is_NG'] == 0) & (df['S1_수치'] > upper_threshold)]
print(f"\n  ※ 경계 케이스 (OK이지만 {upper_threshold:.1f}N 초과): {len(marginal_ok)}건")
print(f"     → 규격 여유 부족, 향후 NG 전환 위험군")


# ============================================================
# 5. 고위험 차종 분석
# ============================================================
print("\n" + "=" * 60)
print("[3] 차종별 NG 위험도 (10건 이상)")
print("=" * 60)
grp_car = df.groupby('차종').agg(
    총건수=('Is_NG','count'), NG건수=('Is_NG','sum'), S1_평균=('S1_수치','mean')
)
grp_car['NG율(%)'] = (grp_car['NG건수'] / grp_car['총건수'] * 100).round(1)
grp_car = grp_car[grp_car['총건수'] >= 10].sort_values('NG율(%)', ascending=False)
print(grp_car.head(15).to_string())


# ============================================================
# 6. 고위험 품번 분석
# ============================================================
print("\n" + "=" * 60)
print("[4] 품번별 NG 위험도 (3건 이상, 상위 15개)")
print("=" * 60)
grp_pn = df.groupby('품번').agg(
    총건수=('Is_NG','count'), NG건수=('Is_NG','sum')
)
grp_pn['NG율(%)'] = (grp_pn['NG건수'] / grp_pn['총건수'] * 100).round(1)
grp_pn = grp_pn[grp_pn['총건수'] >= 3].sort_values('NG율(%)', ascending=False)
print(grp_pn.head(15).to_string())

# 고위험 품번 목록 저장 (이후 예측에 활용)
HIGH_RISK_PN = set(grp_pn[grp_pn['NG율(%)'] >= 30].index.tolist())
print(f"\n  ⚠ NG율 30% 이상 품번: {len(HIGH_RISK_PN)}개")


# ============================================================
# 7. 단계(구분)별 분석
# ============================================================
print("\n" + "=" * 60)
print("[5] 시험 단계별 NG 현황")
print("=" * 60)
grp_stage = df.groupby('단계').agg(
    총건수=('Is_NG','count'), NG건수=('Is_NG','sum')
).sort_values('NG건수', ascending=False)
grp_stage['NG율(%)'] = (grp_stage['NG건수'] / grp_stage['총건수'] * 100).round(1)
print(grp_stage.to_string())


# ============================================================
# 8. AI 모델 학습
# ============================================================
# 범주형 인코딩
categorical_cols = ['구분', '단계', '기종', '차종', '품번', '품명', '동하중번호', 'LOTNO', '로트넘버']
label_encoders = {}

df_model = df.copy()
for col in categorical_cols:
    if col in df_model.columns:
        df_model[col] = df_model[col].astype(str).str.strip()
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

df_model = df_model.fillna(0)

FEATURE_CANDIDATES = ['구분', '단계', '기종', '차종', '품번', '동하중번호', 'LOTNO', '시험_연도', '시험_월', '시험_일']
feature_cols = [c for c in FEATURE_CANDIDATES if c in df_model.columns]

X = df_model[feature_cols]

# ------ 모델 A: S1 수치 회귀 (OK 샘플만) ------
df_reg = df_model[df_model['Is_NG'] == 0].copy()
df_reg['S1_수치'] = df_reg['S1_수치'].replace(0, np.nan).fillna(df_reg['S1_수치'].median())
X_reg, y_reg = df_reg[feature_cols], df_reg['S1_수치']

X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
model_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model_reg.fit(X_tr_r, y_tr_r)
y_pr_r = model_reg.predict(X_te_r)

# ------ 모델 B: NG 분류 (전체 NG 기준, 두 유형 포함) ------
y_cls = df_model['Is_NG']
X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
    X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)
model_cls = RandomForestClassifier(
    n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'
)
model_cls.fit(X_tr_c, y_tr_c)
y_pr_c = model_cls.predict(X_te_c)

print("\n" + "=" * 60)
print("[6] AI 모델 성능")
print("=" * 60)
print(f"  [모델 A] 부하 수치(S1) 회귀 예측")
print(f"    R2 Score : {r2_score(y_te_r, y_pr_r):.4f}")
print(f"    RMSE     : {np.sqrt(mean_squared_error(y_te_r, y_pr_r)):.4f} N")

print(f"\n  [모델 B] NG 분류 예측 (해리불가 + 수치초과 통합)")
print(f"    Accuracy : {accuracy_score(y_te_c, y_pr_c)*100:.2f}%")
print()
print(classification_report(y_te_c, y_pr_c, target_names=['OK','NG']))

# 피처 중요도
fi = pd.Series(model_cls.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"  [피처 중요도 - 분류]")
for name, val in fi.items():
    bar = '█' * int(val * 50)
    print(f"    {name:<15}: {val:.4f}  {bar}")


# ============================================================
# 9. S1 시계열 추세 + 미래 예측
# ============================================================
print("\n" + "=" * 60)
print("[7] S1 해리력 추세 예측 (월별)")
print("=" * 60)

# OK 샘플의 월별 평균 집계 (유효 연도만)
ok_df = df[(df['Is_NG'] == 0) & df['시험_연도'].between(2017, 2026)].copy()
monthly_avg = (
    ok_df.groupby(['시험_연도', '시험_월'])['S1_수치']
    .agg(['mean', 'std', 'count'])
    .reset_index()
)
monthly_avg.columns = ['연도', '월', 'S1_평균', 'S1_std', '건수']
monthly_avg = monthly_avg[monthly_avg['건수'] >= 5].dropna()

# 시간 인덱스: 2017.0 = 2017년 1월, 소수점은 월 비중
monthly_avg['t'] = monthly_avg['연도'] + (monthly_avg['월'] - 1) / 12

# 선형 트렌드 회귀
coeffs = np.polyfit(monthly_avg['t'], monthly_avg['S1_평균'], deg=1)
slope, intercept = coeffs
trend_fn = np.poly1d(coeffs)

# 계절성: 월별 잔차 평균 (트렌드 제거 후 월 편차)
monthly_avg['trend'] = trend_fn(monthly_avg['t'])
monthly_avg['residual'] = monthly_avg['S1_평균'] - monthly_avg['trend']
seasonal = monthly_avg.groupby('월')['residual'].mean()

print(f"  월별 트렌드: 기울기 = {slope:+.3f} N/년  (양수=상승, 음수=하락)")
print(f"  계절성 편차 (월별 평균 잔차):")
for m, v in seasonal.items():
    bar = '▲' if v > 0 else '▼'
    print(f"    {int(m):2}월: {v:+.2f} N  {bar}")

# 향후 12개월 예측 생성
last_t   = monthly_avg['t'].max()
last_yr  = int(monthly_avg['연도'].max())
last_mon = int(monthly_avg[monthly_avg['연도'] == last_yr]['월'].max())

forecast_rows = []
yr, mon = last_yr, last_mon + 1
for _ in range(12):
    if mon > 12:
        mon = 1
        yr += 1
    t_val     = yr + (mon - 1) / 12
    trend_val = trend_fn(t_val)
    seas_val  = seasonal.get(mon, 0)
    pred_val  = trend_val + seas_val
    # 불확실성: 최근 3년 월별 std 평균
    recent_std = monthly_avg[monthly_avg['연도'] >= 2024]['S1_std'].mean()
    forecast_rows.append({
        '연도': yr, '월': mon,
        '예측_S1': round(pred_val, 2),
        '하한(95%)': round(pred_val - 1.96 * recent_std, 2),
        '상한(95%)': round(pred_val + 1.96 * recent_std, 2),
    })
    mon += 1

forecast_monthly = pd.DataFrame(forecast_rows)
print(f"\n  향후 12개월 S1 예측:")
print(f"  {'연도':>5} {'월':>3}  {'예측(N)':>8}  {'하한':>8}  {'상한':>8}")
for _, row in forecast_monthly.iterrows():
    print(f"  {int(row['연도']):5} {int(row['월']):3}월  {row['예측_S1']:8.2f}  "
          f"{row['하한(95%)']:8.2f}  {row['상한(95%)']:8.2f}")


# ============================================================
# 10. 연도별 S1 레벨 예측 (2027~2028)
# ============================================================
print("\n" + "=" * 60)
print("[8] 연도별 S1 평균 예측 (2027~2028)")
print("=" * 60)

annual_avg = (
    ok_df[ok_df['시험_연도'] >= 2021]   # 최근 상승 구간만 사용
    .groupby('시험_연도')['S1_수치']
    .mean()
    .reset_index()
)
annual_avg.columns = ['연도', 'S1_평균']

# 1차 선형 회귀
a_coeffs = np.polyfit(annual_avg['연도'], annual_avg['S1_평균'], deg=1)
a_fn = np.poly1d(a_coeffs)
resid_std = np.std(annual_avg['S1_평균'] - a_fn(annual_avg['연도']))

print(f"  기준 기간: {int(annual_avg['연도'].min())}~{int(annual_avg['연도'].max())}년")
print(f"  기울기: {a_coeffs[0]:+.3f} N/년\n")
print(f"  {'연도':>6}  {'실측/예측':>10}  {'하한(90%)':>10}  {'상한(90%)':>10}  {'구분':>5}")
for _, row in annual_avg.iterrows():
    print(f"  {int(row['연도']):6}  {row['S1_평균']:10.2f}{'':>22}  실측")

future_years = pd.DataFrame({'연도': [2027, 2028]})
for yr in [2027, 2028]:
    pred  = a_fn(yr)
    lower = pred - 1.645 * resid_std
    upper = pred + 1.645 * resid_std
    print(f"  {yr:6}  {'':>10}  {lower:10.2f}  {upper:10.2f}  예측")
    future_years.loc[future_years['연도'] == yr, 'S1_예측'] = round(pred, 2)
    future_years.loc[future_years['연도'] == yr, '하한(90%)'] = round(lower, 2)
    future_years.loc[future_years['연도'] == yr, '상한(90%)'] = round(upper, 2)

print(f"\n  ※ 예측 근거: 2021년 이후 S1 상승 추세 ({a_coeffs[0]:+.2f} N/년)")
print(f"     → 생산 공정 안정화에 따른 해리력 소폭 증가 지속 예상")


# ============================================================
# 11. SPC 관리도 기준값 산출
# ============================================================
print("\n" + "=" * 60)
print("[9] SPC 관리도 기준값 (최근 3년: 2024~2026)")
print("=" * 60)

spc_data = ok_df[ok_df['시험_연도'] >= 2024]['S1_수치'].dropna()
spc_cl  = spc_data.mean()
spc_std = spc_data.std()
spc_ucl = spc_cl + 3 * spc_std
spc_lcl = max(0, spc_cl - 3 * spc_std)

# 경보선 (±2σ)
warn_upper = spc_cl + 2 * spc_std
warn_lower = max(0, spc_cl - 2 * spc_std)

print(f"  기준 n = {len(spc_data)}건")
print(f"")
print(f"  UCL (관리 상한, +3σ) : {spc_ucl:.2f} N")
print(f"  경보 상한 (+2σ)      : {warn_upper:.2f} N")
print(f"  CL  (중심선)         : {spc_cl:.2f} N")
print(f"  경보 하한 (-2σ)      : {warn_lower:.2f} N")
print(f"  LCL (관리 하한, -3σ) : {spc_lcl:.2f} N")

# 최근 데이터 중 관리 이탈 건수
violations_ucl = ok_df[ok_df['시험_연도'] >= 2024][
    ok_df['S1_수치'] > spc_ucl
].shape[0] if len(ok_df[ok_df['시험_연도'] >= 2024]) > 0 else 0
violations_lcl = ok_df[ok_df['시험_연도'] >= 2024][
    ok_df['S1_수치'] < spc_lcl
].shape[0] if len(ok_df[ok_df['시험_연도'] >= 2024]) > 0 else 0
print(f"\n  최근 3년 관리 이탈:")
print(f"    UCL 초과: {violations_ucl}건  /  LCL 미달: {violations_lcl}건")

spc_summary = pd.DataFrame({
    '항목': ['UCL (+3σ)', '경보상한 (+2σ)', 'CL (중심)', '경보하한 (-2σ)', 'LCL (-3σ)'],
    '기준값(N)': [round(spc_ucl,2), round(warn_upper,2), round(spc_cl,2),
                  round(warn_lower,2), round(spc_lcl,2)]
})


# ============================================================
# 12. 엑셀 리포트 자동 저장
# ============================================================
import time as _time
_save_path = REPORT_FILE
for _attempt in range(3):
    try:
        _f = open(_save_path, 'ab')
        _f.close()
        break
    except PermissionError:
        _alt = _save_path.replace('.xlsx', f'_{_attempt+1}.xlsx')
        print(f"  [경고] 파일이 열려 있습니다. 대체 저장: {os.path.basename(_alt)}")
        _save_path = _alt
        break
REPORT_FILE = _save_path

print(f"\n엑셀 리포트 저장 중: {REPORT_FILE}")

def _add_bar(ws, title, cat_col, data_col, n_rows, anchor, width=22, height=14):
    c = BarChart()
    c.type = "col"; c.grouping = "clustered"; c.style = 10
    c.title = title; c.y_axis.title = "NG율(%)"; c.width = width; c.height = height
    c.add_data(Reference(ws, min_col=data_col, min_row=1, max_row=n_rows + 1),
               titles_from_data=True)
    c.set_categories(Reference(ws, min_col=cat_col, min_row=2, max_row=n_rows + 1))
    ws.add_chart(c, anchor)

def _add_line(ws, title, y_label, col_start, col_end, cat_col, n_rows, anchor,
              hdr_row=1, width=28, height=16, smooth=True):
    c = LineChart()
    c.style = 10; c.title = title; c.y_axis.title = y_label; c.width = width; c.height = height
    c.add_data(Reference(ws, min_col=col_start, max_col=col_end,
                         min_row=hdr_row, max_row=hdr_row + n_rows), titles_from_data=True)
    c.set_categories(Reference(ws, min_col=cat_col, min_row=hdr_row + 1, max_row=hdr_row + n_rows))
    if smooth:
        for s in c.series:
            s.smooth = True
    ws.add_chart(c, anchor)

with pd.ExcelWriter(REPORT_FILE, engine='openpyxl') as writer:

    # ── 시트 1: 연도별추세 ──────────────────────────────────────
    # cols: _시트(1) 총건수(2) NG전체(3) NG율(%)(4) NG_해리불가(5) NG_수치초과(6) S1_평균(7) S1_std(8)
    trend_out = trend[['_시트','총건수','NG전체','NG율(%)','NG_해리불가','NG_수치초과','S1_평균','S1_std']]
    n_tr = len(trend_out)
    trend_out.to_excel(writer, sheet_name='연도별추세', index=False)
    ws = writer.sheets['연도별추세']
    _add_bar(ws,  "연도별 NG율(%)",     cat_col=1, data_col=4, n_rows=n_tr, anchor="J2")
    _add_line(ws, "연도별 S1 평균(N)", "S1(N)", col_start=7, col_end=7,
              cat_col=1, n_rows=n_tr, anchor="J24")

    # ── 시트 2: 차종별NG ────────────────────────────────────────
    # cols: 차종(1) 총건수(2) NG건수(3) S1_평균(4) NG율(%)(5)
    n_car = min(15, len(grp_car))
    grp_car.reset_index().head(n_car).to_excel(writer, sheet_name='차종별NG', index=False)
    ws = writer.sheets['차종별NG']
    _add_bar(ws, "차종별 NG율(%) - 상위 15개", cat_col=1, data_col=5,
             n_rows=n_car, anchor="G2", width=26, height=16)

    # ── 시트 3: 품번별NG ────────────────────────────────────────
    # cols: 품번(1) 총건수(2) NG건수(3) NG율(%)(4)
    n_pn = min(15, len(grp_pn))
    grp_pn.reset_index().head(n_pn).to_excel(writer, sheet_name='품번별NG', index=False)
    ws = writer.sheets['품번별NG']
    _add_bar(ws, "품번별 NG율(%) - 상위 15개", cat_col=1, data_col=4,
             n_rows=n_pn, anchor="F2", width=28, height=16)

    # ── 시트 4: 단계별NG ────────────────────────────────────────
    # cols: 단계(1) 총건수(2) NG건수(3) NG율(%)(4)
    n_st = len(grp_stage)
    grp_stage.reset_index().to_excel(writer, sheet_name='단계별NG', index=False)
    ws = writer.sheets['단계별NG']
    _add_bar(ws, "시험 단계별 NG율(%)", cat_col=1, data_col=4, n_rows=n_st, anchor="F2")

    # ── 시트 5: 경계케이스 (raw) ────────────────────────────────
    marginal_ok[['_시트','차종','품번','품명','동하중번호','S1_수치','결과','시험일']].to_excel(
        writer, sheet_name='경계케이스', index=False)

    # ── 시트 6: 전체NG목록 (raw) ────────────────────────────────
    df_ng_list = df[df['Is_NG'] == 1][
        ['_시트','차종','품번','품명','동하중번호','S1','결과','시험일','NG_해리불가','NG_수치초과']
    ].copy()
    df_ng_list['NG유형'] = df_ng_list['NG_해리불가'].map({1:'해리불가', 0:'수치초과'})
    df_ng_list.to_excel(writer, sheet_name='전체NG목록', index=False)

    # ── 시트 7: 월별예측(향후12개월) ────────────────────────────
    # cols: 연도(1) 월(2) 예측_S1(3) 하한(95%)(4) 상한(95%)(5) 기간(6)
    fc_m = forecast_monthly.copy()
    fc_m['기간'] = fc_m.apply(lambda r: f"{int(r['연도'])}-{int(r['월']):02d}", axis=1)
    n_fc = len(fc_m)
    fc_m.to_excel(writer, sheet_name='월별예측(향후12개월)', index=False)
    ws = writer.sheets['월별예측(향후12개월)']
    _add_line(ws, "향후 12개월 S1 해리력 예측 (95% 신뢰구간)", "S1(N)",
              col_start=3, col_end=5, cat_col=6, n_rows=n_fc, anchor="H2", width=30, height=16)

    # ── 시트 8: 연도별예측 ──────────────────────────────────────
    # cols: 연도(1) S1_실측(2) 하한(90%)(3) 상한(90%)(4) 구분(5)
    hist_rows  = annual_avg.rename(columns={'S1_평균': 'S1_실측'}).copy()
    hist_rows['구분'] = '실측'
    future_rows = future_years.rename(columns={'S1_예측': 'S1_실측'}).copy()
    future_rows['구분'] = '예측'
    forecast_annual = pd.concat([hist_rows, future_rows], ignore_index=True)
    forecast_annual = forecast_annual[['연도', 'S1_실측', '하한(90%)', '상한(90%)', '구분']]
    n_ann = len(forecast_annual)
    forecast_annual.to_excel(writer, sheet_name='연도별예측', index=False)
    ws = writer.sheets['연도별예측']
    _add_line(ws, "연도별 S1 실측 + 2027~2028 예측 (90% 구간)", "S1 평균(N)",
              col_start=2, col_end=4, cat_col=1, n_rows=n_ann, anchor="G2", width=28, height=16)

    # ── 시트 9: SPC관리도 ───────────────────────────────────────
    # 상단(rows 1-6): spc_summary 2열
    # 하단(row 9~): spc_out 10열
    #   연도(1) 월(2) S1_평균(3) S1_std(4) 건수(5) CL(6) UCL(7) LCL(8) 이탈여부(9) 기간(10)
    spc_summary.to_excel(writer, sheet_name='SPC관리도', index=False)
    spc_out = monthly_avg[monthly_avg['연도'] >= 2024][['연도','월','S1_평균','S1_std','건수']].copy()
    spc_out['CL']    = round(spc_cl, 2)
    spc_out['UCL']   = round(spc_ucl, 2)
    spc_out['LCL']   = round(spc_lcl, 2)
    spc_out['이탈여부'] = spc_out['S1_평균'].apply(
        lambda x: 'UCL초과' if x > spc_ucl else ('LCL미달' if x < spc_lcl else 'OK'))
    spc_out['기간']   = spc_out.apply(
        lambda r: f"{int(r['연도'])}-{int(r['월']):02d}", axis=1)
    n_spc = len(spc_out)
    spc_out.to_excel(writer, sheet_name='SPC관리도', startrow=8, index=False)
    ws = writer.sheets['SPC관리도']

    # SPC 라인 차트: S1_평균(3), CL(6), UCL(7), LCL(8)  /  기간(10) as category
    hdr = 9          # Excel header row (1-indexed)
    end = hdr + n_spc
    lc_spc = LineChart()
    lc_spc.title = "SPC 관리도 - 부하 해리력(S1)"
    lc_spc.y_axis.title = "S1(N)"; lc_spc.x_axis.title = "기간(연도-월)"
    lc_spc.style = 10; lc_spc.width = 35; lc_spc.height = 20
    for col_idx, col_name in [(3,'S1_평균'),(6,'CL'),(7,'UCL'),(8,'LCL')]:
        lc_spc.add_data(Reference(ws, min_col=col_idx, min_row=hdr, max_row=end),
                        titles_from_data=True)
    lc_spc.set_categories(Reference(ws, min_col=10, min_row=hdr+1, max_row=end))
    for s in lc_spc.series:
        s.smooth = True
    ws.add_chart(lc_spc, "L2")

print("  → 리포트 저장 완료")


# ============================================================
# 10. 신규 샘플 예측 함수
# ============================================================
def predict_future_sample(new_data_dict):
    """
    사용 예:
    predict_future_sample({
        '구분': 'COP', '단계': 'MX', '기종': 'B30', '차종': 'KU0',
        '품번': '89830O3210NNB', '동하중번호': 'SS15206',
        'LOTNO': 'D0051', '시험일': '2026-08-01'
    })
    """
    input_df = pd.DataFrame([new_data_dict])
    input_df['시험일'] = pd.to_datetime(input_df['시험일'])
    input_df['시험_연도'] = input_df['시험일'].dt.year
    input_df['시험_월']  = input_df['시험일'].dt.month
    input_df['시험_일']  = input_df['시험일'].dt.day

    for col in categorical_cols:
        if col in input_df.columns and col in label_encoders:
            val = str(new_data_dict.get(col, '')).strip()
            if val in label_encoders[col].classes_:
                input_df[col] = label_encoders[col].transform([val])[0]
            else:
                print(f"  [경고] '{col}'='{val}' 학습 데이터 미포함 → -1 처리")
                input_df[col] = -1

    X_in = input_df[feature_cols]

    # 개별 트리 예측으로 신뢰구간 계산
    tree_preds = np.array([tree.predict(X_in)[0] for tree in model_reg.estimators_])
    predicted_load = tree_preds.mean()
    pred_std       = tree_preds.std()
    ci_lower       = predicted_load - 1.96 * pred_std
    ci_upper       = predicted_load + 1.96 * pred_std

    ng_prob    = model_cls.predict_proba(X_in)[0][1] * 100
    risk_label = "위험" if ng_prob >= 30 else "주의" if ng_prob >= 10 else "정상"

    # SPC 관리한계 대비 판정
    spc_judge = ""
    if predicted_load > spc_ucl:
        spc_judge = f"  ← UCL({spc_ucl:.1f}N) 초과"
    elif predicted_load > warn_upper:
        spc_judge = f"  ← 경보선({warn_upper:.1f}N) 초과"
    elif predicted_load < spc_lcl:
        spc_judge = f"  ← LCL({spc_lcl:.1f}N) 미달"

    # 고위험 품번 여부 확인
    pn = new_data_dict.get('품번', '')
    is_high_risk_pn = '  [고위험 품번]' if pn in HIGH_RISK_PN else ''

    print(f"\n  [AI 예측 결과]")
    print(f"    예상 S1 해리력   : {predicted_load:.2f} N{is_high_risk_pn}{spc_judge}")
    print(f"    95% 예측 구간    : {ci_lower:.2f} ~ {ci_upper:.2f} N")
    print(f"    NG 발생 위험도   : {ng_prob:.1f}%  [{risk_label}]")
    print(f"    SPC CL / UCL    : {spc_cl:.2f} / {spc_ucl:.2f} N")
    return predicted_load, ci_lower, ci_upper, ng_prob


# ============================================================
# 실행 예시 (주석 해제 후 사용)
# ============================================================
# predict_future_sample({
#     '구분': 'COP', '단계': 'MX', '기종': 'B30', '차종': 'KU0',
#     '품번': '89830O3210NNB', '동하중번호': 'SS15206',
#     'LOTNO': 'D0051', '시험일': '2026-08-01'
# })
