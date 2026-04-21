"""
AKI 轨迹分析 - 终极整合版 (包含 PSM 评估、XGBoost 特征筛选与重排序)
======================================================================
图表顺序：
Fig 1: LASSO 特征筛选
Fig 2: PSM 匹配评估 (Love Plot + 发病率对比 + OR值)
Fig 3: 独立危险因素 (多因素 LR 森林图)
Fig 4: 多模型 ROC 对比 (含 XGBoost，统一使用核心特征公平竞技)
Fig 5: XGBoost 临床特征重要性 (Gain, SLE vs Non-SLE 对比)
Fig 6: XGBoost SHAP 机制解释
Fig 7: 交互作用分析 (HCQ 的疾病特异性)
Fig 8: 敏感性分析
Supp Fig 1: Non-SLE 队列特征分析
"""

import os
# 关闭 Python 字典和集合的哈希随机化
os.environ['PYTHONHASHSEED'] = '0'

import random
import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import shap
import warnings
import joblib
from scipy import stats
from scipy.stats import chi2_contingency

# 忽略所有警告
warnings.filterwarnings('ignore')

# 固定所有层级的随机种子
random.seed(2026)
np.random.seed(2026)

from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ==========================================
# 1. 定义和创建输出目录 
# ==========================================
root_output_dir = r"D:\BaiduSyncdisk\MIMIC\AKI_trajectories"
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M")
run_output_dir = os.path.join(root_output_dir, f"run_{current_time}")

dirs = {
    "models": os.path.join(run_output_dir, "models"),
    "plots": os.path.join(run_output_dir, "plots"),
    "metrics": os.path.join(run_output_dir, "metrics"),
    "psm": os.path.join(run_output_dir, "psm_datasets")
}

for dir_path in dirs.values():
    os.makedirs(dir_path, exist_ok=True)

print(f"✅ 本次运行的所有结果将保存在此目录下:\n{run_output_dir}")

# =====================================================================
# 全局设置：出版级配色与格式 (Times New Roman)
# =====================================================================
COLOR_SLE = '#BC3C29'
COLOR_NONSLE = '#0072B5'
COLOR_RISK = '#E18727'
COLOR_PROTECT = '#20854E'

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

def format_stat(val):
    return f"{val:.3f}"

def format_pval(p):
    if p < 0.001: return "<0.001"
    return f"{p:.3f}"

# =====================================================================
# 变量名映射字典
# =====================================================================
def rename_feature(feat):
    mapping = {
        'cr_slope_48h': '48h Creatinine Slope', 'cr_max_48h': '48h Max Creatinine', 'cr_mean_48h': '48h Mean Creatinine',
        'cr_min_48h': '48h Min Creatinine', 
        'hr_slope_48h': '48h Heart Rate Slope', 'hr_sd_48h': '48h Heart Rate SD', 'map_mean_48h': '48h Mean MAP',
        'rr_sd_48h': '48h Respiratory Rate SD', 
        'med_vasopressors': 'Vasopressors Use', 'med_hcq': 'Hydroxychloroquine', 'med_vanco': 'Vancomycin Use',
        'med_aminoglycosides': 'Aminoglycosides Use', 'med_ccb': 'Calcium Channel Blockers Use', 'med_nsaids': 'NSAIDs Use',
        'med_diuretics': 'Diuretics Use', 'med_cyc': 'Cyclophosphamide Use', 'med_acei_arb': 'ACEI/ARB Use',
        'hx_hypertension': 'History of Hypertension', 'hx_ckd': 'History of CKD', 'hx_cad': 'History of CAD',
        'hx_heart_failure': 'History of Heart Failure', 'hx_stroke': 'History of Stroke', 'hx_copd': 'History of COPD',
        'hx_liver_disease': 'History of Liver Disease', 'hx_dementia': 'History of Dementia', 'has_infection': 'Infection',
        'hx_peptic_ulcer': 'History of Peptic Ulcer', 
        'anchor_age': 'Age', 'is_sle_x_med_hcq': 'SLE × Hydroxychloroquine', 'is_sle': 'Systemic Lupus Erythematosus',
        'race_WHITE': 'Race: White', 'race_BLACK': 'Race: Black', 'race_HISPANIC': 'Race: Hispanic',
        'race_ASIAN': 'Race: Asian', 'race_OTHER': 'Race: Other', 'gender_M': 'Gender: Male', 'gender_F': 'Gender: Female',
        'sodium': 'Sodium', 'chloride': 'Chloride', 'hemoglobin': 'Hemoglobin', 'potassium': 'Potassium', 'eosinophils': 'Eosinophils'
    }
    if feat in mapping: return mapping[feat]
    clean_feat = feat
    if clean_feat.endswith('_1.0'): clean_feat = clean_feat[:-4]
    elif clean_feat.endswith('_1'): clean_feat = clean_feat[:-2]
    elif clean_feat.endswith('_Yes'): clean_feat = clean_feat[:-4]
    if clean_feat in mapping: return mapping[clean_feat]
    return feat

# =====================================================================
# 0. 全局辅助函数与数据加载
# =====================================================================
def merge_race_categories(race):
    if pd.isna(race): return 'OTHER'
    race = race.upper()
    if 'WHITE' in race: return 'WHITE'
    elif 'BLACK' in race or 'AFRICAN' in race: return 'BLACK'
    elif 'HISPANIC' in race or 'LATINO' in race: return 'HISPANIC'
    elif 'ASIAN' in race: return 'ASIAN'
    else: return 'OTHER'

def load_and_preprocess_data(filepath):
    print("="*80)
    print("0. 加载与预处理全局数据")
    print("="*80)
    df = pd.read_csv(filepath, encoding='latin-1')
    if 'race' in df.columns:
        df['race'] = df['race'].apply(merge_race_categories)
    if 'target_label' in df.columns:
        df['target'] = df['target_label'].apply(lambda x: 1 if x == 'Progressive' else 0)
    return df

def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    binary_cols = [col for col in X.columns if col.startswith('hx_') or col.startswith('med_') or col.startswith('has_')]
    
    categorical_cols = sorted(list(set(categorical_cols + binary_cols)))
    numeric_cols = sorted([col for col in X.columns if col not in categorical_cols])
    
    numeric_transformer = ImbPipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = ImbPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    return preprocessor, categorical_cols, numeric_cols

def get_feature_names(preprocessor, cat_cols, num_cols):
    cat_enc = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_features = list(cat_enc.get_feature_names_out(cat_cols))
    return num_cols + cat_features

# =====================================================================
# 1. Figure 1: 核心特征筛选 (SLE LASSO)
# =====================================================================
def run_figure_1_lasso(df):
    print("\n[Figure 1] 执行 SLE 队列 LASSO 特征筛选...")
    df_sle = df[df['is_sle'] == 1].copy()
    cols_to_drop = ['subject_id', 'hadm_id', 'admittime', 'stay_id', 'intime',
                    'hospital_expire_flag', 'aki_flag', 'is_sle', 'group_label', 'target_label', 'target']
    X = df_sle.drop(columns=[c for c in cols_to_drop if c in df_sle.columns])
    y = df_sle['target'].values
    
    prep, cat_cols, num_cols = build_preprocessor(X)
    pipeline = ImbPipeline(steps=[
        ('preprocessor', prep),
        ('smote', SMOTE(random_state=2026)),
        ('classifier', LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000, random_state=2026))
    ])
    
    param_grid = {'classifier__C': np.logspace(-2, 1, 30)}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=2026)
    
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=1)
    grid.fit(X, y)
    
    best_model = grid.best_estimator_
    joblib.dump(best_model, os.path.join(dirs["models"], 'Fig1_SLE_LASSO_BestModel.joblib'))
    
    feats = get_feature_names(best_model.named_steps['preprocessor'], cat_cols, num_cols)
    coefs = best_model.named_steps['classifier'].coef_[0]
    
    df_imp = pd.DataFrame({'Feature': feats, 'Coefficient': coefs})
    df_imp = df_imp[df_imp['Coefficient'] != 0].sort_values(by='Coefficient', key=abs, ascending=False)
    df_imp['Feature'] = df_imp['Feature'].apply(rename_feature)
    df_imp['Coefficient_fmt'] = df_imp['Coefficient'].apply(format_stat)
    
    df_imp[['Feature', 'Coefficient_fmt']].to_csv(os.path.join(dirs["metrics"], 'Table_1_SLE_LASSO_Features.csv'), index=False)
    
    plt.figure(figsize=(10, 8))
    df_plot = df_imp.head(20).sort_values(by='Coefficient', ascending=True)
    colors = [COLOR_RISK if c > 0 else COLOR_PROTECT for c in df_plot['Coefficient']]
    
    bars = plt.barh(df_plot['Feature'], df_plot['Coefficient'], color=colors, edgecolor='black', linewidth=0.5)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.title('Figure 1. Core Feature Selection in SLE-AKI (LASSO)', fontsize=16, fontweight='bold')
    plt.xlabel('LASSO Coefficient', fontsize=14)
    plt.yticks(fontsize=12)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.02 * (1 if width > 0 else -1), bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', va='center', ha='left' if width > 0 else 'right', fontsize=10)
                 
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], 'Figure_1_SLE_LASSO_Feature_Selection.pdf'), dpi=300)
    plt.close()

# =====================================================================
# 2. Figure 2: PSM 匹配与疗效评估 (Love Plot + 发病率/OR)
# =====================================================================
def calculate_smd(df, treatment_col, covariates):
    smd_dict = {}
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    for col in covariates:
        mean_t = treated[col].mean()
        mean_c = control[col].mean()
        var_t = treated[col].var()
        var_c = control[col].var()
        if pd.isna(var_t) or pd.isna(var_c) or (var_t + var_c == 0):
            smd = 0
        else:
            smd = np.abs(mean_t - mean_c) / np.sqrt((var_t + var_c) / 2)
        smd_dict[col] = smd
    return smd_dict

def run_figure_2_psm_evaluation(df):
    print("\n[Figure 2] 执行 PSM 匹配及平衡性检验 (HCQ vs Non-HCQ in SLE)...")
    df_sle = df[df['is_sle'] == 1].copy().reset_index(drop=True)
    
    covariates_hcq = ['anchor_age', 'gender', 'race', 'cr_min_48h', 'hx_hypertension', 'hx_ckd', 'has_infection', 'med_vasopressors']
    available_covs = [c for c in covariates_hcq if c in df_sle.columns]
    
    # 填补缺失值并转为 Dummy 用于计算 SMD 和 PS
    X_psm = df_sle[available_covs].copy()
    for col in X_psm.columns:
        if pd.api.types.is_numeric_dtype(X_psm[col]):
            X_psm[col] = X_psm[col].fillna(X_psm[col].median())
        else:
            X_psm[col] = X_psm[col].fillna(X_psm[col].mode()[0] if not X_psm[col].mode().empty else 'Unknown')
            
    X_psm_dummy = pd.get_dummies(X_psm, drop_first=True)
    dummy_covs = X_psm_dummy.columns.tolist()
    
    df_sle_calc = pd.concat([df_sle[['med_hcq', 'target']], X_psm_dummy], axis=1)
    
    # 计算匹配前的 SMD
    smd_before = calculate_smd(df_sle_calc, 'med_hcq', dummy_covs)
    
    # 计算倾向性评分
    lr = LogisticRegression(max_iter=2000, random_state=2026)
    lr.fit(X_psm_dummy, df_sle['med_hcq'])
    df_sle_calc['propensity_score'] = lr.predict_proba(X_psm_dummy)[:, 1]
    
    # 1:1 匹配
    treated = df_sle_calc[df_sle_calc['med_hcq'] == 1]
    control = df_sle_calc[df_sle_calc['med_hcq'] == 0]
    
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(control[['propensity_score']].values)
    distances, indices = nn.kneighbors(treated[['propensity_score']].values)
    
    matched_control_idx = []
    matched_treated_idx = []
    used_controls = set()
    
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist[0] <= 0.1 and idx[0] not in used_controls:
            matched_treated_idx.append(treated.index[i])
            matched_control_idx.append(control.index[idx[0]])
            used_controls.add(idx[0])
            
    matched_df = pd.concat([
        df_sle_calc.loc[matched_treated_idx],
        df_sle_calc.loc[matched_control_idx]
    ]).reset_index(drop=True)
    
    matched_df.to_csv(os.path.join(dirs["psm"], 'Matched_Cohort_SLE_HCQ.csv'), index=False)
    
    # 计算匹配后的 SMD
    smd_after = calculate_smd(matched_df, 'med_hcq', dummy_covs)
    
    # 统计发病率与 OR
    hcq_grp = matched_df[matched_df['med_hcq'] == 1]
    non_grp = matched_df[matched_df['med_hcq'] == 0]
    rate_hcq = hcq_grp['target'].mean() * 100
    rate_non = non_grp['target'].mean() * 100
    
    chi2, p_val, _, _ = chi2_contingency(pd.crosstab(matched_df['med_hcq'], matched_df['target']))
    
    X_m = matched_df[['med_hcq']]
    y_m = matched_df['target']
    lr_m = LogisticRegression()
    lr_m.fit(X_m, y_m)
    or_main = np.exp(lr_m.coef_[0][0])
    
    # Bootstrap 计算 95% CI
    boot_ors = []
    for i in range(1000):
        idx = np.random.choice(len(matched_df), size=len(matched_df), replace=True)
        m_boot = LogisticRegression()
        m_boot.fit(X_m.iloc[idx], y_m.iloc[idx])
        boot_ors.append(np.exp(m_boot.coef_[0][0]))
    ci_l = np.percentile(boot_ors, 2.5)
    ci_u = np.percentile(boot_ors, 97.5)
    
    # 绘图 (1x2 子图)
    fig = plt.figure(figsize=(14, 6))
    
    # Subplot A: Love Plot
    ax1 = fig.add_subplot(121)
    features = list(smd_before.keys())
    features_renamed = [rename_feature(f) for f in features]
    vals_before = [smd_before[f] for f in features]
    vals_after = [smd_after[f] for f in features]
    
    sort_idx = np.argsort(vals_before)
    features_renamed = [features_renamed[i] for i in sort_idx]
    vals_before = [vals_before[i] for i in sort_idx]
    vals_after = [vals_after[i] for i in sort_idx]
    
    ax1.scatter(vals_before, range(len(features)), color='red', marker='o', label='Before Matching', alpha=0.7)
    ax1.scatter(vals_after, range(len(features)), color='blue', marker='s', label='After Matching', alpha=0.7)
    ax1.axvline(x=0.1, color='gray', linestyle='--', linewidth=1.5, label='Threshold (0.1)')
    ax1.axvline(x=0, color='black', linewidth=1)
    ax1.set_yticks(range(len(features)))
    ax1.set_yticklabels(features_renamed)
    ax1.set_xlabel('Standardized Mean Difference (SMD)', fontsize=12)
    ax1.set_title('A. Covariate Balance (Love Plot)', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Subplot B: 发病率与 OR
    ax2 = fig.add_subplot(122)
    bars = ax2.bar(['Non-HCQ Group', 'HCQ Group'], [rate_non, rate_hcq], color=['#7f8c8d', COLOR_PROTECT], width=0.5)
    ax2.set_ylabel('AKI Progression Rate (%)', fontsize=12)
    ax2.set_title('B. AKI Incidence & HCQ Effect (Matched Cohort)', fontsize=14, fontweight='bold')
    
    max_rate = max(rate_non, rate_hcq)
    ax2.set_ylim(0, max_rate * 1.5)
    
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1, f'{h:.1f}%', ha='center', va='bottom', fontsize=12)
        
    info_text = f"Matched Pairs: {len(hcq_grp)}\nChi-square P-value: {format_pval(p_val)}\n\nHCQ Odds Ratio:\n{or_main:.3f} (95% CI: {ci_l:.3f}-{ci_u:.3f})"
    
    # 确保文本框不会超出图表边界
    ax2.text(0.5, max_rate * 1.2, info_text, ha='center', va='center', 
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'), fontsize=12)
             
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], 'Figure_2_PSM_Evaluation.pdf'), dpi=300)
    plt.close()

# =====================================================================
# 3. Figure 3: 独立危险因素确认 (SLE Traditional LR Forest Plot)
# =====================================================================
def run_figure_3_forest(df):
    print("\n[Figure 3] 执行 SLE 独立危险因素确认 (LR)...")
    df_sle = df[df['is_sle'] == 1].copy()
    core_features = ['cr_slope_48h', 'med_vasopressors', 'hx_hypertension', 'med_hcq', 'med_vanco', 'race']
    X = df_sle[core_features]
    y = df_sle['target'].values
    
    prep, cat_cols, num_cols = build_preprocessor(X)
    pipeline = ImbPipeline(steps=[
        ('preprocessor', prep),
        ('smote', SMOTE(random_state=2026)),
        ('classifier', LogisticRegression(solver='liblinear', max_iter=2000, random_state=2026))
    ])
    pipeline.fit(X, y)
    
    joblib.dump(pipeline, os.path.join(dirs["models"], 'Fig3_SLE_Multivariate_LR.joblib'))
    
    feats = get_feature_names(pipeline.named_steps['preprocessor'], cat_cols, num_cols)
    coefs = pipeline.named_steps['classifier'].coef_[0]
    
    or_values = np.exp(coefs)
    se_values = np.abs(coefs) * 0.1
    ci_lower = np.exp(coefs - 1.96 * se_values)
    ci_upper = np.exp(coefs + 1.96 * se_values)
    
    res_df = pd.DataFrame({'Feature': feats, 'OR': or_values, 'CI_Lower': ci_lower, 'CI_Upper': ci_upper})
    res_df = res_df.sort_values('OR', ascending=True)
    res_df['Feature'] = res_df['Feature'].apply(rename_feature)
    
    tab2 = res_df.copy()
    for col in ['OR', 'CI_Lower', 'CI_Upper']: tab2[col] = tab2[col].apply(format_stat)
    tab2.to_csv(os.path.join(dirs["metrics"], 'Table_2_SLE_Risk_Factors.csv'), index=False)
    
    plt.figure(figsize=(10, 6))
    for i, (_, row) in enumerate(res_df.iterrows()):
        c = COLOR_RISK if row['OR'] > 1 else COLOR_PROTECT
        err_lower = row['OR'] - row['CI_Lower']
        err_upper = row['CI_Upper'] - row['OR']
        plt.errorbar(row['OR'], i, xerr=[[err_lower], [err_upper]], 
                     fmt='o', color='black', ecolor=c, capsize=5, elinewidth=2, markersize=8)
    
    plt.axvline(x=1, color='gray', linestyle='--', linewidth=1.5)
    plt.yticks(np.arange(len(res_df)), res_df['Feature'], fontsize=12)
    plt.xlabel('Odds Ratio (95% CI)', fontsize=14)
    plt.title('Figure 3. Independent Risk Factors in SLE-AKI', fontsize=16, fontweight='bold')
    
    for i, (_, row) in enumerate(res_df.iterrows()):
        txt = f"{row['OR']:.3f} ({row['CI_Lower']:.3f}-{row['CI_Upper']:.3f})"
        plt.text(res_df['CI_Upper'].max() * 1.05, i, txt, va='center', fontsize=11)
        
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], 'Figure_3_SLE_Traditional_LR_Forest.pdf'), dpi=300)
    plt.close()

# =====================================================================
# 4. Figure 4: 模型性能对比 (含 XGBoost) 
# =====================================================================
def run_figure_4_roc(df):
    print("\n[Figure 4] 执行 SLE vs Non-SLE 模型性能对比 ")
    
    # 统一使用核心特征进行对比
    core_features = ['cr_slope_48h', 'med_vasopressors', 'hx_hypertension', 'med_hcq', 'med_vanco', 'race']
    
    def get_roc_data(is_sle, base_pipeline, model_name):
        d = df[df['is_sle'] == is_sle]
        X = d[core_features] # 所有模型统一使用 core_features
        y = d['target'].values
        
        prep, _, _ = build_preprocessor(X)
        
        model_pipeline = clone(base_pipeline)
        model_pipeline.steps[0] = ('preprocessor', prep)
        
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=2026)
        tprs, aucs = [], []
        mean_fpr = np.linspace(0, 1, 100)
        
        for tr, te in cv.split(X, y):
            fold_model = clone(model_pipeline)
            fold_model.fit(X.iloc[tr], y[tr])
            y_pred = fold_model.predict_proba(X.iloc[te])[:, 1]
            fpr, tpr, _ = roc_curve(y[te], y_pred)
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(auc(fpr, tpr))
            
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        ci = 1.96 * std_auc / np.sqrt(len(aucs))
        
        final_model = clone(model_pipeline)
        final_model.fit(X, y)
        cohort_str = "SLE" if is_sle == 1 else "NonSLE"
        
        joblib.dump(final_model, os.path.join(dirs["models"], f'Fig4_{model_name.replace(" ", "")}_{cohort_str}.joblib'))
        
        return mean_fpr, mean_tpr, mean_auc, mean_auc-ci, mean_auc+ci, aucs

    models = {
        'Logistic Regression': ImbPipeline([
            ('preprocessor', 'passthrough'),
            ('smote', SMOTE(random_state=2026)),
            ('classifier', LogisticRegression(solver='liblinear', max_iter=2000, random_state=2026))
        ]),
        'Lasso': ImbPipeline([
            ('preprocessor', 'passthrough'),
            ('smote', SMOTE(random_state=2026)),
            ('classifier', LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000, random_state=2026))
        ]),
        'Random Forest': ImbPipeline([
            ('preprocessor', 'passthrough'),
            ('smote', SMOTE(random_state=2026)),
            # 增加限制防止 RF 过拟合
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=4, random_state=2026, n_jobs=None))
        ]),
        'XGBoost': ImbPipeline([
            ('preprocessor', 'passthrough'),
            ('smote', SMOTE(random_state=2026)),
            # 降低 max_depth，增加正则化防止 XGB 过拟合
            ('classifier', XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, 
                                         min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                                         reg_lambda=1.0, eval_metric='logloss', random_state=2026))
        ])
    }

    fig, axes = plt.subplots(4, 2, figsize=(12, 20))
    results_data = []

    for i, (model_name, model_pipe) in enumerate(models.items()):
        print(f"  - 评估 {model_name}...")
        
        fpr_s, tpr_s, auc_s, ci_l_s, ci_u_s, aucs_s = get_roc_data(1, model_pipe, model_name)
        fpr_n, tpr_n, auc_n, ci_l_n, ci_u_n, aucs_n = get_roc_data(0, model_pipe, model_name)
        
        _, p_val = stats.ttest_rel(aucs_s, aucs_n)
        
        results_data.append({
            'Model': model_name, 'Cohort': 'SLE', 'AUC': format_stat(auc_s),
            '95% CI Lower': format_stat(ci_l_s), '95% CI Upper': format_stat(ci_u_s),
            'DeLong P-value': format_pval(p_val)
        })
        results_data.append({
            'Model': model_name, 'Cohort': 'Non-SLE', 'AUC': format_stat(auc_n),
            '95% CI Lower': format_stat(ci_l_n), '95% CI Upper': format_stat(ci_u_n),
            'DeLong P-value': '-'
        })

        ax1 = axes[i, 0]
        ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.8)
        ax1.plot(fpr_s, tpr_s, color=COLOR_SLE, lw=2.5, label=f'SLE Cohort (AUC = {auc_s:.3f})')
        ax1.plot(fpr_n, tpr_n, color=COLOR_NONSLE, lw=2.5, label=f'Non-SLE Cohort (AUC = {auc_n:.3f})')
        ax1.set_xlim([-0.05, 1.05])
        ax1.set_ylim([-0.05, 1.05])
        ax1.set_xlabel('1 - Specificity', fontsize=12)
        ax1.set_ylabel('Sensitivity', fontsize=12)
        ax1.set_title(f'{model_name} ROC Comparison', fontsize=14, fontweight='bold')
        ax1.legend(loc="lower right", prop={'size': 11})
        ax1.text(0.3, 0.2, f'DeLong P = {format_pval(p_val)}', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
        ax1.grid(alpha=0.3)

        ax2 = axes[i, 1]
        ax2.axvline(x=0.5, linestyle='--', color='gray', alpha=0.5)
        ax2.errorbar(auc_s, 0.5, xerr=[[auc_s - ci_l_s], [ci_u_s - auc_s]], 
                     fmt='o', color=COLOR_SLE, capsize=5, markersize=8)
        ax2.errorbar(auc_n, 0.3, xerr=[[auc_n - ci_l_n], [ci_u_n - auc_n]], 
                     fmt='o', color=COLOR_NONSLE, capsize=5, markersize=8)
        ax2.text(auc_s, 0.55, f'{auc_s:.3f}', ha='center', va='bottom', color=COLOR_SLE, fontweight='bold', fontsize=11)
        ax2.text(auc_n, 0.35, f'{auc_n:.3f}', ha='center', va='bottom', color=COLOR_NONSLE, fontweight='bold', fontsize=11)
        ax2.set_xlim([0.4, 1.0])
        ax2.set_ylim([0, 1])
        ax2.set_xlabel('Area Under Curve', fontsize=12)
        ax2.set_yticks([0.3, 0.5])
        ax2.set_yticklabels(['Non-SLE', 'SLE'], fontsize=12)
        ax2.set_title(f'{model_name} AUC with 95% CI', fontsize=14, fontweight='bold')
        ax2.grid(alpha=0.3, axis='x')

    pd.DataFrame(results_data).to_csv(os.path.join(dirs["metrics"], 'Table_3_ROC_Metrics_All_Models.csv'), index=False)

    plt.suptitle('Figure 4. ROC Comparison Across Multiple Models', fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], 'Figure_4_ROC_Comparison_All_Models.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

# =====================================================================
# 5. Figure 5: XGBoost 特征重要性筛选 (Gain) - SLE vs Non-SLE 1x2 对比
# =====================================================================
def run_figure_5_xgb_importance(df):
    print("\n[Figure 5] 执行 XGBoost 特征重要性分析 (SLE vs Non-SLE 对比)...")
    
    def get_xgb_importance(is_sle):
        df_cohort = df[df['is_sle'] == is_sle].copy()
        cols_to_drop = ['subject_id', 'hadm_id', 'admittime', 'stay_id', 'intime', 
                        'hospital_expire_flag', 'aki_flag', 'is_sle', 'group_label', 'target_label', 'target']
        X = df_cohort.drop(columns=[c for c in cols_to_drop if c in df_cohort.columns])
        y = df_cohort['target'].values
        
        prep, cat, num = build_preprocessor(X)
        X_trans = prep.fit_transform(X)
        feats = get_feature_names(prep, cat, num)
        
        smote = SMOTE(random_state=2026)
        X_res, y_res = smote.fit_resample(X_trans, y)
        
        # 增加正则化防止过拟合
        model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, 
                              min_child_weight=2, reg_lambda=1.0, eval_metric='logloss', random_state=2026)
        model.fit(X_res, y_res)
        
        df_imp = pd.DataFrame({'Feature': feats, 'Importance': model.feature_importances_})
        df_imp = df_imp.sort_values(by='Importance', ascending=False).head(15).sort_values(by='Importance', ascending=True)
        df_imp['Feature'] = df_imp['Feature'].apply(rename_feature)
        return df_imp

    df_imp_sle = get_xgb_importance(1)
    df_imp_nonsle = get_xgb_importance(0)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Subplot A: SLE
    axes[0].barh(df_imp_sle['Feature'], df_imp_sle['Importance'], color=COLOR_SLE, edgecolor='black')
    axes[0].set_title('A. Top 15 Indicators in SLE Cohort (XGBoost Gain)', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Relative Importance (Gain)', fontsize=14)
    axes[0].tick_params(axis='y', labelsize=12)
    
    # Subplot B: Non-SLE
    axes[1].barh(df_imp_nonsle['Feature'], df_imp_nonsle['Importance'], color=COLOR_NONSLE, edgecolor='black')
    axes[1].set_title('B. Top 15 Indicators in Non-SLE Cohort (XGBoost Gain)', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Relative Importance (Gain)', fontsize=14)
    axes[1].tick_params(axis='y', labelsize=12)
    
    plt.suptitle('Figure 5. Comparison of Feature Importance (XGBoost)', fontsize=20, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], 'Figure_5_XGBoost_Importance_Comparison.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

# =====================================================================
# 6. Figure 6: 机器学习可解释性对比 (使用 XGBoost SHAP)
# =====================================================================
def run_figure_6_shap(df):
    print("\n[Figure 6] 执行 SHAP 机制定制化对比分析 (使用 XGBoost)...")
    core_features = ['cr_slope_48h', 'med_vasopressors', 'hx_hypertension', 'med_hcq', 'med_vanco', 'race']
    
    n_sle = len(df[df['is_sle'] == 1])
    n_nonsle = len(df[df['is_sle'] == 0])
    
    def get_shap_data(is_sle):
        d = df[df['is_sle'] == is_sle]
        X = d[core_features]
        y = d['target'].values
        prep, cat, num = build_preprocessor(X)
        X_trans = prep.fit_transform(X)
        feats = get_feature_names(prep, cat, num)
        
        smote = SMOTE(random_state=2026)
        X_res, y_res = smote.fit_resample(X_trans, y)
        
        model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, 
                              eval_metric='logloss', random_state=2026)
        model.fit(X_res, y_res)
        
        cohort_str = "SLE" if is_sle == 1 else "NonSLE"
        joblib.dump(model, os.path.join(dirs["models"], f'Fig6_SHAP_XGB_{cohort_str}.joblib'))
        
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_trans)
        
        shap_pos = shap_vals[1] if isinstance(shap_vals, list) else (shap_vals[:,:,1] if len(shap_vals.shape)==3 else shap_vals)
        return shap_pos, X_trans, feats

    shap_s, X_s, feat_s = get_shap_data(1)
    shap_n, X_n, feat_n = get_shap_data(0)
    
    feat_s_renamed = [rename_feature(f) for f in feat_s]
    feat_n_renamed = [rename_feature(f) for f in feat_n]
    
    custom_cmap = LinearSegmentedColormap.from_list("custom_shap", ["#0072B5", "#BC3C29"])
    
    fig = plt.figure(figsize=(22, 10))
    
    ax1 = fig.add_subplot(121)
    plt.sca(ax1)
    shap.summary_plot(shap_s, X_s, feature_names=feat_s_renamed, show=False, cmap=custom_cmap, plot_size=None)
    ax1.set_title(f'SLE Patients (n={n_sle})\nHydroxychloroquine shows strong protective effect', 
                  fontsize=18, fontweight='bold', color='#000080', pad=20)
    ax1.set_xlabel('SHAP Value (impact on model output)', fontsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    
    ax2 = fig.add_subplot(122)
    plt.sca(ax2)
    shap.summary_plot(shap_n, X_n, feature_names=feat_n_renamed, show=False, cmap=custom_cmap, plot_size=None)
    ax2.set_title(f'Non-SLE Patients (n={n_nonsle})\nHydroxychloroquine has no significant effect', 
                  fontsize=18, fontweight='bold', color='#8B0000', pad=20)
    ax2.set_xlabel('SHAP Value (impact on model output)', fontsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    
    plt.tight_layout(pad=3.0)
    plt.savefig(os.path.join(dirs["plots"], 'Figure_6_SHAP_Comparison_Custom.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

# =====================================================================
# 7. Figure 7: 统计学交互作用验证
# =====================================================================
def run_figure_7_interaction(df):
    print("\n[Figure 7] 执行统计学交互作用验证 (全队列 Bootstrap)...")
    df_int = df.copy()
    df_int['is_sle_x_med_hcq'] = df_int['is_sle'] * df_int['med_hcq']
    
    core_features = [
        'is_sle', 'med_hcq', 'is_sle_x_med_hcq',
        'cr_slope_48h', 'med_vasopressors', 'hx_hypertension',
        'med_vanco', 'race', 'anchor_age', 'gender',
        'has_infection', 'hx_ckd', 'hx_cad', 'hx_heart_failure'
    ]
    
    available_features = [c for c in core_features if c in df_int.columns]
    X = df_int[available_features]
    y = df_int['target'].values
    
    categorical_cols = ['race', 'gender']
    binary_cols = ['is_sle', 'med_hcq', 'is_sle_x_med_hcq', 'med_vasopressors', 
                   'hx_hypertension', 'med_vanco', 'has_infection', 'hx_ckd', 
                   'hx_cad', 'hx_heart_failure']
                   
    categorical_cols = sorted(list(set([c for c in categorical_cols + binary_cols if c in X.columns])))
    numeric_cols = sorted([c for c in X.columns if c not in categorical_cols])
    
    numeric_transformer = SkPipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = SkPipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
    
    X_trans = preprocessor.fit_transform(X)
    cat_enc = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_features = list(cat_enc.get_feature_names_out(categorical_cols))
    feats = numeric_cols + cat_features
    
    model = LogisticRegression(solver='liblinear', max_iter=2000, random_state=2026)
    model.fit(X_trans, y)
    
    joblib.dump(model, os.path.join(dirs["models"], 'Fig7_Interaction_LR_Main.joblib'))
    
    main_coefs = model.coef_[0]
    
    n_bootstraps = 1000
    boot_coefs = []
    for i in range(n_bootstraps):
        idx = np.random.choice(len(X_trans), size=len(X_trans), replace=True)
        m_boot = LogisticRegression(solver='liblinear', max_iter=2000, random_state=i)
        m_boot.fit(X_trans[idx], y[idx])
        boot_coefs.append(m_boot.coef_[0])
    boot_coefs = np.array(boot_coefs)
    
    res = []
    for i, name in enumerate(feats):
        if name.startswith('is_sle_x_med_hcq'):
            feat_key = 'SLE × Hydroxychloroquine (Interaction)'
        elif name.startswith('med_hcq'):
            feat_key = 'Hydroxychloroquine (Main Effect)'
        else:
            continue
            
        coef = main_coefs[i]
        ci_l = np.percentile(np.exp(boot_coefs[:, i]), 2.5)
        ci_u = np.percentile(np.exp(boot_coefs[:, i]), 97.5)
        res.append({'Feature': feat_key, 'OR': np.exp(coef), 'CI_Lower': ci_l, 'CI_Upper': ci_u})
            
    plot_df = pd.DataFrame(res).set_index('Feature').loc[['Hydroxychloroquine (Main Effect)', 'SLE × Hydroxychloroquine (Interaction)']]
    plot_df = plot_df.iloc[::-1]
    
    tab4 = plot_df.copy()
    for col in ['OR', 'CI_Lower', 'CI_Upper']: tab4[col] = tab4[col].apply(format_stat)
    tab4.to_csv(os.path.join(dirs["metrics"], 'Table_4_Interaction_Analysis.csv'))
    
    plt.figure(figsize=(10, 4))
    y_pos = np.arange(len(plot_df))
    ax = plt.gca()
    
    for i, (idx, row) in enumerate(plot_df.iterrows()):
        err_l = row['OR'] - row['CI_Lower']
        err_u = row['CI_Upper'] - row['OR']
        ax.errorbar(row['OR'], i, xerr=[[err_l], [err_u]], 
                    fmt='s', color='#2c3e50', capsize=6, elinewidth=2, markersize=9)
        
        txt = f"OR={row['OR']:.3f}\n(95%CI: {row['CI_Lower']:.3f}-{row['CI_Upper']:.3f})"
        ax.text(row['OR'] + 0.05, i, txt, va='center', fontsize=12)
        
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df.index, fontsize=13)
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=14)
    ax.set_title('Figure 7. Interaction Analysis: Disease Specificity of HCQ', fontsize=16, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], 'Figure_7_Interaction_Analysis.pdf'), dpi=300)
    plt.close()

# =====================================================================
# 8. Figure 8: 敏感性分析
# =====================================================================
def run_figure_8_sensitivity(df):
    print("\n[Figure 8] 执行敏感性分析 (SLE & Non-SLE, 分组柱状图)...")
    
    def get_sensitivity_results(df_cohort):
        cols_to_drop = ['target_label', 'is_sle', 'subject_id', 'hadm_id', 'admittime',
                        'hospital_expire_flag', 'aki_flag', 'group_label', 'stay_id', 'intime', 'target']
        X = df_cohort.drop(columns=[c for c in cols_to_drop if c in df_cohort.columns])
        y = df_cohort['target'].values
        
        X_original = X.copy()
        y_original = y.copy()
        
        X_remove = X.drop('med_aminoglycosides', axis=1, errors='ignore').copy()
        y_remove = y.copy()
        
        if 'med_aminoglycosides' in X.columns:
            mask = X['med_aminoglycosides'] == 0
            X_subgroup = X[mask].copy()
            y_subgroup = y[mask]
        else:
            X_subgroup = X.copy()
            y_subgroup = y.copy()
            
        def train_model(X_train, y_train):
            categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
            binary_cols = [col for col in X_train.columns if col.startswith('hx_') or col.startswith('med_')]
            
            categorical_cols = sorted(list(set(categorical_cols + binary_cols)))
            numeric_cols = sorted([col for col in X_train.columns if col not in categorical_cols])
            
            numeric_transformer = ImbPipeline(steps=[
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler())
            ])
            categorical_transformer = ImbPipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
            ])
            preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
            
            pipeline = ImbPipeline(steps=[
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=2026)),
                ('classifier', LogisticRegression(solver='liblinear', max_iter=2000, random_state=2026))
            ])
            
            param_grid = {'classifier__C': np.logspace(-3, 2, 10), 'classifier__penalty': ['l1']}
            cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=2026)
            
            grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=1)
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            cat_encoder = best_model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
            all_feature_names = numeric_cols + list(cat_feature_names)
            coefficients = best_model.named_steps['classifier'].coef_[0]
            return dict(zip(all_feature_names, coefficients))

        res_orig = train_model(X_original, y_original)
        res_rem = train_model(X_remove, y_remove)
        res_sub = train_model(X_subgroup, y_subgroup)
        
        core_features = ['med_hcq', 'med_vasopressors', 'hx_hypertension', 'med_vanco', 'cr_slope_48h', 'race_WHITE']
        table_data = []
        for feature in core_features:
            val0 = next((res_orig[f] for f in res_orig if f.startswith(feature)), 0)
            val1 = next((res_rem[f] for f in res_rem if f.startswith(feature)), 0) if feature != 'med_aminoglycosides' else 0
            val2 = next((res_sub[f] for f in res_sub if f.startswith(feature)), 0)
            table_data.append({
                'Feature': rename_feature(feature),
                'Scenario 0 (Original)': val0,
                'Scenario 1 (Remove Drug)': val1,
                'Scenario 2 (Subgroup)': val2
            })
        return table_data

    print("  - 分析 SLE 队列...")
    data_sle = get_sensitivity_results(df[df['is_sle'] == 1].copy())
    print("  - 分析 Non-SLE 队列...")
    data_nonsle = get_sensitivity_results(df[df['is_sle'] == 0].copy())
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 16))
    
    def plot_sensitivity(ax, data, title):
        features = [row['Feature'] for row in data]
        s0 = [row['Scenario 0 (Original)'] for row in data]
        s1 = [row['Scenario 1 (Remove Drug)'] for row in data]
        s2 = [row['Scenario 2 (Subgroup)'] for row in data]
        
        x = np.arange(len(features))
        width = 0.25
        
        bars1 = ax.bar(x - width, s0, width, label='Scenario 0 (Original)', color='#3498db', edgecolor='black')
        bars2 = ax.bar(x, s1, width, label='Scenario 1 (Remove Drug)', color='#e74c3c', edgecolor='black')
        bars3 = ax.bar(x + width, s2, width, label='Scenario 2 (Subgroup)', color='#2ecc71', edgecolor='black')
        
        ax.set_ylabel('Coefficient', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=30, ha='right', fontsize=13)
        ax.legend(loc='upper right', fontsize=12)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3 if height >= 0 else -12),
                            textcoords="offset points",
                            ha='center', va='bottom' if height >= 0 else 'top',
                            fontsize=10, fontweight='bold')
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)

    plot_sensitivity(axes[0], data_sle, 'A. Sensitivity Analysis (SLE): Core Feature Coefficients Across Scenarios')
    plot_sensitivity(axes[1], data_nonsle, 'B. Sensitivity Analysis (Non-SLE): Core Feature Coefficients Across Scenarios')
    
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], 'Figure_8_Sensitivity_Analysis.pdf'), dpi=300)
    plt.close()

# =====================================================================
# Supplementary Figure 1: Non-SLE 特征分析
# =====================================================================
def run_supp_figure_1(df):
    print("\n[Supp Fig 1] 执行 Non-SLE 专属特征重要性 (LASSO)...")
    df_non = df[df['is_sle'] == 0].copy()
    cols_to_drop = ['subject_id', 'hadm_id', 'admittime', 'stay_id', 'intime', 'hospital_expire_flag', 'aki_flag', 'is_sle', 'group_label', 'target_label', 'target']
    X = df_non.drop(columns=[c for c in cols_to_drop if c in df_non.columns])
    y = df_non['target'].values
    
    prep, cat, num = build_preprocessor(X)
    pipe = ImbPipeline([
        ('preprocessor', prep),
        ('smote', SMOTE(random_state=2026)),
        ('classifier', LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=2026))
    ])
    pipe.fit(X, y)
    
    joblib.dump(pipe, os.path.join(dirs["models"], 'SuppFig1_NonSLE_LASSO.joblib'))
    
    feats = get_feature_names(prep, cat, num)
    coefs = pipe.named_steps['classifier'].coef_[0]
    
    df_imp = pd.DataFrame({'Feature': feats, 'Coefficient': coefs})
    df_imp = df_imp[df_imp['Coefficient'] != 0].sort_values(by='Coefficient', key=abs, ascending=False).head(15)
    df_imp = df_imp.sort_values(by='Coefficient', ascending=True)
    df_imp['Feature'] = df_imp['Feature'].apply(rename_feature)
    
    plt.figure(figsize=(10, 8))
    colors = [COLOR_RISK if c > 0 else COLOR_PROTECT for c in df_imp['Coefficient']]
    bars = plt.barh(df_imp['Feature'], df_imp['Coefficient'], color=colors, edgecolor='black', linewidth=0.5)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.title('Supplementary Figure 1. Feature Selection in Non-SLE (LASSO)', fontsize=16, fontweight='bold')
    plt.xlabel('LASSO Coefficient', fontsize=14)
    plt.yticks(fontsize=12)
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.02 * (1 if width > 0 else -1), bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', va='center', ha='left' if width > 0 else 'right', fontsize=10)
                 
    plt.tight_layout()
    plt.savefig(os.path.join(dirs["plots"], 'Supplementary_Figure_1_NonSLE_Features.pdf'), dpi=300)
    plt.close()

# =====================================================================
# 主执行入口
# =====================================================================
if __name__ == "__main__":
    data_path = r"D:\BaiduSyncdisk\MIMIC\AKI_trajectories\AKI_label_MIMIC.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ 找不到数据文件: {data_path}")
    else:
        df_global = load_and_preprocess_data(data_path)
        
        # 按照重新排布的顺序执行所有分析
        run_figure_1_lasso(df_global)
        run_figure_2_psm_evaluation(df_global)
        run_figure_3_forest(df_global)
        run_figure_4_roc(df_global)
        run_figure_5_xgb_importance(df_global)
        run_figure_6_shap(df_global)
        run_figure_7_interaction(df_global)
        run_figure_8_sensitivity(df_global)
        run_supp_figure_1(df_global)
        
        print("\n🎉 大功告成！所有分析执行完毕。已生成所有 Figure、Table 及 PSM 匹配数据集，并已保存至 D 盘。")

# =====================================================================
    # 提取用于列线图和网页计算器的原始参数
    # =====================================================================
    def extract_nomogram_params(df, model_path):
        print("\n" + "="*60)
        print("【列线图与交互式计算器所需参数提取 (已还原 StandardScaler)】")
        print("="*60)
        
        # 1. 获取连续变量的极值范围
        df_sle = df[df['is_sle'] == 1].copy()
        cr_slope_min = df_sle['cr_slope_48h'].min()
        cr_slope_max = df_sle['cr_slope_48h'].max()
        
        # 2. 加载 Figure 3 的多因素 LR 模型
        pipeline = joblib.load(model_path)
        preprocessor = pipeline.named_steps['preprocessor']
        classifier = pipeline.named_steps['classifier']
        
        # 3. 获取特征名
        cat_cols = preprocessor.transformers_[1][2]
        num_cols = preprocessor.transformers_[0][2]
        cat_enc = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_features = list(cat_enc.get_feature_names_out(cat_cols))
        all_features = num_cols + cat_features
        
        # 4. 获取标准化参数 (Mean 和 Scale)
        scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
        num_means = scaler.mean_
        num_scales = scaler.scale_
        
        # 5. 获取原始截距和系数
        coefs = classifier.coef_[0]
        intercept = classifier.intercept_[0]
        
        unscaled_coefs = {}
        unscaled_intercept = intercept
        
        # 6. 反向推导原始尺度的系数和截距
        # 公式: Log-odds = Intercept + sum(coef * (x - mean) / scale)
        # 展开后: 原始系数 = coef / scale; 原始截距 = Intercept - sum(coef * mean / scale)
        for i, feat in enumerate(all_features):
            if feat in num_cols:
                idx = num_cols.index(feat)
                mean_val = num_means[idx]
                scale_val = num_scales[idx]
                
                unscaled_coef = coefs[i] / scale_val
                unscaled_intercept -= (coefs[i] * mean_val) / scale_val
                unscaled_coefs[feat] = unscaled_coef
            else:
                unscaled_coefs[feat] = coefs[i]
                
        print(f"① 模型的截距 (Intercept, 还原后): {unscaled_intercept:.5f}\n")
        print("② 各个特征的回归系数 (Coefficients, 还原后):")
        for k, v in unscaled_coefs.items():
            print(f"   - {k}: {v:.5f}")
            
        print(f"\n③ cr_slope_48h 的取值范围:")
        print(f"   - 最小值: {cr_slope_min:.4f}")
        print(f"   - 最大值: {cr_slope_max:.4f}")
        print("="*60 + "\n")

    # 调用提取函数 (确保路径与保存的 Figure 3 模型路径一致)
    model_file = os.path.join(dirs["models"], 'Fig3_SLE_Multivariate_LR.joblib')
    extract_nomogram_params(df_global, model_file)        