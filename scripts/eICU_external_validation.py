import os
import glob
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 路径配置
# ==========================================
mimic_root_dir = r"D:\BaiduSyncdisk\MIMIC\AKI_trajectories"
eicu_data_path = r"D:\BaiduSyncdisk\MIMIC\AKI_trajectories\AKI_label_eICU.csv"

run_dirs = glob.glob(os.path.join(mimic_root_dir, "run_*"))
latest_run_dir = max(run_dirs, key=os.path.getmtime)
model_dir = os.path.join(latest_run_dir, "models")
plot_dir = os.path.join(latest_run_dir, "plots")

print(f"📂 当前使用的模型文件夹: {model_dir}")

# ==========================================
# 2. 加载 eICU 数据与预处理
# ==========================================
df_eicu_raw = pd.read_csv(eicu_data_path, encoding='latin-1')

# 统一 race 格式 
def merge_race_categories(race):
    if pd.isna(race): return 'OTHER'
    race = str(race).upper()
    if 'WHITE' in race: return 'WHITE'
    elif 'BLACK' in race or 'AFRICAN' in race: return 'BLACK'
    elif 'HISPANIC' in race or 'LATINO' in race: return 'HISPANIC'
    elif 'ASIAN' in race: return 'ASIAN'
    else: return 'OTHER'

if 'race' in df_eicu_raw.columns:
    df_eicu_raw['race'] = df_eicu_raw['race'].apply(merge_race_categories)
else:
    df_eicu_raw['race'] = 'OTHER'

if 'target' not in df_eicu_raw.columns and 'target_label' in df_eicu_raw.columns:
    df_eicu_raw['target'] = df_eicu_raw['target_label'].apply(lambda x: 1 if x == 'Progressive' else 0)

# 核心特征
core_features = ['cr_slope_48h', 'med_vasopressors', 'hx_hypertension', 'med_hcq', 'med_vanco', 'race']

# 确保 eICU 数据包含这些特征，缺失的补齐
for col in core_features:
    if col not in df_eicu_raw.columns:
        print(f"⚠️ eICU 数据缺少列: {col}，已自动填充默认值。")
        df_eicu_raw[col] = 'OTHER' if col == 'race' else 0

COLOR_SLE = '#BC3C29'
COLOR_NONSLE = '#0072B5'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

def format_pval(p): return "<0.001" if p < 0.001 else f"{p:.3f}"

def get_bootstrap_metrics(y_true, y_pred_proba, n_bootstraps=1000, seed=2026):
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_pred_proba), len(y_pred_proba))
        if len(np.unique(y_true[idx])) < 2: continue
        aucs.append(roc_auc_score(y_true[idx], y_pred_proba[idx]))
    mean_auc = roc_auc_score(y_true, y_pred_proba)
    return mean_auc, np.percentile(aucs, 2.5), np.percentile(aucs, 97.5), np.std(aucs)

# ==========================================
# 3. PSM (倾向性评分匹配) - 匹配 SLE 与 Non-SLE
# ==========================================
def perform_psm(df, treatment_col='is_sle', covariates=None):
    print("⚖️ 正在进行 eICU 数据的倾向性评分匹配 (PSM 1:1)...")
    df_psm = df.copy()
    
    # 填补缺失值用于计算 PS
    for col in covariates:
        if pd.api.types.is_numeric_dtype(df_psm[col]):
            df_psm[col] = df_psm[col].fillna(0)
        else:
            df_psm[col] = df_psm[col].fillna('Unknown')
            
    X_psm_dummy = pd.get_dummies(df_psm[covariates], drop_first=True)
    
    ps_model = LogisticRegression(random_state=2026, max_iter=1000, solver='liblinear')
    ps_model.fit(X_psm_dummy, df_psm[treatment_col])
    df_psm['propensity_score'] = ps_model.predict_proba(X_psm_dummy)[:, 1]
    
    treated = df_psm[df_psm[treatment_col] == 1].copy()
    control = df_psm[df_psm[treatment_col] == 0].copy()
    
    if len(treated) == 0 or len(control) == 0:
        print("⚠️ 警告: 某一组样本量为0，跳过 PSM。")
        return df
        
    nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree')
    nn.fit(control[['propensity_score']])
    distances, indices = nn.kneighbors(treated[['propensity_score']])
    
    matched_control_indices = control.iloc[indices.flatten()].index
    matched_control = df_psm.loc[matched_control_indices]
    
    matched_df = pd.concat([treated, matched_control]).drop_duplicates()
    print(f"✅ PSM 完成: SLE组 n={len(treated)}, 匹配后的Non-SLE组 n={len(matched_control)}")
    return matched_df

# 使用核心特征进行 PSM 匹配
psm_covariates = [c for c in core_features if c != 'race'] # 简化匹配变量
df_eicu_matched = perform_psm(df_eicu_raw, treatment_col='is_sle', covariates=psm_covariates)

# ==========================================
# 4. 评估并绘制 ROC 对比图
# ==========================================
models_to_test = [
    ('Lasso', 'Lasso'),                             
    ('LogisticRegression', 'Logistic Regression'),  
    ('RandomForest', 'Random Forest'),              
    ('XGBoost', 'XGBoost')                          
]

fig, axes = plt.subplots(4, 2, figsize=(12, 20))

for i, (model_file_name, display_name) in enumerate(models_to_test):
    letter_roc = chr(65 + i * 2)      
    letter_auc = chr(65 + i * 2 + 1)  
    ax1 = axes[i, 0]
    ax2 = axes[i, 1]
    ax1.set_title(f'{letter_roc}. {display_name} ROC Comparison', fontsize=14, fontweight='bold', loc='left')
    ax2.set_title(f'{letter_auc}. {display_name} AUC with 95% CI', fontsize=14, fontweight='bold', loc='left')
    
    print(f"🚀 正在评估: {display_name}...")
    roc_data, metrics = {}, {}
    
    for is_sle, cohort_name in [(1, 'SLE'), (0, 'NonSLE')]:
        df_sub = df_eicu_matched[df_eicu_matched['is_sle'] == is_sle].copy()
        if len(df_sub) == 0: continue
            
        # 提取核心特征
        X = df_sub[core_features]
        y = df_sub['target'].values
        
        # 加载保存的完整 Pipeline 模型
        model_file = os.path.join(model_dir, f'Fig4_{model_file_name}_{cohort_name}.joblib')
        
        if os.path.exists(model_file):
            model_pipeline = joblib.load(model_file)
            
            # 直接使用 Pipeline 预测
            y_pred_proba = model_pipeline.predict_proba(X)[:, 1]
            
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            auc_val, ci_l, ci_u, std_auc = get_bootstrap_metrics(y, y_pred_proba)
            roc_data[cohort_name] = (fpr, tpr)
            metrics[cohort_name] = (auc_val, ci_l, ci_u, std_auc)
        else:
            print(f"❌ 找不到模型文件: {model_file}")
            ax1.text(0.5, 0.5, 'Model Not Found', ha='center', va='center', color='red', fontsize=14)
            
    if 'SLE' in metrics and 'NonSLE' in metrics:
        z_score = abs(metrics['SLE'][0] - metrics['NonSLE'][0]) / np.sqrt(metrics['SLE'][3]**2 + metrics['NonSLE'][3]**2)
        p_val = 2 * (1 - norm.cdf(z_score))
    else: p_val = 1.0

    # 绘制 ROC 曲线 (左列)
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.8)
    if 'SLE' in roc_data: ax1.plot(roc_data['SLE'][0], roc_data['SLE'][1], color=COLOR_SLE, lw=2.5, label=f"SLE (AUC={metrics['SLE'][0]:.3f})")
    if 'NonSLE' in roc_data: ax1.plot(roc_data['NonSLE'][0], roc_data['NonSLE'][1], color=COLOR_NONSLE, lw=2.5, label=f"Non-SLE (AUC={metrics['NonSLE'][0]:.3f})")
    ax1.set_xlim([-0.05, 1.05]); ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlabel('1 - Specificity', fontsize=12); ax1.set_ylabel('Sensitivity', fontsize=12)
    if roc_data:
        ax1.legend(loc="lower right", prop={'size': 11})
        ax1.text(0.3, 0.2, f'P-value = {format_pval(p_val)}', fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    ax1.grid(alpha=0.3)

    # 绘制 AUC 森林图 (右列)
    ax2.axvline(x=0.5, linestyle='--', color='gray', alpha=0.5)
    if 'SLE' in metrics:
        auc_s, ci_l_s, ci_u_s, _ = metrics['SLE']
        ax2.errorbar(auc_s, 0.5, xerr=[[auc_s - ci_l_s], [ci_u_s - auc_s]], fmt='o', color=COLOR_SLE, capsize=5, markersize=8)
        ax2.text(auc_s, 0.55, f'{auc_s:.3f}', ha='center', va='bottom', color=COLOR_SLE, fontweight='bold', fontsize=11)
    if 'NonSLE' in metrics:
        auc_n, ci_l_n, ci_u_n, _ = metrics['NonSLE']
        ax2.errorbar(auc_n, 0.3, xerr=[[auc_n - ci_l_n], [ci_u_n - auc_n]], fmt='o', color=COLOR_NONSLE, capsize=5, markersize=8)
        ax2.text(auc_n, 0.35, f'{auc_n:.3f}', ha='center', va='bottom', color=COLOR_NONSLE, fontweight='bold', fontsize=11)
    ax2.set_xlim([0.4, 1.0]); ax2.set_ylim([0, 1])
    ax2.set_xlabel('Area Under Curve', fontsize=12)
    ax2.set_yticks([0.3, 0.5]); ax2.set_yticklabels(['Non-SLE', 'SLE'], fontsize=12)
    ax2.grid(alpha=0.3, axis='x')

plt.suptitle('Figure 9. eICU External Validation ROC Comparison (Post-PSM)', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
out_plot = os.path.join(plot_dir, "Figure_9_eICU_External_Validation_Combined.pdf")
plt.savefig(out_plot, dpi=300, bbox_inches='tight')
plt.close()
print(f"🎉 绘图完成！请查看: {out_plot}")