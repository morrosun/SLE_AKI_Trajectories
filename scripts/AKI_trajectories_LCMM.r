# ---------------------------------------------------------
# 0. 加载必要的包与工作目录设置
# ---------------------------------------------------------
library(readr)
library(lcmm)
library(tidymodels)
library(xgboost)
library(themis)    
library(vip)       
library(shapviz)  
library(yardstick) 
library(ggplot2)
library(dplyr)

# 设置工作目录 
setwd("D:/BaiduSyncdisk/MIMIC/AKI_trajectories")

# ---------------------------------------------------------
# 1. 纵向数据清洗与 LCMM 轨迹提取
# ---------------------------------------------------------
longitudinal_cr <- read_csv("longitudinal_cr_MIMIC.csv")

longitudinal_cr$stay_id <- as.numeric(longitudinal_cr$stay_id)
longitudinal_cr$time_window <- as.numeric(longitudinal_cr$time_window)

longitudinal_cr_clean <- longitudinal_cr %>%
  filter(time_window <= 7) %>%
  filter(!is.na(creatinine)) %>%
  group_by(stay_id) %>%
  mutate(obs_count = n()) %>%
  ungroup() %>%
  filter(obs_count >= 3) %>%
  mutate(log_cr = log(creatinine + 0.01))

# 拟合基础模型与网格搜索最优 3 类别模型
m1_log <- hlme(log_cr ~ time_window, random = ~ 1, subject = 'stay_id', data = longitudinal_cr_clean)

set.seed(123)
m3_grid <- gridsearch(
  rep = 15, maxiter = 10, minit = m1_log,
  hlme(log_cr ~ time_window, mixture = ~ time_window, random = ~ 1, 
       classmb = ~ 1, ng = 3, subject = 'stay_id', data = longitudinal_cr_clean)
)

# ---------------------------------------------------------
# 2. 标签映射与基线特征合并
# ---------------------------------------------------------
SLE_aki_features <- read_csv("SLE_aki_features.csv")

trajectory_labels <- m3_grid$pprob %>% 
  select(stay_id, class) %>%
  mutate(
    target_label = case_when(
      class == 1 ~ "Progressive", 
      class == 2 ~ "Stable",      
      class == 3 ~ "Recovery"     
    ),
    target_label = factor(target_label, levels = c("Stable", "Progressive", "Recovery"))
  )

ml_dataset <- SLE_aki_features %>%
  inner_join(trajectory_labels %>% select(stay_id, target_label), by = "stay_id")

write_csv(ml_dataset, "AKI_label_MIMIC.csv")

# [输出 Table 1]: 队列类别分布
cat("\n=== Table 1: Final Cohort Class Distribution ===\n")
print(table(ml_dataset$target_label, useNA = "ifany"))

# ==============================================================================
# 2. 外部验证队列：使用 LCMM 和 eICU 纵向肌酐数据生成真实的轨迹标签 (Ground Truth)
# ==============================================================================
cat("\n==========================================================\n")
cat("开始处理 eICU 外部验证集并生成真实轨迹标签...\n")

# ---------------------------------------------------------
# 步骤 2.1: 加载并清洗 eICU 纵向肌酐数据
# ---------------------------------------------------------
# 请确保路径正确
longitudinal_cr_eICU <- read_csv("D:/BaiduSyncdisk/MIMIC/AKI_trajectories/longitudinal_cr_eICU.csv")

# 统一主键名称为 stay_id 
if("patientunitstayid" %in% colnames(longitudinal_cr_eICU)) {
  longitudinal_cr_eICU <- longitudinal_cr_eICU %>% rename(stay_id = patientunitstayid)
}

longitudinal_cr_eICU$stay_id <- as.numeric(longitudinal_cr_eICU$stay_id)
longitudinal_cr_eICU$time_window <- as.numeric(longitudinal_cr_eICU$time_window)

cat("正在清洗 eICU 纵向数据 (与 MIMIC 保持相同标准)...\n")
longitudinal_cr_eICU_clean <- longitudinal_cr_eICU %>%
  filter(time_window <= 7) %>%
  filter(!is.na(creatinine)) %>%
  group_by(stay_id) %>%
  mutate(obs_count = n()) %>%
  ungroup() %>%
  filter(obs_count >= 3) %>%
  mutate(log_cr = log(creatinine + 0.01))

# ---------------------------------------------------------
# 步骤 2.2: 使用 MIMIC 训练好的 LCMM 模型 (m3_grid) 分配真实轨迹
# ---------------------------------------------------------
cat("🤖 正在使用预训练的 LCMM 模型为 eICU 分配真实轨迹 (Ground Truth)...\n")
# lcmm 包的 predictClass 函数可以根据已有的模型，为新患者计算后验概率并分类
eicu_class_pred <- predictClass(m3_grid, newdata = longitudinal_cr_eICU_clean)

# 映射标签 (保持与 MIMIC 完全一致的类别定义)
eicu_trajectory_labels <- eicu_class_pred %>%
  select(stay_id, class) %>%
  mutate(
    target_label = case_when(
      class == 1 ~ "Progressive", 
      class == 2 ~ "Stable",      
      class == 3 ~ "Recovery"     
    ),
    target_label = factor(target_label, levels = c("Stable", "Progressive", "Recovery"))
  )
# ---------------------------------------------------------
# 步骤 2.3: 加载新提取的完整 eICU 基线特征并与真实标签合并
# ---------------------------------------------------------
# 读取刚刚用 SQL 跑出来的完整特征表
eicu_features_path <- "full_eicu_features.csv" 
cat("正在读取完整的 eICU 特征表...\n")
eicu_features <- read_csv(eicu_features_path)

eicu_features_aligned <- eicu_features

# 统一 ID 列名。SQL 导出的叫 patientunitstayid，改为 stay_id
if("patientunitstayid" %in% colnames(eicu_features_aligned)) {
  eicu_features_aligned <- eicu_features_aligned %>% rename(stay_id = patientunitstayid)
}

# 统一 SLE 列名。
if("SLE" %in% colnames(eicu_features_aligned)) {
  eicu_features_aligned <- eicu_features_aligned %>% rename(is_sle = SLE)
} else if ("sle" %in% colnames(eicu_features_aligned)) {
  eicu_features_aligned <- eicu_features_aligned %>% rename(is_sle = sle)
}

if (!"hadm_id" %in% colnames(eicu_features_aligned)) {
  eicu_features_aligned <- eicu_features_aligned %>% mutate(hadm_id = stay_id)
}
if (!"subject_id" %in% colnames(eicu_features_aligned)) {
  eicu_features_aligned <- eicu_features_aligned %>% mutate(subject_id = stay_id)
}

# 对齐 Race 
if ("ethnicity" %in% colnames(eicu_features_aligned) && !"race" %in% colnames(eicu_features_aligned)) {
  eicu_features_aligned <- eicu_features_aligned %>% rename(race = ethnicity)
}

if ("race" %in% colnames(eicu_features_aligned)) {
  eicu_features_aligned <- eicu_features_aligned %>%
    mutate(
      race = toupper(race),
      race = case_when(
        grepl("CAUCASIAN|WHITE", race) ~ "WHITE",
        grepl("AFRICAN AMERICAN|BLACK", race) ~ "BLACK",
        grepl("HISPANIC", race) ~ "HISPANIC",
        grepl("ASIAN", race) ~ "ASIAN",
        TRUE ~ "OTHER"
      )
    )
}

# 合并特征与 LCMM 生成的真实标签
cat("正在将基线特征与 LCMM 轨迹标签合并...\n")
eicu_final_dataset <- eicu_features_aligned %>%
  inner_join(eicu_trajectory_labels %>% select(stay_id, target_label), by = "stay_id") %>%
  mutate(
    target_label = as.character(target_label),
    target = ifelse(target_label == "Progressive", 1, 0),
    group_label = target_label
  )

# ---------------------------------------------------------
# 步骤 8.4: 导出最终的 eICU 标签文件
# ---------------------------------------------------------
output_eicu_path <- "AKI_label_eICU.csv"
write_csv(eicu_final_dataset, output_eicu_path)

cat("🎉 eICU 真实标签与特征合并完成！\n")
cat("最终 eICU 分析数据集已成功保存至: ", output_eicu_path, "\n")
cat("最终数据集总人数: ", nrow(eicu_final_dataset), "\n")

# 打印 eICU 队列的真实轨迹分布情况
cat("\n📊 eICU 真实轨迹分布概况 (Ground Truth):\n")
print(table(eicu_final_dataset$target_label))

if ("is_sle" %in% colnames(eicu_final_dataset)) {
  cat("\n📊 eICU 中 SLE 与 Non-SLE 的真实轨迹分布:\n")
  print(table(eicu_final_dataset$is_sle, eicu_final_dataset$target_label))
}
cat("==========================================================\n")