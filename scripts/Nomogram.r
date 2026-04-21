# 安装并加载必要的包
if(!require(rms)) install.packages("rms")
library(rms)


set.seed(2026)
n <- 1000
cr_slope_48h <- runif(n, -0.55, 0.15)
hx_hypertension <- sample(c(0, 1), n, replace=TRUE)
med_hcq <- sample(c(0, 1), n, replace=TRUE)
med_vanco <- sample(c(0, 1), n, replace=TRUE)
med_vasopressors <- sample(c(0, 1), n, replace=TRUE)
race <- sample(c("ASIAN", "WHITE", "BLACK", "HISPANIC", "OTHER"), n, replace=TRUE)

log_odds <- -3.41856 + 
  (39.14527 * cr_slope_48h) + 
  (1.58920 * hx_hypertension) + 
  (-1.17393 * med_hcq) + 
  (1.86603 * med_vanco) + 
  (2.25173 * med_vasopressors) +
  ifelse(race=="WHITE", -1.80337, 
  ifelse(race=="BLACK", -1.63859, 
  ifelse(race=="HISPANIC", 0.70632, 
  ifelse(race=="OTHER", -0.39582, 0))))

prob <- 1 / (1 + exp(-log_odds))
target <- rbinom(n, 1, prob)

df <- data.frame(cr_slope_48h, hx_hypertension, med_hcq, med_vanco, med_vasopressors, race, target)
df$race <- factor(df$race, levels=c("ASIAN", "WHITE", "BLACK", "HISPANIC", "OTHER"))

# 2. 添加标签
label(df$cr_slope_48h) <- "48h Creatinine Slope"
label(df$hx_hypertension) <- "History of Hypertension"
label(df$med_hcq) <- "Hydroxychloroquine Use"
label(df$med_vanco) <- "Vancomycin Use"
label(df$med_vasopressors) <- "Vasopressors Use"
label(df$race) <- "Race"

dd <- datadist(df)
options(datadist="dd")

fit <- lrm(target ~ cr_slope_48h + hx_hypertension + med_hcq + med_vanco + med_vasopressors + race, data=df)

# 3. 绘图并保存
pdf("Nomogram_SLE_AKI_Perfect.pdf", width=11, height=8)

# fun.at 只保留 0.1, 0.5, 0.9，拉开物理间距
nom <- nomogram(fit, 
                fun=function(x)1/(1+exp(-x)), 
                fun.at=c(0.1, 0.5, 0.9), 
                funlabel="Probability of AKI Progression",
                lp=FALSE)

# cex.axis=0.85 缩小坐标轴数字大小，防止任何可能的边缘触碰
plot(nom, xfrac=0.35, cex.axis=0.85, cex.var=1.0) 
dev.off()

print("列线图已保存为 Nomogram_SLE_AKI.pdf")