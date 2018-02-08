library(ggplot2)
library(tidyverse)
library(reshape2)
library(stargazer)
library(sandwich)

## Params

filename_output <- "../html_tables/tables_all_market.html"
folder <- '~/Github/OrderRoutingAndPFOF/data/processed/'
table_name <- 'All Brokers'

## Data Prep ------------------------------------------------------------------

data_panel <- read_csv(paste(folder, 'regression_data_levels.csv', sep = ''))
data_fdiff <- read_csv(paste(folder, 'regression_data_fdiffs.csv', sep = ''))
data_panel_lag <- read_csv(paste(folder, 'regression_data_levels_lag.csv', sep = ''))
data_fdiff_lag <- read_csv(paste(folder, 'regression_data_fdiffs_lag.csv', sep = ''))
data_bin <- read_csv(paste(folder, 'regression_data_levels_binaries.csv', sep = ''))
data_dmd <- read_csv(paste(folder, 'regression_data_levels_demeaned.csv', sep = ''))
 
# Filter data

data_panel  <- na.omit(filter(data_panel, OrderType == "Market"))#, Rebate_Dummy == 1))
data_fdiff  <- na.omit(filter(data_fdiff, OrderType == "Market"))#, Rebate_Dummy == 1))
data_bin    <- na.omit(filter(data_bin,   OrderType == "Market"))#, Rebate_Dummy == 1))
data_dmd    <- na.omit(filter(data_dmd))#,                         Rebate_Dummy == 1))

## Funcs ----------------------------------------------------------------------

testLRT <- function(fit0, fit) {
  
  r = fit$dim$p - 1 
  wald_stat = -2*(fit0$logLik -fit$logLik)
  return(1-pchisq(wald_stat, r))
  
}

createHTMLTable <- function(fits, title_txt, omit_labs = NULL, 
                            output_type = "html", nospace = NULL,
                            dep_var_caption = NULL, se_input = NULL,
                            notes_txt = '*p<0.05, **p<0.01, ***p<0.001') {
  
  s = capture.output(stargazer(fits, title = title_txt, 
                               font.size = 'large',
                               omit = omit_labs,
                               dep.var.labels.include = FALSE,
                               dep.var.caption = dep_var_caption,
                               multicolumn = FALSE,
                               column.sep.width = '5em',
                               se = se_input,
                               no.space = nospace,
                               star.char = c("*", "**", "***"),
                               star.cutoffs = c(0.05, 0.01, 0.001),
                               notes = notes_txt,
                               notes.append = F,
                               type = output_type))
  
  return(s)
  
}

hausman_test <- function(fit_a, fit_b, r) {
  
  cov1 = data.matrix(vcov(fit_a))[2:r,2:r]
  cov2 = data.matrix(vcov(fit_b))[2:r,2:r]
  
  b1 = matrix(as.numeric(fit_a$coefficients))[2:r]
  b2 = matrix(as.numeric(fit_b$coefficients))[2:r]
  
  wald_stat = t(b1-b2) %*% solve(cov2-cov1) %*% (b1-b2)
  pval = 1-pchisq(wald_stat, r-1)
  print(paste0('Wald: ', wald_stat))
  return(pval)
  
}

## Regressions ----------------------------------------------------------------


# Some potential fits
# fit1.1a <- lm(MktShare ~ PrImp_Pct + PrImp_AvgAmt + PrImp_AvgT - 1, data = data_fit1)
# fit1.1b <- lm(MktShare ~ PrImp_Pct + PrImp_AvgAmt + PrImp_AvgT, data = data_fit1)
# fit1.3a <- lm(MktShare ~ PrImp_ExpAmt + PrImp_AvgT - 1, data = data_fit1)
# fit1.3b <- lm(MktShare ~ PrImp_ExpAmt + PrImp_AvgT, data = data_fit1)
# fit1.4a <- lm(MktShare ~ PrImp_ExpAmt + All_AvgT - 1, data = data_fit1)
# fit1.4b <- lm(MktShare ~ PrImp_ExpAmt + All_AvgT, data = data_fit1)
# fit1.2a <- lm(MktShare ~ Rel_PrImp_ExpAmt + Rel_All_AvgT - 1, data = data_fit1)
# fit1.2b <- lm(MktShare ~ Rel_PrImp_ExpAmt + Rel_All_AvgT , data = data_fit1)
# fit2.1 <- lm(MktShare ~ PrImp_Pct + PrImp_AvgAmt + PrImp_AvgT - 1, data = data_fit2)
# fit2.3 <- lm(MktShare ~ PrImp_ExpAmt + PrImp_AvgT - 1, data = data_fit2)
# fit2.3b <- lm(MktShare ~ PrImp_ExpAmt + All_AvgT - 1, data = data_fit2)
# fit2.2 <- lm(MktShare ~ Rel_PrImp_ExpAmt + Rel_All_AvgT - 1, data = data_fit2)

# Remove old output file file (will throw warning on first run)
file.remove(paste0(folder, filename_output))

### Formulas

fit0_formula = 'MktShare ~ 1'
fit1_formula = 'MktShare ~ PrImp_Pct + PrImp_AvgAmt + PrImp_AvgT'
fit2_formula = 'MktShare ~ PrImp_ExpAmt + PrImp_AvgT'
fit3_formula = 'MktShare ~ PrImp_Pct + PrImp_AvgAmt + All_AvgT'
fit4_formula = 'MktShare ~ PrImp_ExpAmt + All_AvgT'


###############################################################################
### OLS Regressions

fit0 = lm(as.formula(fit0_formula), data = data_panel)
fit1 = lm(as.formula(fit1_formula), data = data_panel)
fit2 = lm(as.formula(fit2_formula), data = data_panel)
fit3 = lm(as.formula(fit3_formula), data = data_panel)
fit4 = lm(as.formula(fit4_formula), data = data_panel)

fits = list(fit1, fit2, fit3, fit4)

print('LR Test results')

for (fit in fits) {

  print(testLRT(fit0, fit))

}

SEs = lapply(fits, function(x) {(sqrt(diag(vcovHC(x))))})
s = createHTMLTable(fits, paste(table_name, "(OLS)"), dep_var_caption = 'Market Share',
                    se = SEs)

cat(paste(s,"\n"),
    file=paste0(folder, filename_output),
    append=TRUE)


###############################################################################
### Panel Regression with Binary Vars

# broker_bin_labels  <- paste(gsub(" ", "_", unique(data_bin$Broker)), 
#                            "_ind", sep = "")
# broker_bin_formula <- paste(broker_bin_labels, collapse = " + ")
# 
# mktctr_bin_labels  <- paste(unique(data_bin$MarketCenter), "_ind", sep = "")
# mktctr_bin_formula <- paste(mktctr_bin_labels, collapse = " + ")
# 
# # grabs specific col numbers, may change if data cols changed
# both_bin_labels  <- colnames(data_bin)[-1:-79]
# both_bin_formula <- paste(both_bin_labels, collapse = " + ")

# all binary bin labels
all_bin_labels  <- colnames(data_bin)[-0:-49]
all_bin_formula <- paste(all_bin_labels, collapse = " + ")

fit0_formula_bin = paste0(fit0_formula, ' + ', all_bin_formula)
fit1_formula_bin = paste0(fit1_formula, ' + ', all_bin_formula)
fit2_formula_bin = paste0(fit2_formula, ' + ', all_bin_formula)
fit3_formula_bin = paste0(fit3_formula, ' + ', all_bin_formula)
fit4_formula_bin = paste0(fit4_formula, ' + ', all_bin_formula)

fit0.bin = lm(as.formula(fit0_formula_bin), data = data_bin)
fit1.bin = lm(as.formula(fit1_formula_bin), data = data_bin)
fit2.bin = lm(as.formula(fit2_formula_bin), data = data_bin)
fit3.bin = lm(as.formula(fit3_formula_bin), data = data_bin)
fit4.bin = lm(as.formula(fit4_formula_bin), data = data_bin)

fits_bin = list(fit1.bin, fit2.bin, fit3.bin, fit4.bin)

SEs_bin = lapply(fits_bin, function(x) {(sqrt(diag(vcov(x))))})
s = createHTMLTable(fits_bin, title = paste(table_name, "(Random Effects)"), 
                    omit_labs = c(all_bin_labels),
                    dep_var_caption = 'Market Share', se_input = SEs_bin)

cat("<br><br>",
    file=paste0(folder, filename_output),
    append=TRUE)

cat(paste(s,"\n"),
    file=paste0(folder, filename_output),
    append=TRUE)


###############################################################################
### Demeaned

fit0.dmd = lm(as.formula(fit0_formula), data = data_dmd)
fit1.dmd = lm(as.formula(fit1_formula), data = data_dmd)
fit2.dmd = lm(as.formula(fit2_formula), data = data_dmd)
fit3.dmd = lm(as.formula(fit3_formula), data = data_dmd)
fit4.dmd = lm(as.formula(fit4_formula), data = data_dmd)

fits_dmd = list(fit1.dmd, fit2.dmd, fit3.dmd, fit4.dmd)

SEs_dmd = lapply(fits_dmd, function(x) {(sqrt(diag(vcovHC(x))))})
s = createHTMLTable(fits_dmd, paste(table_name, "(Fixed Effects)"), 
                    dep_var_caption = 'Market Share', se_input = SEs_dmd)

cat("<br><br>",
    file=paste0(folder, filename_output),
    append=TRUE)

cat(paste(s,"\n"),
    file=paste0(folder, filename_output),
    append=TRUE)

###############################################################################
### Hausman

for (i in c(1:length(fits))) {
  
  fit_a = fits_bin[[i]]
  fit_b = fits_dmd[[i]]
  print(paste0('p: ', hausman_test(fit_a, fit_b, length(fit_b$coefficients)-1)))
  
}

###############################################################################
### LaTeX

latex_file <- file(paste0(folder, 'Output/tables_termpaper.tex'))

# s = createHTMLTable(fits, "All Brokers (OLS)", dep_var_caption = 'Market Share',
#                     se = SEs, output_type = 'latex', nospace=TRUE)
# 
# cat(s, file = latex_file, sep = '\n')
# 
# s = createHTMLTable(fits_bin, title = 'All Brokers (Random Effects)',
#                     omit_labs = c(all_bin_labels),
#                     dep_var_caption = 'Market Share', se_input = SEs_bin,
#                     output_type = 'latex', nospace=TRUE)
# 
# cat(s, file = latex_file, append=TRUE, sep = '\n')

s = createHTMLTable(fits_dmd, 'All Brokers (Fixed Effects)',
                    dep_var_caption = 'Market Share', se_input = SEs_dmd,
                    output_type = 'latex', nospace=TRUE)

cat(s, file = latex_file, append=TRUE, sep = '\n')

###############################################################################
### Scratch




