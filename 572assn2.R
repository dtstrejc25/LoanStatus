---
  title: "IDS_572Assignment_2"
output: html_document
---
  #Libraries 
  library(C50)
library(caret)
library(Cubist)
library(lubridate)
library(gbm)
library(partykit)
library(ranger)
library(rpart)
library(rpart.plot)
library(tidyverse)
library(knitr)
library(modeldata)
library(ROCR)
library(magrittr)
library(e1071)
library (glmnet)
library(ROSE)
library('glmnet')

###importing the csv file
my_data = read.csv("~/Downloads/lcData5m.csv",header=TRUE)

###converting it into a dataframe 
df = as.data.frame(my_data)

###Changing the issue_d into a date now 

df$issue_d <- str_remove(df$issue_d, "T00:00:00Z")
summary(df$issue_d)
df$issue_d<-paste(df$issue_d, sep = "")
df$issue_d<-parse_date(df$issue_d,  format = "%Y-%m-%d")
str(df$issue_d)

###62 fails to parse bc they're blank

###put median in should be good now
str(df$last_pymnt_d)
df<- df %>% replace_na(list(last_pymnt_d=median(df$last_pymnt_d, na.rm=TRUE)))
df$last_pymnt_d<-paste(df$last_pymnt_d, "-01", sep = "")
df$last_pymnt_d<-parse_date(df$last_pymnt_d, format = "%b-%Y-%d")
str(df$last_pymnt_d)

### last_pymnt_d is date as well

#=============

df$annRet <- ((df$total_pymnt -df$funded_amnt)/df$funded_amnt)*(12/36)*100

#Actual Term 
#=============
###df$actualTerm <- ifelse(df$loan_status=="Fully Paid", as.duration(df$issue_d - df$last_pymnt_d))

df$actualTerm <- abs(df$issue_d- df$last_pymnt_d)

df$actualTerm <- ifelse(df$loan_status=="Fully Paid", abs(df$issue_d- df$last_pymnt_d),0)

###df$actualReturn <- ifelse(df$actualTerm>0, ((df$total_pymnt - df$funded_amnt)/df$funded_amnt)*(30*100/df$actualTerm),0)
###actual monthly percent return? is this what we want? *12 to get the predicted annual actual return pct?

###may be able to use the below code for us- account for 1% service charge

df$actualReturn <- ifelse(df$actualTerm>0, ((df$total_pymnt-df$funded_amnt))*(30/df$actualTerm)*0.99, 0)
###this is a number- *30 would be monthly return can do this or return on investment %


head(df$actualReturn)

#============
#Cleaning the data 
#============
###there one variable within the data that not categorize as charged ot fully paid and it was filtered out 
df <- df %>%  filter(loan_status !="Current")


df$loan_status <- factor(df$loan_status, levels=c("Fully Paid", "Charged Off"))

knitr::opts_chunk$set(echo = TRUE)

varsToRemove=c("issue_d", "application_type","earliest_cr_line","verification_status",
               "hardship_flag", "disbursement_method","term",
               "debt_settlement_flag","total_pymnt_inv","home_ownership",
               "num_actv_rev_tl","total_rec_prncp","last_pymnt_amnt",
               "emp_title","pymnt_plan","title","last_credit_pull_d",
               "initial_list_status","last_pymnt_d",
               "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", 
               "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl",
               "mths_since_recent_bc", "mths_since_recent_inq",
               "zip_code","addr_state","tax_liens",
               "pub_rec_bankruptcies","num_tl_120dpd_2m",
               "num_tl_30dpd","num_tl_90g_dpd_24m","num_tl_op_past_12m",
               "num_accts_ever_120_pd","num_actv_bc_tl",
               "delinq_amnt","chargeoff_within_12_mths","tot_coll_amt",
               "acc_now_delinq","policy_code",
               "collection_recovery_fee","recoveries","total_rec_late_fee",
               "out_prncp_inv","out_prncp","pub_rec","inq_last_6mths",
               "delinq_2yrs", "emp_length", "collections_12_mths_ex_med")
df <- df %>% select(-varsToRemove)

df <- df %>% select_if(function(x){!all(is.na(x))})


###missing value proportions in each column
colMeans(is.na(df))
###or, get only those columns where there are missing values
colMeans(is.na(df))[colMeans(is.na(df))>0]


nm<-names(df)[colMeans(is.na(df))>0.5]
df <- df %>% select(-nm)


###Impute missing values - first get the columns with missing values
colMeans(is.na(df))[colMeans(is.na(df))>0]
###summary of data in these columns
nm<- names(df)[colSums(is.na(df))>0]
summary(df[, nm])


###Similarly for the other variables
###If we are sure this is working and what we want, can replace the missing values on the lcdf dataset
df<- df %>% replace_na(list(mths_since_last_delinq=500, revol_util=median(df$revol_util, na.rm=TRUE)))


df<- df %>% replace_na(list(bc_open_to_buy=median(df$bc_open_to_buy, na.rm=TRUE)))

df<- df %>% replace_na(list(percent_bc_gt_75=median(df$percent_bc_gt_75, na.rm=TRUE)))

###reshaping the variable
df$purpose <- fct_recode(df$purpose, other="wedding", other="educational", other="renewable_energy")


###Note - character variables can cause a problem with some model packages, so better to convert all of these to factors
df= df %>% mutate_if(is.character, as.factor)


df <- na.omit(df)


#==============
#Spliting the data 70/30 test /training 
TRG_PCT=0.7
nr=nrow(df)
trnIndex = sample(1:nr, size = round(TRG_PCT*nr), replace=FALSE)

dfTrn=df[trnIndex,]   #training data with the randomly selected row-indices
dfTst = df[-trnIndex,]

summary(dfTrn)
dfTrn$loan_status <-as.factor(dfTrn$loan_status)

###if we want to remove EVERYTHING that has a strong correlation
###loam amount, funded amount, funded amount inv, int rate, total payment, num tv bal gt 0
###num sats, open acc, installment, revol bal, num op rev tl, num rev accts, total bc limit
###bc open to buy, bc_util, rev balance, percent gt 75

data=subset(df, select=-c(loan_amnt, funded_amnt, funded_amnt_inv, int_rate, total_pymnt, num_rev_tl_bal_gt_0,num_sats, open_acc, installment,revol_bal,
                          num_op_rev_tl, num_rev_accts, total_bc_limit, bc_open_to_buy, bc_util, revol_bal, percent_bc_gt_75, total_rec_int))


TRG_PCT=0.7
nr=nrow(data)
trnIndex = sample(1:nr, size = round(TRG_PCT*nr), replace=FALSE)

subsetTrn=data[trnIndex,]   #training data with the randomly selected row-indices
subsetTst= data[-trnIndex,]


prop.table(table(dfTrn$loan_status))
summary(dfTrn$loan_status)


#50/50 split on maybe just a few models-m1, m2, 1 more
TRG_PCT=0.5
nr=nrow(df)
trnIndex = sample(1:nr, size = round(TRG_PCT*nr), replace=FALSE)

halfdfTrn=data[trnIndex,]   #training data with the randomly selected row-indices
halfdfTst= data[-trnIndex,]


###preprocess the dfTrn, center seemed to be the least "bad"
pre_proc_val <- preProcess(df, method = c("center"))

###make the test and trains sets
dfTrnPP = predict(pre_proc_val, dfTrn)
dfTestPP = predict(pre_proc_val, dfTst)


###preprocess data subset
dpre_proc_val <- preProcess(data, method = c("BoxCox"))

ddTrnPP = predict(dpre_proc_val, subsetTrn)
ddTestPP = predict(dpre_proc_val, subsetTst)


#============== 
#Question 1 
##Part(B)
###List of what we plan to do 
####assume normal distrib- OLS
####what to do about highly correlated data, regularization-like lasso/ridge
####cut off at .8 to remove most highly correlated x vars. corrl matrix- then do confusion
####over test set. lasso automatically gets rid of the corrl ones. then compare methods 


dfTrn <- dfTrn[c(1,2,3,4,5,10,14:43,6,7,8,9,11,12,13)]

library(corrplot)  #had to install it



###Correlation Matrix
####Regular correlation, but we ignore na values

dt.cor = cor(dfTrn[,1:33], use="pairwise.complete.obs")
corrplot(dt.cor)


####without the highly correlated data- see new vars step up 
m1 <- glm(loan_status ~ ., data= subset(subsetTrn, select = -c(annRet, actualTerm, actualReturn)), family = binomial)
summary(m1)

pred1<-predict(m1, newdata = dfTst)
accuracy.meas(dfTst$loan_status, pred1)  #bad--so let's try to balance the loan status

pred.t <- predict(m1, newdata = dfTst)
roc.curve(dfTst$loan_status, pred.t)   #AUC is .734- not as good as the model with BOTH

testpred<-predict(m1,dfTst,type="response")
etabtest <- table(testpred, dfTst[,1])
m1testerr<-sum(diag(etabtest))/sum(etabtest)
m1testerr  #.00003277


#balance with both over and under sampling
dfBOTH <- ovun.sample(loan_status ~ ., data = dfTrn, method = "both")$data
table(dfBOTH$loan_status)

#try ROSE too
df.rose <- ROSE(loan_status ~ ., data = dfTrn, seed = 1)$data
table(df$loan_status)


####compare rose and both with m1

m1ROSE <- glm(loan_status ~ ., data= subset(df.rose, select = -c(annRet, actualTerm, actualReturn, total_pymnt)), family = binomial)
summary(m1ROSE)

m1BOTH <- glm(loan_status ~ ., data= subset(dfBOTH, select = -c(annRet, actualTerm, actualReturn, total_pymnt)), family = binomial)
summary(m1BOTH)
#AIC = 54171

pred.tree.rose <- predict(m1ROSE, newdata = dfTst)
pred.tree.both <- predict(m1BOTH, newdata = dfTst)

####how our new- balanced- data performs on test data
roc.curve(dfTst$loan_status, pred.tree.rose)   #AUC is .711
roc.curve(dfTst$loan_status, pred.tree.both)   #AUC is .739

#ROSE further evaluation
pred1 <- predict(m1ROSE, newdata = dfTst)
accuracy.meas(dfTst$loan_status, pred1)
#gives precision, recall, and F statistics

testpred<-predict(m1ROSE,dfTst,type="response")
etabtest <- table(testpred, dfTst[,1])
m1testerr<-sum(diag(etabtest))/sum(etabtest)
m1testerr  

#BOTH further evaluation
pred1 <- predict(m1BOTH, newdata = dfTst)
accuracy.meas(dfTst$loan_status, pred1)

testpred<-predict(m1BOTH,dfTst,type="response")
etabtest <- table(testpred, dfTst[,1])
m1testerr<-sum(diag(etabtest))/sum(etabtest)
m1testerr  

#now we do our preprocess model
m1PP <- glm(loan_status ~ ., data= subset(dfTrnPP, select = -c(annRet, actualTerm, actualReturn, total_pymnt)), family = binomial)
summary(m1PP)

predm1PP <- predict(m1PP, newdata = dfTst)
roc.curve(dfTst$loan_status, predm1PP)
#AUC .737

testpred<-predict(m1PP,dfTst,type="response")
etabtest <- table(testpred, dfTst[,1])
m1testerr<-sum(diag(etabtest))/sum(etabtest)
m1testerr  #.00003267

pred1 <- predict(m1PP, newdata = dfTst)
accuracy.meas(dfTst$loan_status, pred1)

######SUBSET models===============================
m2<-glm(loan_status ~ ., data= subset(subsetTrn, select = -c(annRet, actualTerm, actualReturn)), family = binomial)
summary(m2)

predm2 <- predict(m2, newdata = dfTst)
roc.curve(dfTst$loan_status, predm2)
#AUC 694

pred1 <- predict(m2, newdata = dfTst)
accuracy.meas(dfTst$loan_status, pred1)

testpred<-predict(m2,dfTst,type="response")
etabtest <- table(testpred, dfTst[,1])
m1testerr<-sum(diag(etabtest))/sum(etabtest)
m1testerr 

#now the PP model for our subset
m2PP<-glm(loan_status ~ ., data= subset(ddTrnPP, select = -c(annRet, actualTerm, actualReturn)), family = binomial)
summary(m2PP)
#AIC-56301- lower- when i tried scale and center the AIC values were the same

predm2PP <- predict(m2PP, newdata = dfTst)
roc.curve(dfTst$loan_status, predm2PP)
#AUC .573

pred1 <- predict(m2PP, newdata = dfTst)
accuracy.meas(dfTst$loan_status, pred1)

testpred<-predict(m2PP,dfTst,type="response")
etabtest <- table(testpred, dfTst[,1])
m1testerr<-sum(diag(etabtest))/sum(etabtest)
m1testerr  #.003866


#############using our split dataset###############
########################################################################

#=========================do the 50/50 split on maybe just a few models-m1, m1BOTH, m1PP
TRG_PCT=0.5
nr=nrow(df)
trnIndex = sample(1:nr, size = round(TRG_PCT*nr), replace=FALSE)

halfdfTrn=data[trnIndex,]   #training data with the randomly selected row-indices
halfdfTst= data[-trnIndex,]


m1h <- glm(loan_status ~ ., data= subset(halfdfTrn, select = -c(annRet, actualTerm, actualReturn)), family = binomial)
summary(m1h)

#AIC 40134

pred.t <- predict(m1h, newdata = halfdfTst)
roc.curve(halfdfTst$loan_status, pred.t)   #

testpred<-predict(m1h,halfdfTst,type="response")
etabtest <- table(testpred, halfdfTst[,1])
m1testerr<-sum(diag(etabtest))/sum(etabtest)
m1testerr  #0

pred1 <- predict(m1h, newdata = halfdfTst)
accuracy.meas(halfdfTst$loan_status, pred1)


#=========================both sampling half===================

library(ROSE)
dfBOTHtrnh <- ovun.sample(loan_status ~ ., data = halfdfTrn, method = "both")$data
dfBOTHtsth <- ovun.sample(loan_status ~ ., data = halfdfTst, method = "both")$data

m1BOTHh <- glm(loan_status ~ ., data= subset(dfBOTHtrnh, select = -c(annRet, actualTerm, actualReturn)), family = binomial)
summary(m1BOTHh)  #AIC 64441

pred.t <- predict(m1BOTHh, newdata = dfBOTHtsth)
roc.curve(dfBOTHtsth$loan_status, pred.t)   #

testpred<-predict(m1BOTHh,dfBOTHtsth,type="response")
etabtest <- table(testpred, dfBOTHtsth[,1])
m1testerr<-sum(diag(etabtest))/sum(etabtest)
m1testerr  #0

pred1 <- predict(m1BOTHh, newdata = dfBOTHtsth)
accuracy.meas(dfBOTHtsth$loan_status, pred1)

halfdfTrn$loan_status <- ifelse(halfdfTrn$loan_status=="Fully Paid",1,0)
halfdfTst$loan_status <- ifelse(halfdfTst$loan_status=="Fully Paid",1,0)


#PreProcess the 50/50 set##########################################################################

#make the test and trains sets

dfPPtrnh <- predict(pre_proc_val, halfdfTrn)
dfPPtsth <- predict(pre_proc_valT, halfdfTst)


m1PPh <- glm(loan_status ~ ., data= subset(dfPPtrnh, select = -c(annRet, actualTerm, actualReturn)), family = binomial)
summary(m1PPh)   #64613

pred.t <- predict(m1PPh, newdata = dfPPtsth)
roc.curve(dfPPtsth$loan_status, pred.t)   

testpred<-predict(m1PPh,dfPPtsth,type="response")
etabtest <- table(testpred, dfPPtsth[,1])
m1testerr<-sum(diag(etabtest))/sum(etabtest)
m1testerr  #

pred1 <- predict(m1PPh, newdata = dfPPtsth)
accuracy.meas(dfPPtsth$loan_status, pred1)

########now try the subset and a 50/50 split #######################

TRG_PCT=0.5
nr=nrow(data)
trnIndex = sample(1:nr, size = round(TRG_PCT*nr), replace=FALSE)

subhalfdfTrn=data[trnIndex,]   #training data with the randomly selected row-indices
subhalfdfTst= data[-trnIndex,]

subhalfdfTrn$loan_status <- ifelse(subhalfdfTrn$loan_status=="Fully Paid",1,0)
subhalfdfTst$loan_status <- ifelse(subhalfdfTst$loan_status=="Fully Paid",1,0)

ssm1h <- glm(loan_status ~ ., data= subset(subhalfdfTrn, select = -c(annRet, actualTerm, actualReturn)), family = binomial)
summary(ssm1h)

#AIC 39762- best so far
str(subhalfdfTst$loan_status)

pred.t <- predict(ssm1h, newdata = subhalfdfTst)
roc.curve(subhalfdfTst$loan_status, pred.t)   #.688

testpred<-predict(ssm1h,subhalfdfTst,type="response")
etabtest <- table(testpred, subhalfdfTst[,1])
m1testerr<-sum(diag(etabtest))/sum(etabtest)
m1testerr  #.0000197

pred1 <- predict(ssm1h, newdata = dfTst)
accuracy.meas(dfTst$loan_status, pred1)





################################# GLMNET FUNCTIONS #########################################
#setting up matrices to use for glmnet
#make a regular one, subset one, and then also test for evaluation
xD<-dfTrn %>% select(-loan_status, -actualTerm, -annRet, -actualReturn, -total_pymnt)
yD<- dfTrn$loan_status

xDsub <- subsetTrn %>% select(-loan_status, -actualTerm, -annRet, -actualReturn)
yDsub <- subsetTrn$loan_status

xDsubT <- subsetTst %>% select(-loan_status, -actualTerm, -annRet, -actualReturn)
yDsubT <- subsetTst$loan_status

xDTst<-dfTst %>% select(-loan_status, -actualTerm, -annRet, -actualReturn, -total_pymnt)
yDTst<- dfTst$loan_status

dfTrn$loan_status <- ifelse(dfTrn$loan_status=="Fully Paid",1,0)
dfTst$loan_status <- ifelse(dfTst$loan_status=="Fully Paid",1,0)

subsetTrn$loan_status <- ifelse(subsetTrn$loan_status=="Fully Paid",1,0)
subsetTst$loan_status <- ifelse(subsetTst$loan_status=="Fully Paid",1,0)




#for nfolds =5
set.seed(1234)
glmDefault_cvL<- cv.glmnet(data.matrix(xD), dfTrn$loan_status, family="binomial", nfolds=5, alpha=1, type.measure = "auc")
glmDefault_cvR<- cv.glmnet(data.matrix(xD), dfTrn$loan_status, family="binomial", nfolds=5, alpha=0, type.measure = "auc")
glmDefault_cvE<- cv.glmnet(data.matrix(xD), dfTrn$loan_status, family="binomial", nfolds=5, alpha=0.5, type.measure = "auc")

#let's do ridge with the subset-since it already removed corrl vars
glmDefault_cvS <- cv.glmnet(data.matrix(xDsub), subsetTrn$loan_status, family="binomial", nfolds=5, alpha=0, type.measure = "auc")


mean<-glmDefault_cvL$cvm
mean(mean) #
mean<-glmDefault_cvR$cvm
mean(mean) #
mean<-glmDefault_cvE$cvm
mean(mean) #
mean<-glmDefault_cvS$cvm
mean(mean) #

#get AUC and test accuracy
dfTrn$loan_status <- ifelse(dfTrn$loan_status=="Fully Paid",1,0)
dfTst$loan_status <- ifelse(dfTst$loan_status=="Fully Paid",1,0)

subsetTrn$loan_status <- ifelse(subsetTrn$loan_status=="Fully Paid",1,0)
subsetTst$loan_status <- ifelse(subsetTst$loan_status=="Fully Paid",1,0)
subsetTst$loan_status

a<-assess.glmnet(glmDefault_cvE, data.matrix(xDTst), newy = dfTst$loan_status)
a
a<-assess.glmnet(glmDefault_cvR, data.matrix(xDTst), newy = dfTst$loan_status)
a

a<-assess.glmnet(glmDefault_cvL, data.matrix(xDTst), newy = dfTst$loan_status)
a

a<-assess.glmnet(glmDefault_cvS, data.matrix(xDsubT), newy = subsetTst$loan_status)
a

par(mfrow=c(2,2))
plot(glmDefault_cvL)
plot(glmDefault_cvR)
plot(glmDefault_cvE)
plot(glmDefault_cvS)


confusion.glmnet(glmDefault_cvL, data.matrix(xDTst), newy = dfTst$loan_status)  #
confusion.glmnet(glmDefault_cvR, data.matrix(xDTst), newy = dfTst$loan_status)  #
confusion.glmnet(glmDefault_cvE, data.matrix(xDTst), newy = dfTst$loan_status)  #

confusion.glmnet(glmDefault_cvS, data.matrix(xDsubT), newy = subsetTst$loan_status)  #

# with ridge--determine the best lambdas------------then plug those in glmnet
glmDefault_cvR$lambda.1se
#.01143

enet2 <- glmnet(data.matrix(xD), dfTrn$loan_status, alpha = 0.5,family = "binomial", lambda = .01143)
lassoreg2<-glmnet(data.matrix(xD), dfTrn$loan_status, family = "binomial", alpha = 1, lambda = .01143)

(enet2) #deviance
lassoreg2  

assess.glmnet(lassoreg2, data.matrix(xDTst), newy = dfTst$loan_status)   #use test sets for test error
assess.glmnet(enet2, data.matrix(xDTst), newy = dfTst$loan_status)  #

confusion.glmnet(enet2, data.matrix(xDTst), newy = dfTst$loan_status) # percent correct
confusion.glmnet(lassoreg2, data.matrix(xDTst), newy = dfTst$loan_status) # percent correct

#see how these models do

#==========auc 5 -------------with dftrnPP and ddtrnpp=======================================
pre_proc_val <- preProcess(df, method = c("center"))

dfTrnPP = predict(pre_proc_val, dfTrn)
dfTestPP = predict(pre_proc_val, dfTst)


xD<-dfTrnPP %>% select(-actualReturn, -annRet, -actualTerm, -total_pymnt, -loan_status)

xDTst<-dfTestPP %>% select(-actualReturn, -annRet, -actualTerm, -total_pymnt, -loan_status)

#PP for the subset
dpre_proc_val <-preProcess(data, method = c("BoxCox"))

ddTrnPP = predict(dpre_proc_val, subsetTrn)
ddTestPP = predict(dpre_proc_val, subsetTst)

xD1<-ddTrnPP %>% select(-loan_status, -actualTerm, -annRet, -actualReturn)
xD1Tst<-ddTestPP %>% select(-actualReturn, -annRet, -actualTerm, -loan_status)


glmDefault_cvmL<- cv.glmnet(data.matrix(xD), dfTrnPP$loan_status, family="binomial", nfolds=5, alpha=1, type.measure = "auc")

glmDefault_cvR<- cv.glmnet(data.matrix(xD), dfTrnPP$loan_status, family="binomial", nfolds=5, alpha=0, type.measure = "auc")
glmDefault_cvE<- cv.glmnet(data.matrix(xD), dfTrnPP$loan_status, family="binomial", nfolds=5, alpha=0.5, type.measure = "auc")

#let's do ridge with the subset-since it already removed corrl vars
glmDefault_cvS <- cv.glmnet(data.matrix(xD1), ddTrnPP$loan_status, family="binomial", nfolds=5, alpha=0, type.measure = "auc")


mean<-glmDefault_cvmL$cvm
mean(mean) #
mean<-glmDefault_cvR$cvm
mean(mean) #
mean<-glmDefault_cvE$cvm
mean(mean) #
mean<-glmDefault_cvS$cvm
mean(mean) #

#get AUC
a<-assess.glmnet(glmDefault_cvE, data.matrix(xDTst), newy = dfTestPP$loan_status)
a

a<-assess.glmnet(glmDefault_cvR, data.matrix(xDTst), newy = dfTestPP$loan_status)
a$auc

a<-assess.glmnet(glmDefault_cvmL, data.matrix(xDTst), newy = dfTestPP$loan_status)
a$auc

a<-assess.glmnet(glmDefault_cvS, data.matrix(xD1Tst), newy = ddTestPP$loan_status)
a$auc

par(mfrow=c(2,2))
plot(glmDefault_cvmL)
plot(glmDefault_cvR)
plot(glmDefault_cvE)
plot(glmDefault_cvS)

dfTrnPP$loan_status <- ifelse(dfTrnPP$loan_status=="Fully Paid",1,0)
dfTestPP$loan_status <- ifelse(dfTestPP$loan_status=="Fully Paid",1,0)

confusion.glmnet(glmDefault_cvmL, data.matrix(xDTst), newy = dfTestPP$loan_status)  #
confusion.glmnet(glmDefault_cvR, data.matrix(xDTst), newy = dfTestPP$loan_status)  #
confusion.glmnet(glmDefault_cvE, data.matrix(xDTst), newy = dfTestPP$loan_status)  #

confusion.glmnet(glmDefault_cvS, data.matrix(xD1Tst), newy = ddTestPP$loan_status)  #

#######################try lasso and enet with the ridge model lambda min
glmDefault_cvR$lambda.1se
#.0166

enet2 <- glmnet(data.matrix(xD), dfTrn$loan_status, alpha = 0.5,family = "binomial", lambda = .0166)
lassoreg2<-glmnet(data.matrix(xD), dfTrn$loan_status, family = "binomial", alpha = 1, lambda = .0166)

(enet2) #deviance- 
lassoreg2  #

assess.glmnet(lassoreg2, data.matrix(xDTst), newy = dfTst$loan_status)   #use test sets for test error
assess.glmnet(enet2, data.matrix(xDTst), newy = dfTst$loan_status)  #

confusion.glmnet(enet2, data.matrix(xDTst), newy = dfTst$loan_status) # percent correct
confusion.glmnet(lassoreg2, data.matrix(xDTst), newy = dfTst$loan_status) # percent correct


#try the subset model we got here, PP- and eval the lambdas
glmDefault_cvS$lambda.min
#0.0079

enet2 <- glmnet(data.matrix(xD), dfTrn$loan_status, alpha = 0.5,family = "binomial", lambda = 0.0079)
lassoreg2<-glmnet(data.matrix(xD), dfTrn$loan_status, family = "binomial", alpha = 1, lambda = 0.0079)

(enet2) #deviance- 
lassoreg2  #.

assess.glmnet(lassoreg2, data.matrix(xDTst), newy = dfTst$loan_status)   #use test sets for test error
assess.glmnet(enet2, data.matrix(xDTst), newy = dfTst$loan_status)  #

confusion.glmnet(enet2, data.matrix(xDTst), newy = dfTst$loan_status) # percent correct
confusion.glmnet(lassoreg2, data.matrix(xDTst), newy = dfTst$loan_status) # percent correct



#=======================MSE with 5 folds=======================
########################## with PP data only  ###############
pre_proc_val <- preProcess(df, method = c("center"))

dfTrnPP = predict(pre_proc_val, dfTrn)
dfTestPP = predict(pre_proc_val, dfTst)


xD<-dfTrnPP %>% select(-actualReturn, -annRet, -actualTerm, -total_pymnt, -loan_status)

xDTst<-dfTestPP %>% select(-actualReturn, -annRet, -actualTerm, -total_pymnt, -loan_status)

#PP for the subset-try center here
dpre_proc_val <-preProcess(data, method = c("center"))

ddTrnPP = predict(dpre_proc_val, subsetTrn)
ddTestPP = predict(dpre_proc_val, subsetTst)

xD1<-ddTrnPP %>% select(-loan_status, -actualTerm, -annRet, -actualReturn)
xD1Tst<-ddTestPP %>% select(-actualReturn, -annRet, -actualTerm, -loan_status)


glmDefault_cvmL<- cv.glmnet(data.matrix(xD), dfTrnPP$loan_status, family="binomial", nfolds=5, alpha=1, type.measure = "mse")

glmDefault_cvR<- cv.glmnet(data.matrix(xD), dfTrnPP$loan_status, family="binomial", nfolds=5, alpha=0, type.measure = "mse")
glmDefault_cvE<- cv.glmnet(data.matrix(xD), dfTrnPP$loan_status, family="binomial", nfolds=5, alpha=0.5, type.measure = "mse")

#let's do ridge with the subset-since it already removed corrl vars
glmDefault_cvS <- cv.glmnet(data.matrix(xD1), ddTrnPP$loan_status, family="binomial", nfolds=5, alpha=0, type.measure = "mse")

mean<-glmDefault_cvmL$cvm
mean(mean) #
mean<-glmDefault_cvR$cvm
mean(mean) #
mean<-glmDefault_cvE$cvm
mean(mean) #
mean<-glmDefault_cvS$cvm
mean(mean) #

#get AUC 
a<-assess.glmnet(glmDefault_cvE, data.matrix(xDTst), newy = dfTestPP$loan_status)
a$mse
a$auc

a<-assess.glmnet(glmDefault_cvR, data.matrix(xDTst), newy = dfTestPP$loan_status)
a$mse
a$auc

a<-assess.glmnet(glmDefault_cvmL, data.matrix(xDTst), newy = dfTestPP$loan_status)
a$mse
a$auc

a<-assess.glmnet(glmDefault_cvS, data.matrix(xD1Tst), newy = ddTestPP$loan_status)
a$mse
a$auc

par(mfrow=c(2,2))
plot(glmDefault_cvmL)
plot(glmDefault_cvR)
plot(glmDefault_cvE)
plot(glmDefault_cvS)

dfTrnPP$loan_status <- ifelse(dfTrnPP$loan_status=="Fully Paid",1,0)
dfTestPP$loan_status <- ifelse(dfTestPP$loan_status=="Fully Paid",1,0)

confusion.glmnet(glmDefault_cvmL, data.matrix(xDTst), newy = dfTestPP$loan_status)  #
confusion.glmnet(glmDefault_cvR, data.matrix(xDTst), newy = dfTestPP$loan_status)  #
confusion.glmnet(glmDefault_cvE, data.matrix(xDTst), newy = dfTestPP$loan_status)  #

confusion.glmnet(glmDefault_cvS, data.matrix(xD1Tst), newy = ddTestPP$loan_status)  #

################################################################

#############################now use the both AND df PP to see how that works=============

dfBOTHtrn <- ovun.sample(loan_status ~ ., data = dfTrn, method = "both")$data
dfBOTHtst <- ovun.sample(loan_status ~ ., data = dfTst, method = "both")$data

pre_proc_val <- preProcess(dfBOTHtrn, method = c("center"))
pre_proc_val2 <- preProcess(dfBOTHtst, method = c("center"))

dfbothTrnPP = predict(pre_proc_val, dfBOTHtrn)
dfbothTestPP = predict(pre_proc_val2, dfBOTHtst)

#subset
ddBOTHtrn <- ovun.sample(loan_status ~ ., data = subsetTrn, method = "both")$data
ddBOTHtst <- ovun.sample(loan_status ~ ., data = subsetTst, method = "both")$data

pre_proc_val <- preProcess(ddBOTHtrn, method = c("center"))
pre_proc_val2 <- preProcess(ddBOTHtst, method = c("center"))

ddBothTrnPP = predict(pre_proc_val, ddBOTHtrn)
ddBothTestPP = predict(pre_proc_val2, ddBOTHtst)


xD<-dfbothTrnPP %>% select(-actualReturn, -annRet, -actualTerm, -total_pymnt, -loan_status)
xDTst<-dfbothTestPP %>% select(-actualReturn, -annRet, -actualTerm, -total_pymnt, -loan_status)

#PP for the subset

xD1<-ddBothTrnPP %>% select(-loan_status, -actualTerm, -annRet, -actualReturn)
xD1Tst<-ddBothTestPP %>% select(-actualReturn, -annRet, -actualTerm, -loan_status)


glmDefault_cvmL<- cv.glmnet(data.matrix(xD), dfbothTrnPP$loan_status, family="binomial", nfolds=5, alpha=1, type.measure = "auc")

glmDefault_cvR<- cv.glmnet(data.matrix(xD), dfbothTrnPP$loan_status, family="binomial", nfolds=5, alpha=0, type.measure = "auc")
glmDefault_cvE<- cv.glmnet(data.matrix(xD), dfbothTrnPP$loan_status, family="binomial", nfolds=5, alpha=0.5, type.measure = "auc")

#let's do ridge with the subset-since it already removed corrl vars
glmDefault_cvS <- cv.glmnet(data.matrix(xD1), ddBothTrnPP$loan_status, family="binomial", nfolds=5, alpha=0, type.measure = "auc")

mean<-glmDefault_cvmL$cvm
mean(mean) #
mean<-glmDefault_cvR$cvm
mean(mean) #
mean<-glmDefault_cvE$cvm
mean(mean) #
mean<-glmDefault_cvS$cvm
mean(mean) #

#get AUC and MAE
a<-assess.glmnet(glmDefault_cvE, data.matrix(xDTst), newy = dfbothTestPP$loan_status)
a

a<-assess.glmnet(glmDefault_cvR, data.matrix(xDTst), newy = dfbothTestPP$loan_status)
a

a<-assess.glmnet(glmDefault_cvmL, data.matrix(xDTst), newy = dfbothTestPP$loan_status)
a

a<-assess.glmnet(glmDefault_cvS, data.matrix(xD1Tst), newy = ddBothTestPP$loan_status)
a

par(mfrow=c(2,2))
plot(glmDefault_cvmL)
plot(glmDefault_cvR)
plot(glmDefault_cvE)
plot(glmDefault_cvS)


confusion.glmnet(glmDefault_cvmL, data.matrix(xDTst), newy = dfbothTestPP$loan_status)  #
confusion.glmnet(glmDefault_cvR, data.matrix(xDTst), newy = dfbothTestPP$loan_status)  #
confusion.glmnet(glmDefault_cvE, data.matrix(xDTst), newy = dfbothTestPP$loan_status)  #

confusion.glmnet(glmDefault_cvS, data.matrix(xD1Tst), newy = ddBothTestPP$loan_status)  #

###########################################################################

#auc 3-------------with dftrnPP and ddtrnpp======================preprocess=================
pre_proc_val <- preProcess(df, method = c("center"))

dfTrnPP = predict(pre_proc_val, dfTrn)
dfTestPP = predict(pre_proc_val, dfTst)


xD<-dfTrnPP %>% select(-actualReturn, -annRet, -actualTerm, -total_pymnt, -loan_status)

xDTst<-dfTestPP %>% select(-actualReturn, -annRet, -actualTerm, -total_pymnt, -loan_status)

#PP for the subset
dpre_proc_val <-preProcess(data, method = c("BoxCox"))

ddTrnPP = predict(dpre_proc_val, subsetTrn)
ddTestPP = predict(dpre_proc_val, subsetTst)

xD1<-ddTrnPP %>% select(-loan_status, -actualTerm, -annRet, -actualReturn)
xD1Tst<-ddTestPP %>% select(-actualReturn, -annRet, -actualTerm, -loan_status)


glmDefault_cvmL<- cv.glmnet(data.matrix(xD), dfTrnPP$loan_status, family="binomial", nfolds=3, alpha=1, type.measure = "auc")

glmDefault_cvR<- cv.glmnet(data.matrix(xD), dfTrnPP$loan_status, family="binomial", nfolds=3, alpha=0, type.measure = "auc")
glmDefault_cvE<- cv.glmnet(data.matrix(xD), dfTrnPP$loan_status, family="binomial", nfolds=3, alpha=0.5, type.measure = "auc")

#let's do ridge with the subset-since it already removed corrl vars
glmDefault_cvS <- cv.glmnet(data.matrix(xD1), ddTrnPP$loan_status, family="binomial", nfolds=3, alpha=0, type.measure = "auc")


mean<-glmDefault_cvmL$cvm
mean(mean) #
mean<-glmDefault_cvR$cvm
mean(mean) #
mean<-glmDefault_cvE$cvm
mean(mean) #
mean<-glmDefault_cvS$cvm
mean(mean) #

#get AUC
a<-assess.glmnet(glmDefault_cvE, data.matrix(xDTst), newy = dfTestPP$loan_status)
a

a<-assess.glmnet(glmDefault_cvR, data.matrix(xDTst), newy = dfTestPP$loan_status)
a$auc

a<-assess.glmnet(glmDefault_cvmL, data.matrix(xDTst), newy = dfTestPP$loan_status)
a$auc

a<-assess.glmnet(glmDefault_cvS, data.matrix(xD1Tst), newy = ddTestPP$loan_status)
a$auc

par(mfrow=c(2,2))
plot(glmDefault_cvmL)
plot(glmDefault_cvR)
plot(glmDefault_cvE)
plot(glmDefault_cvS)

dfTrnPP$loan_status <- ifelse(dfTrnPP$loan_status=="Fully Paid",1,0)
dfTestPP$loan_status <- ifelse(dfTestPP$loan_status=="Fully Paid",1,0)

confusion.glmnet(glmDefault_cvmL, data.matrix(xDTst), newy = dfTestPP$loan_status)  #
confusion.glmnet(glmDefault_cvR, data.matrix(xDTst), newy = dfTestPP$loan_status)  #
confusion.glmnet(glmDefault_cvE, data.matrix(xDTst), newy = dfTestPP$loan_status)  #

confusion.glmnet(glmDefault_cvS, data.matrix(xD1Tst), newy = ddTestPP$loan_status)  #



################################################################
#=============
#Random Forest Model

library(ranger)

rfRangerModel4 = ranger(loan_status ~.,data=subset(dfTrn,select=-c(annRet, actualTerm,actualReturn,total_pymnt)), num.trees = 200, importance='permutation', probability = TRUE)
rfRangerModel4

importance(rfRangerModel4)

### OOB prediction error (Brier s.):  0.1160559 

###test the performance
###calculate the MSE. "Fully Paid" is 1. "Charged Off" is 0. Any difference increases error by 1. It's 2.49% 
MSE <- sqrt(mean((ifelse(predict(rfRangerModel4, dfTrn)$predictions[,1]>predict(rfRangerModel4, dfTrn)$predictions[,2],1,0)
                  -ifelse(matrix(dfTrn$loan_status) == "Fully Paid",1,0))^2)) 

# Predict
PredfRangerModel4 <- predict(rfRangerModel4, dfTst)$predictions
PredfRangerModel4Bin <- ifelse(PredfRangerModel4[,1]>PredfRangerModel4[,2],1,0)
dfTst$loan_statusBin <- ifelse(matrix(dfTst$loan_status) == "Fully Paid",1,0)
dfTst$Predloan_status <- ifelse(PredfRangerModel4Bin == 1,"Pred. Fully Paid","Pred. Charged Off")

# Bar plot: Real vs. Predicted Loan Status
BP <- barplot(table(c(dfTst$Predloan_status,as.character(dfTst$loan_status))),
              main="Real vs. Predicted Loan Status",
              xlab="Category", ylab="Counts",col=c("darkblue","darkred","lightblue","red"),
              legend = rownames(table(c(dfTst$Predloan_status,as.character(dfTst$loan_status)))))
text(0.8,0.9*max(table(c(dfTst$Predloan_status,as.character(dfTst$loan_status)))), labels = paste0("MSE = ",sprintf("%4.2f", MSE)))

# Performance
Pred_RangerModel4 <- prediction(PredfRangerModel4[,2],dfTst$loan_statusBin, label.ordering=c(1,0))
Perf_RangerModel4 <-performance(Pred_RangerModel4, "tpr", "fpr")
aucRangerModel4 <- performance(Pred_RangerModel4, "auc") # aucRangerModel4@y.values # This is the AUC value

# AUC plot
plot(Perf_RangerModel4, avg= "threshold", lwd= 3, main= "ROC curve", lty=3, col="grey78") #colorize=TRUE
abline(a=0, b= 1)
text(0.2,0.7, labels = paste0("AUC = ",sprintf("%4.2f", aucRangerModel4@y.values)))

# Random Foorest for Loan Status
rfRangerModel4AR = ranger(actualReturn ~.,data=subset(dfTrn,select=-c(annRet, actualTerm, loan_status)), num.trees = 200, importance='permutation')
rfRangerModel4AR

importance(rfRangerModel4AR)
# R squared (OOB):                  0.8893754 

###test the performance

###calculate the MSE. "Fully Paid" is 1. "Charged Off" is 0. Any difference increases error by 1. It's 945.07% 
MSE_Trn <- sqrt(mean((predict(rfRangerModel4AR, dfTrn)$predictions - dfTrn$actualReturn)^2))
MSE_Tst <- sqrt(mean((predict(rfRangerModel4AR, dfTst)$predictions - dfTst$actualReturn)^2))

###plot train and test to see what the model is getting
par(mfrow =c(1,2))
plot((predict(rfRangerModel4AR, dfTrn))$predictions, dfTrn$actualReturn, xlab="Predicted Return", ylab="Actual Return",main="Training Data")
text(300,800, labels = paste0("MSE Training= ",sprintf("%4.2f", MSE_Trn)))
plot((predict(rfRangerModel4AR, dfTst))$predictions, dfTst$actualReturn, xlab="Predicted Return", ylab="Actual Return",main="Test Data")
text(200,700, labels = paste0("MSE Test= ",sprintf("%4.2f", MSE_Tst)))

# Performance by deciles on TRAINING DATA
predRet_Trn <- dfTrn %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet=(predict(rfRangerModel4AR, dfTrn))$predictions)
predRet_Trn <- predRet_Trn %>% mutate(tile=ntile(-predRet, 10))
predRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"), avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(actualTerm), totA=sum(grade=="A"), totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )

# Performance by deciles on TEST DATA
predRet_Tst <- dfTst %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet=(predict(rfRangerModel4AR, dfTst))$predictions)
predRet_Tst <- predRet_Tst %>% mutate(tile=ntile(-predRet, 10))
predRet_Tst %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"), avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(actualTerm), totA=sum(grade=="A"), totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )


#====================
#Part (A)
#Gmb boosting
dfTrn$loan_status <- ifelse(dfTrn$loan_status=="Fully Paid",1,0)
dfTst$loan_status <- ifelse(dfTst$loan_status=="Fully Paid",1,0)


###Experimenting with different parameters 
#Model 1 the data training set from assignment 1
gbm_M1 <- gbm(loan_status~., data=subset(dfTrn,select=-c(annRet, actualTerm,actualReturn,total_pymnt)), distribution = "bernoulli", n.trees=300, shrinkage=0.01, interaction.depth = 4,bag.fraction=0.5, cv.folds = 5, n.cores=16)  

#Model 2
gbm_M2 <- gbm(loan_status~., data=subset(dfTrn,select=-c(annRet, actualTerm,actualReturn,total_pymnt)), distribution = "adaboost", n.trees=300, shrinkage=0.01, interaction.depth = 4, bag.fraction=0.5, cv.folds = 5, n.cores=16) 


dfTrnPP$loan_status <- ifelse(dfTrnPP$loan_status=="Fully Paid",1,0)
dfTestPP$loan_status <- ifelse(dfTestPP$loan_status=="Fully Paid",1,0)

#Model 3 preprocessing 
gbm_M3 <- gbm(loan_status~., data=subset(dfTrnPP, select = -c(annRet, actualTerm, actualReturn, total_pymnt)), distribution = "bernoulli", n.trees=300, shrinkage=0.01, interaction.depth = 4, bag.fraction=0.5, cv.folds = 3, n.cores=16)  

#Model 4  
gbm_M4 <- gbm(loan_status~., data=subset(dfTrnPP, select = -c(annRet, actualTerm, actualReturn, total_pymnt)), distribution = "adaboost", n.trees=300, shrinkage=0.01, interaction.depth = 4, bag.fraction=0.5, cv.folds = 5, n.cores=16)  

subsetTrn$loan_status <- ifelse(subsetTrn$loan_status=="Fully Paid",1,0)
subsetTst$loan_status <- ifelse(subsetTst$loan_status=="Fully Paid",1,0)

#Model 5 taking out high correlated variables
gbm_M5 <- gbm(loan_status~., data= subset(subsetTrn, select = -c(annRet, actualTerm, actualReturn)), distribution = "bernoulli", n.trees=300, shrinkage=0.01, interaction.depth = 4, bag.fraction=0.5, cv.folds = 3, n.cores=16)  
#Model 6
gbm_M7 <- gbm(loan_status~., data= subset(subsetTrn, select = -c(annRet, actualTerm, actualReturn)), distribution = "adaboost", n.trees=300, shrinkage=0.01, interaction.depth = 4, bag.fraction=0.5, cv.folds = 5, n.cores=16) 

halfdfTrn$loan_status <- ifelse(halfdfTrn$loan_status=="Fully Paid",1,0)
halfdfTst$loan_status <- ifelse(halfdfTst$loan_status=="Fully Paid",1,0)

#Model 7 the data processed 50/50
gbm_M6 <- gbm(loan_status~., data= subset(halfdfTrn, select = -c(annRet, actualTerm, actualReturn)), distribution = "bernoulli", n.trees=300, shrinkage=0.01, interaction.depth = 4, bag.fraction=0.5, cv.folds = 3, n.cores=16)  

#Model 8 
gbm_M8 <- gbm(loan_status~., data= subset(halfdfTrn, select = -c(annRet, actualTerm, actualReturn)), distribution = "adaboost", n.trees=300, shrinkage=0.01, interaction.depth = 4, bag.fraction=0.5, cv.folds = 5, n.cores=16)

#Model 9 balancing the model
gbm_M9 <- gbm(loan_status~., data= subset(df.rose, select = -c(annRet, actualTerm, actualReturn, total_pymnt)), distribution = "adaboost", n.trees=300, shrinkage=0.01, interaction.depth = 4, bag.fraction=0.5, cv.folds = 5, n.cores=16)

#Model 10 
gbm_M10 <- gbm(loan_status~., data=subset(dfBOTH, select = -c(annRet, actualTerm, actualReturn, total_pymnt)), distribution = "adaboost", n.trees=300, shrinkage=0.01, interaction.depth = 4, bag.fraction=0.5, cv.folds = 5, n.cores=16)

###Summary
summary(gbm_M1)
summary(gbm_M2)
summary(gbm_M3)
summary(gbm_M4)
summary(gbm_M5)
summary(gbm_M7)
summary(gbm_M6)
summary(gbm_M8)
summary(gbm_M9)
summary(gbm_M10)

### CM Testing 

pred=predict(gbm_M1,dfTst, type="response") 
head(pred)
table(pred>0.5,dfTst$loan_status)

pred2=predict(gbm_M2,dfTst, type="response") 
head(pred2)
table(pred2>0.5,dfTst$loan_status)

pred4=predict(gbm_M3,dfTestPP) 
head(pred4)
table(pred4>0.5,dfTestPP$loan_status)

pred6=predict(gbm_M4,dfTestPP) 
head(pred6)
table(pred6>0.5,dfTestPP$loan_status)

pred8=predict(gbm_M5,subsetTst) 
head(pred8)
table(pred8>0.5,subsetTst$loan_status)

pred9=predict(gbm_M7,subsetTst) 
head(pred9)
table(pred9>0.5,subsetTst$loan_status)

pred12=predict(gbm_M6,halfdfTst) 
head(pred12)
table(pred12>0.5,halfdfTst$loan_status)

pred13=predict(gbm_M8,halfdfTst) 
head(pred13)
table(pred13>0.5,halfdfTst$loan_status)

pred15=predict(gbm_M9,df.rose) 
head(pred15)
table(pred15>0.5,df.rose$loan_status)

pred16=predict(gbm_M10,dfBOTH) 
head(pred16)
table(pred16>0.5,dfBOTH$loan_status)

###Performances
print(gbm_M1)
bestIter<-gbm.perf(gbm_M1, method='cv')
scores_gbmM1<- predict(gbm_M1, newdata=dfTst, n.tree= bestIter, type="response")
head(scores_gbmM1)

min(gbm_M1$cv.error)
mean(gbm_M1$cv.error)

print(gbm_M2)
bestIter<-gbm.perf(gbm_M2, method='cv')
scores_gbmM2<- predict(gbm_M2, newdata=dfTst, n.tree= bestIter, type="response")
head(scores_gbmM2)

min(gbm_M2$cv.error)
mean(gbm_M2$cv.error)

print(gbm_M3)
bestIter<-gbm.perf(gbm_M3, method='cv')
scores_gbmM3<- predict(gbm_M3, newdata=dfTst, n.tree= bestIter, type="response")
head(scores_gbmM3)

min(gbm_M3$cv.error)
mean(gbm_M3$cv.error)

print(gbm_M4)
bestIter<-gbm.perf(gbm_M4, method='cv')
scores_gbmM4<- predict(gbm_M4, newdata=dfTst, n.tree= bestIter, type="response")
head(scores_gbmM4)

min(gbm_M4$cv.error)
mean(gbm_M4$cv.error)

print(gbm_M5)
summary(gbm_M5, cbars=TRUE)
bestIter<-gbm.perf(gbm_M5, method='cv')
scores_gbmM5<- predict(gbm_M5, newdata=dfTst, n.tree= bestIter, type="response")
head(scores_gbmM5)

min(gbm_M5$cv.error)
mean(gbm_M5$cv.error)


print(gbm_M6)
bestIter<-gbm.perf(gbm_M6, method='cv')
scores_gbmM6<- predict(gbm_M6, newdata=dfTst, n.tree= bestIter, type="response")
head(scores_gbmM6)

min(gbm_M6$cv.error)
mean(gbm_M6$cv.error)

print(gbm_M7)
bestIter<-gbm.perf(gbm_M7, method='cv')
scores_gbmM7<- predict(gbm_M7, newdata=dfTst, n.tree= bestIter, type="response")
head(scores_gbmM7)

min(gbm_M7$cv.error)
mean(gbm_M7$cv.error)

print(gbm_M8)
bestIter<-gbm.perf(gbm_M8, method='cv')
scores_gbmM8<- predict(gbm_M8, newdata=dfTst, n.tree= bestIter, type="response")
head(scores_gbmM8)

min(gbm_M8$cv.error)
mean(gbm_M8$cv.error)

print(gbm_M9)
bestIter<-gbm.perf(gbm_M9, method='cv')
scores_gbmM9<- predict(gbm_M9, newdata=dfTst, n.tree= bestIter, type="response")
head(scores_gbmM9)

min(gbm_M9$cv.error)
mean(gbm_M9$cv.error)

print(gbm_M10)
bestIter<-gbm.perf(gbm_M10, method='cv')
scores_gbmM10<- predict(gbm_M10, newdata=dfTst, n.tree= bestIter, type="response")
head(scores_gbmM10)

min(gbm_M10$cv.error)
mean(gbm_M10$cv.error)

###ROC AUC 

pred_gbmM1=prediction(scores_gbmM1, dfTst$loan_status)
aucPerf_gbmM1 <-performance(pred_gbmM1, "tpr", "fpr")
aucPerf_gbm1=performance(pred_gbmM1, "auc")
aucPerf_gbmM1@y.values
plot(aucPerf_gbmM1,col=2,legacy.axes=TRUE,print.auc=TRUE,main="ROC(GBM)")
abline(a=0, b= 1)

pred_gbmM2=prediction(scores_gbmM2, dfTst$loan_status)
aucPerf_gbmM2 <-performance(pred_gbmM2, "tpr", "fpr")
aucPerf_gbm2=performance(pred_gbmM2, "auc")
aucPerf_gbmM2@y.values
plot(aucPerf_gbmM2,col=3,add=TRUE)

pred_gbmM3=prediction(scores_gbmM3, dfTst$loan_status)
aucPerf_gbmM3 <-performance(pred_gbmM3, "tpr", "fpr")
aucPerf_gbm3=performance(pred_gbmM3, "auc")
aucPerf_gbmM3@y.values
plot(aucPerf_gbmM3,col=4,add=TRUE)

pred_gbmM4=prediction(scores_gbmM4, dfTst$loan_status)
aucPerf_gbmM4 <-performance(pred_gbmM4, "tpr", "fpr")
aucPerf_gbm4=performance(pred_gbmM4, "auc")
aucPerf_gbmM4@y.values
plot(aucPerf_gbmM4,col=5,add=TRUE)

pred_gbmM5=prediction(scores_gbmM5, dfTst$loan_status)
aucPerf_gbmM5 <-performance(pred_gbmM5, "tpr", "fpr")
aucPerf_gbm5=performance(pred_gbmM5, "auc")
aucPerf_gbmM5@y.values
plot(aucPerf_gbmM5,col=6,add=TRUE)

pred_gbmM6=prediction(scores_gbmM6, dfTst$loan_status)
aucPerf_gbmM6 <-performance(pred_gbmM6, "tpr", "fpr")
aucPerf_gbm6=performance(pred_gbmM6, "auc")
aucPerf_gbmM6@y.values
plot(aucPerf_gbmM6,col='lavender',add=TRUE)

pred_gbmM7=prediction(scores_gbmM7, dfTst$loan_status)
aucPerf_gbmM7 <-performance(pred_gbmM7, "tpr", "fpr")
aucPerf_gbm7=performance(pred_gbmM7, "auc")
aucPerf_gbmM7@y.values
plot(aucPerf_gbmM7,col='magenta',add=TRUE)

pred_gbmM8=prediction(scores_gbmM8, dfTst$loan_status)
aucPerf_gbmM8 <-performance(pred_gbmM8, "tpr", "fpr")
aucPerf_gbm8=performance(pred_gbmM8, "auc")
aucPerf_gbmM8@y.values
plot(aucPerf_gbmM8,col='turquoise4',add=TRUE)

pred_gbmM9=prediction(scores_gbmM9, dfTst$loan_status)
aucPerf_gbmM9 <-performance(pred_gbmM9, "tpr", "fpr")
aucPerf_gbm9=performance(pred_gbmM9, "auc")
aucPerf_gbmM9@y.values
plot(aucPerf_gbmM9,col=67,add=TRUE)

pred_gbmM10=prediction(scores_gbmM10, dfTst$loan_status)
aucPerf_gbmM10 <-performance(pred_gbmM10, "tpr", "fpr")
aucPerf_gbm10=performance(pred_gbmM10, "auc")
aucPerf_gbmM10@y.values
plot(aucPerf_gbmM10,col=34,add=TRUE)

###Relative Influence Bar Graphs for the models
par(oma=c(1,5.0,1,1))
summary(gbm_M1,6,19,las=2,main="Model 1")
summary(gbm_M2,6,19,las=2,main="Model 2")
summary(gbm_M3,7,19,las=2,main="Model 3")
summary(gbm_M4,7,19,las=2,main="Model 4")
par(oma=c(1,6.0,1,1))
summary(gbm_M5,6,19,las=2,main="Model 5")
summary(gbm_M6,6,19,las=2,main="Model 7")
summary(gbm_M7,6,19,las=2,main="Model 6")
summary(gbm_M8,6,19,las=2,main="Model 8")
summary(gbm_M9,6,19,las=2,main="Model 9")
summary(gbm_M10,6,19,las=2,main="Model 10")
par(oma=c(1,1,1,1))

###Actual Return

predXgbRet_Trn <- subsetTrn %>% select(grade, loan_status, actualReturn, actualTerm) %>%
  mutate(predXgbRet=predict(gbm_M5,subset(subsetTrn, select = -c(annRet, actualTerm, actualReturn))))
predXgbRet_Trn <- predXgbRet_Trn %>% mutate(tile=ntile(-predXgbRet, 10))
predXgbRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgPredRet=mean(predXgbRet), numDefaults=sum(loan_status=="Charged Off"),
                                                avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(actualTerm), totA=sum(grade=="A"),
                                                totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F"))

predXgbRet_Trn <- ddTrnPP %>% select(grade, loan_status, actualReturn, actualTerm) %>%
  mutate(predXgbRet=predict(gbm_M8,subset(ddTrnPP, select = -c(annRet, actualTerm, actualReturn))))
predXgbRet_Trn <- predXgbRet_Trn %>% mutate(tile=ntile(-predXgbRet, 10))
predXgbRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgPredRet=mean(predXgbRet), numDefaults=sum(loan_status=="Charged Off"),
                                                avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(actualTerm), totA=sum(grade=="A"),
                                                totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F"))

predXgbRet_Trn <- subsetTst %>% select(grade, loan_status, actualReturn, actualTerm) %>%
  mutate(predXgbRet=predict(gbm_M5,subset(subsetTst, select = -c(annRet, actualTerm, actualReturn))))
predXgbRet_Trn <- predXgbRet_Trn %>% mutate(tile=ntile(-predXgbRet, 10))
predXgbRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgPredRet=mean(predXgbRet), numDefaults=sum(loan_status=="Charged Off"),
                                                avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(actualTerm), totA=sum(grade=="A"),
                                                totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F"))

predXgbRet_Trn <- ddTestPP %>% select(grade, loan_status, actualReturn, actualTerm) %>%
  mutate(predXgbRet=predict(gbm_M8,subset(ddTestPP, select = -c(annRet, actualTerm, actualReturn))))
predXgbRet_Trn <- predXgbRet_Trn %>% mutate(tile=ntile(-predXgbRet, 10))
predXgbRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgPredRet=mean(predXgbRet), numDefaults=sum(loan_status=="Charged Off"),
                                                avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(actualTerm), totA=sum(grade=="A"),
                                                totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F"))

#===============
#Question 2 
#===============

dpre_proc_val <-preProcess(data, method = c("BoxCox"))

ddTrnPP = predict(dpre_proc_val, subsetTrn)
ddTestPP = predict(dpre_proc_val, subsetTst)


library('glmnet')

#use what we saw in the correlation to remove certain vars
#we preprocessed early with "center" so this is the subset with PP
xD1<-ddTrnPP %>% select(-loan_status, -actualTerm, -annRet, -actualReturn)
yD1<- ddTrnPP$actualReturn

xD1Tst<-ddTestPP %>% select(-actualReturn, -annRet, -actualTerm, -loan_status)
yD1Tst <- ddTestPP$actualReturn

#cross validation for alpha=1 
cv_m<-cv.glmnet(data.matrix(xD1), yD1, family="gaussian", alpha = 1)

a<-assess.glmnet(cv_m, data.matrix(xD1Tst), newy = yD1Tst)   #for here they are similar
a$mae
26.8/mean(yD1)   #.393
mean(yD1)
#predictions
prTrn <- predict(cv_m, data.matrix(xD1))
prTst <- predict(cv_m, data.matrix(xD1Tst))


predRet_Trn <- ddTrnPP %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet= prTrn)
predRet_Trn <- predRet_Trn %>% mutate(tile=ntile(-predRet, 10))

#actual return get the percentiles
predRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(actualTerm), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )

#now the test table

predRet_Tst <- ddTestPP %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet= prTst)
predRet_Tst <- predRet_Tst %>% mutate(tile=ntile(-predRet, 10))

#actual return get the percentiles
predRet_Tst %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(actualTerm), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )


############### same thing but ridge this time##################################

cv_m<-cv.glmnet(data.matrix(xD1), yD1, family="gaussian", alpha = 0)

a<-assess.glmnet(cv_m, data.matrix(xD1Tst), newy = yD1Tst)   #for here they are similar
a$mae
40/mean(yD1)   #.600

#predictions
prTrn <- predict(cv_m, data.matrix(xD1))
prTst <- predict(cv_m, data.matrix(xD1Tst))


predRet_Trn <- ddTrnPP %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet= prTrn)
predRet_Trn <- predRet_Trn %>% mutate(tile=ntile(-predRet, 10))

#actual return get the percentiles
predRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(actualTerm), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )

#now the test table

predRet_Tst <- ddTestPP %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet= prTst)
predRet_Tst <- predRet_Tst %>% mutate(tile=ntile(-predRet, 10))

#actual return get the percentiles
predRet_Tst %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(actualTerm), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )



###now compare the above to what we get when we DON'T manually remove the correlated x vars

#preprocess the dfTrn, center seemed to be the least "bad". scale and corr were worse...
pre_proc_val <- preProcess(df, method = c("center"))

#make the test and trains sets
dfTrnPP = predict(pre_proc_val, dfTrn)
dfTestPP = predict(pre_proc_val, dfTst)


xD<-dfTrnPP %>% select(-actualReturn, -annRet, -actualTerm, -total_pymnt, -loan_status)
yD<- dfTrnPP$actualReturn


xDTst<-dfTestPP %>% select(-actualReturn, -annRet, -actualTerm, -total_pymnt, -loan_status)
yDTst<- dfTestPP$actualReturn

#cross validation
#
cv_m1<-cv.glmnet(data.matrix(xD), yD, family="gaussian", nfolds = 5, alpha = 1, type.measure = "mse")

a<-assess.glmnet(cv_m1, data.matrix(xDTst), newy = yDTst)
(a$mae)/mean(yD)


#predictions
prTrn <- predict(cv_m1, data.matrix(xD))

predRet_Trn <- dfTrnPP %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet= prTrn)
predRet_Trn <- predRet_Trn %>% mutate(tile=ntile(-predRet, 10))

#actual return get the percentiles
predRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(abs(actualTerm)), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )



#test and predict and get the percentiles
testcv <- predict(cv_m1, data.matrix(xDTst))

pred_glm_Ret <- dfTestPP %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>%  mutate(predRet=testcv)

pred_glm_Ret <- pred_glm_Ret %>% mutate(tile=ntile(-predRet, 10))

pred_glm_Ret %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                              avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(abs(actualTerm)), totA=sum(grade=="A"),
                                              totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )
#so now we have the % for train and test


#======================compare now with ridge======================

cv_m1r<-cv.glmnet(data.matrix(xD), yD, family="gaussian", nfolds = 5, alpha = 0, type.measure = "mse")

assess.glmnet(cv_m1r, data.matrix(xDTst), newy = yDTst)
a<-assess.glmnet(cv_m1r, data.matrix(xDTst), newy = yDTst)
a$mae
28.28/mean(yD)

prTrn <- predict(cv_m1r, data.matrix(xD))

predRet_Trn <- dfTrnPP %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet= prTrn)
predRet_Trn <- predRet_Trn %>% mutate(tile=ntile(-predRet, 10))

#actual return get the percentiles
predRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(abs(actualTerm)), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )


testcv <- predict(cv_m1r, data.matrix(xDTst))

pred_glm_Ret <- dfTestPP %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>%  mutate(predRet=testcv)

pred_glm_Ret <- pred_glm_Ret %>% mutate(tile=ntile(-predRet, 10))

pred_glm_Ret %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                              avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(abs(actualTerm)), totA=sum(grade=="A"),
                                              totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )

min(dfTrnPP$actualTerm)
#===========================compare with elasticnet======================================================
cv_m1e<-cv.glmnet(data.matrix(xD), yD, family="gaussian", nfolds = 5, alpha = 0.5, type.measure = "mse")

assess.glmnet(cv_m1e, data.matrix(xDTst), newy = yDTst)
27.06/mean(yD)

prTrn <- predict(cv_m1e, data.matrix(xD))

predRet_Trn <- dfTrnPP %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet= prTrn)
predRet_Trn <- predRet_Trn %>% mutate(tile=ntile(-predRet, 10))

#actual return get the percentiles
predRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(abs(actualTerm)), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )


testcv <- predict(cv_m1e, data.matrix(xDTst))

pred_glm_Ret <- dfTestPP %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>%  mutate(predRet=testcv)

pred_glm_Ret <- pred_glm_Ret %>% mutate(tile=ntile(-predRet, 10))

pred_glm_Ret %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                              avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(abs(actualTerm)), totA=sum(grade=="A"),
                                              totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )


#=================================actual returns but with 50/50 split================================================e first============
pre_proc_val <- preProcess(halfdfTrn, method = c("center"))
pre_proc_valT <- preProcess(halfdfTst, method = c("center"))

#make the test and trains sets

dfPPtrnh <- predict(pre_proc_val, halfdfTrn)
dfPPtsth <- predict(pre_proc_valT, halfdfTst)


xD<-dfPPtrnh %>% select(-loan_status, -actualTerm, -annRet, -actualReturn)
xDTst <- dfPPtsth %>% select(-loan_status, -actualTerm, -annRet, -actualReturn)


hcv_m1e<-cv.glmnet(data.matrix(xD), dfPPtrnh$actualReturn, family="gaussian", nfolds = 5, alpha = 0.5, type.measure = "mse")
a<-assess.glmnet(hcv_m1e, data.matrix(xDTst), newy = dfPPtsth$actualReturn)   #for here they are similar
a$mae
12.78/
  
  mean(dfPPtsth$actualReturn) 

prTrn <- predict(hcv_m1e, data.matrix(xD))

predRet_Trn <- dfPPtrnh %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet= prTrn)
predRet_Trn <- predRet_Trn %>% mutate(tile=ntile(-predRet, 10))

#actual return get the percentiles
predRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(abs(actualTerm)), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )
#this is the train predicted avg return table
#now for the test set

prTst <- predict(hcv_m1e, data.matrix(xDTst))

predRet_Tst <- dfPPtsth %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet= prTst)
predRet_Tst <- predRet_Tst %>% mutate(tile=ntile(-predRet, 10))

#actual return get the percentiles
predRet_Tst %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(abs(actualTerm)), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )

###########now lasso for 50/50=============================

xD<-dfPPtrnh %>% select(-loan_status, -actualTerm, -annRet, -actualReturn)
xDTst <- dfPPtsth %>% select(-loan_status, -actualTerm, -annRet, -actualReturn)


hcv_m1e<-cv.glmnet(data.matrix(xD), dfPPtrnh$actualReturn, family="gaussian", nfolds = 5, alpha = 1, type.measure = "mse")
a<-assess.glmnet(hcv_m1e, data.matrix(xDTst), newy = dfPPtsth$actualReturn)   #for here they are similar
a$mae
12.33/mean(dfPPtsth$actualReturn) 

prTrn <- predict(hcv_m1e, data.matrix(xD))

predRet_Trn <- dfPPtrnh %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet= prTrn)
predRet_Trn <- predRet_Trn %>% mutate(tile=ntile(-predRet, 10))

#actual return get the percentiles
predRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(abs(actualTerm)), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )
#this is the train predicted avg return table
#now for the test set

prTst <- predict(hcv_m1e, data.matrix(xDTst))

predRet_Tst <- dfPPtsth %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet= prTst)
predRet_Tst <- predRet_Tst %>% mutate(tile=ntile(-predRet, 10))

#actual return get the percentiles
predRet_Tst %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(abs(actualTerm)), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )


################## ridge for 50/50#####################

hcv_m1e<-cv.glmnet(data.matrix(xD), dfPPtrnh$actualReturn, family="gaussian", nfolds = 5, alpha = 0, type.measure = "mse")
a<-assess.glmnet(hcv_m1e, data.matrix(xDTst), newy = dfPPtsth$actualReturn)   #for here they are similar
a$mae
22.73/mean(dfPPtsth$actualReturn) 

prTrn <- predict(hcv_m1e, data.matrix(xD))

predRet_Trn <- dfPPtrnh %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet= prTrn)
predRet_Trn <- predRet_Trn %>% mutate(tile=ntile(-predRet, 10))

#actual return get the percentiles
predRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(abs(actualTerm)), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )
#this is the train predicted avg return table
#now for the test set

prTst <- predict(hcv_m1e, data.matrix(xDTst))

predRet_Tst <- dfPPtsth %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet= prTst)
predRet_Tst <- predRet_Tst %>% mutate(tile=ntile(-predRet, 10))

#actual return get the percentiles
predRet_Tst %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(abs(actualTerm)), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )




##########################now do subset 50/50 PP##################################################
pre_proc_val <- preProcess(data, method = c("center"))
#pre_proc_valT <- preProcess(subhalfdfTst, method = c("center"))

#make the test and trains sets

ddPPtrnh <- predict(pre_proc_val, subhalfdfTrn)
ddPPtsth <- predict(pre_proc_val, subhalfdfTst)

summary(ddPPtsth)
xD<-ddPPtrnh %>% select(-loan_status, -actualTerm, -annRet, -actualReturn)
xDTst <- ddPPtsth %>% select(-loan_status, -actualTerm, -annRet, -actualReturn)


hcv_m1e<-cv.glmnet(data.matrix(xD), dfPPtrnh$actualReturn, family="gaussian", nfolds = 5, alpha = 1, type.measure = "mse")
assess.glmnet(hcv_m1e, data.matrix(xDTst), newy = dfPPtsth$actualReturn)
46.28/mean(ddPPtsth$actualReturn)  #is this right

prTrn <- predict(hcv_m1e, data.matrix(xD))

predRet_Trn <- ddPPtrnh %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet= prTrn)
predRet_Trn <- predRet_Trn %>% mutate(tile=ntile(-predRet, 10))

#actual return get the percentiles
predRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(abs(actualTerm)), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )
#this is the train predicted avg return table
#now for the test set

prTst <- predict(hcv_m1e, data.matrix(xDTst))

predRet_Tst <- ddPPtsth %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet= prTst)
predRet_Tst <- predRet_Tst %>% mutate(tile=ntile(-predRet, 10))

#actual return get the percentiles
predRet_Tst %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(actualTerm), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )






#===================

#Gbm(xgboost) Models

#===================

df<-df %>% filter(!is.na(df$loan_status))

#Needs all data to be numeric -- so we convert categorical (i.e. factor) variables ##using one-hot encoding - multiple ways to do this
##use the dummyVars function in the 'caret' package to convert factor variables to 
##dummy-variables

fdum<-dummyVars(~.,data=df %>% select(-loan_status))
dxdf <- predict(fdum, df)
##for loan_status, check levels and convert to dummy vars and drop the second dummy var
fpdf <- class2ind(df$loan_status, drop2nd = TRUE)

#Training, test subsets
dxlcdfTrn <- dxdf[trnIndex,]
fplcdfTrn <- fpdf[trnIndex]
dxlcdfTst <- dxdf[-trnIndex,]
fplcdfTst <- fpdf[-trnIndex]

library(xgboost)

dxTrn <- xgb.DMatrix( subset(dxlcdfTrn, select=-c(annRet, actualTerm, actualReturn, total_pymnt)), label=fplcdfTrn)
dxTst <- xgb.DMatrix( subset( dxlcdfTst,select=-c(annRet, actualTerm, actualReturn, total_pymnt)), label=fplcdfTst)

##looking at the test data performance to determine the best model
xgbWatchlist <- list(train = dxTrn, eval = dxTst)
#We can watch the progress of learning thru performance on these datasets

##list of parameters for the xgboost model development functions
xgbParam <- list (
  max_depth = 4, eta = 0.01,
  objective = "binary:logistic",
  eval_metric="error", eval_metric = "auc")


##can specify which evaluation metrics we want to watch
xgb_lsM1 <- xgb.train( xgbParam, dxTrn, nrounds = 100,
                       xgbWatchlist, early_stopping_rounds = 10 )

xgb_lsM1$best_iteration

#Performance
##Predictions from the model 
xpredTrg <- predict(xgb_lsM1,dxTrn)
head(xpredTrg)

##Confusion matrix
table(pred=as.numeric(xpredTrg>0.5),act=fplcdfTrn)

##ROC, AUC performance
xpredTst<-predict(xgb_lsM1, dxTst)
pred_xgb_lsM1=prediction(xpredTst, dfTst$loan_status, label.ordering =
                           c("Charged Off", "Fully Paid"))
aucPerf_xgb_lsM1=performance(pred_xgb_lsM1, "tpr", "fpr")
aucPerf_xgb_lsm1=performance(pred_xgb_lsM1, "auc")
plot(aucPerf_xgb_lsM1)
abline(a=0, b= 1)

#Use cross-validation on training dataset to determine best model
xgbParam <- list (
  max_depth = 3, eta = 0.11,
  objective = "binary:logistic",
  eval_metric="error", eval_metric = "auc")
xgb_lscv <- xgb.cv( xgbParam, dxTrn, nrounds = 100, nfold=5, early_stopping_rounds = 10 )
##best iteration
xgb_lscv$best_iteration

##the best iteration based on performance measure (among those specified in xgbParam)
best_cvIter <- which.max(xgb_lscv$evaluation_log$test_auc_mean)

##which.min(xgb_lscv$evaluation_log$test_error_mean)
##best model
xgb_lsbest <- xgb.train( xgbParam, dxTrn, nrounds = xgb_lscv$best_iteration)

##variable importance
xgb.importance(model = xgb_lsbest) %>% view()

xgb_lscv$evaluation_log


xgbParam <- list (
  max_depth = 4,
  objective = "binary:logistic",
  eval_metric="error", eval_metric = "auc")
xgb_lsM1 <- xgb.train(xgbParam, dxTrn, nrounds = 100,
                      xgbWatchlist,early_stopping_rounds = 10,eta=1)

xgbParam <- list (objective = "binary:logistic",
                  eval_metric="error", eval_metric = "auc")

xgb_lsM1 <- xgb.train( xgbParam, dxTrn, nrounds = 100, xgbWatchlist,
                       early_stopping_rounds = 10, eta=0.1, max_depth=6 )

xgb_lsM1 <- xgb.train( xgbParam, dxTrn, nrounds = 100, xgbWatchlist,
                       early_stopping_rounds = 10, eta=0.1, max_depth=6, lambda=0.05 )

xgb_lsM1 <- xgb.train( xgbParam, dxTrn, nrounds = 100, xgbWatchlist, early_stopping_rounds = 10,
                       eta=0.1, max_depth=6, lambda=0.05, subsample=0.7, colsample_bytree=0.5 )

xgb_lsM1 <- xgb.train( xgbParam, dxTrn, nrounds = 100, xgbWatchlist, early_stopping_rounds
                       = 10, eta=0.01, max_depth=6, subsample=0.7, colsample_bytree=0.5) 


#Models for actualReturn
##Needs all data to be numeric - another way to sconvert categorical (i.e. factor) variables using one-hot encoding
ohlcdfTrn<- model.matrix(~.+0, data=dfTrnPP %>% select(-c('loan_status'))) 
##we need to exclude loan_staus in modeling actualReturn
ohlcdfTst<- model.matrix(~.+0, data=dfTestPP %>% select(-c('loan_status')))


##Model 1
dtrain <- xgb.DMatrix(subset(ohlcdfTrn,select=-c(annRet, actualTerm, actualReturn, total_pymnt)), label=ohlcdfTrn[,"actualReturn"])

xgb_Mrcv <- xgb.cv( data = dtrain, nrounds = 100, max.depth=6, nfold = 5, eta=0.1, objective="reg:squarederror")

bestIter = which.min(xgb_Mrcv$evaluation_log$test_rmse_mean)

xgb_Mr<- xgb.train(data = dtrain,nrounds =bestIter, max.depth=6, eta=0.1,objective="reg:squarederror")

xgb_Mr_importance <- xgb.importance(model=xgb_Mr)
xgb_Mr_importance%>%view()

##Model 2 
xgb_Mr2<- xgboost( data = dtrain, nrounds = bestIter, max.depth=4, eta=0.05, objective="reg:squarederror")

xgb_Mr_importance <- xgb.importance(model = xgb_Mr2)
xgb_Mr_importance %>% view()


##Model 3 
xgb_Mr3<- xgboost(data=dtrain,xgbParams, nrounds=100, eta=0.01, subsample=0.7)

xgb_Mr_importance <- xgb.importance(model = xgb_Mr3)
xgb_Mr_importance %>% view()

y_pred <- predict(xgb_Mr3,dtrain)


##Model 4 
xgbr_Mr4 <- xgb.cv( data = dtrain, nrounds = 100, max.depth=4, nfold = 5,
                    eta=0.05, objective="reg:linear" )

xgbr_Mr4

##########
##Model 5 Boosting linear models

xgb_Lin_Rcv <- xgb.cv( data = dtrain, nrounds=100, nfold = 5, eta=0.3, subsample=1,
                       early_stopping_rounds=10, booster="gblinear", alpha=0.0001)

xgb_Lin_Rcv

xgb_Lin_R1 <- xgboost( data = dtrain, nrounds = xgb_Lin_Rcv$best_iteration,
                       eta=0.3, subsample=1, booster="gblinear", alpha=0.0001 )


xgb.importance(model=xgb_Lin_R1) %>% view()

##Model 6 
xgb_Lin_R2 <- xgboost( data = dtrain, nrounds = xgb_Lin_Rcv$best_iteration,
                       eta=0.01, booster="gbtree",min_child_weight=1,
                       colsample_bytree=0.6, max_depth = 5)

xgb.importance(model=xgb_Lin_R2) %>% view()


#==========================
library(xgboost)
ohlcdfTrn<- model.matrix(~.+0, data=subsetTrn %>% select(-c('loan_status'))) 
ohlcdfTst<- model.matrix(~.+0, data=subsetTst %>% select(-c('loan_status')))


##Model 7
dtrain <- xgb.DMatrix(subset(ohlcdfTrn,select=-c(annRet, actualTerm, actualReturn)), label=ohlcdfTrn[,"actualReturn"])

xgb_Mrcv <- xgb.cv( data = dtrain, nrounds = 100, max.depth=6, nfold = 5, eta=0.1, objective="reg:squarederror")

bestIter = which.min(xgb_Mrcv$evaluation_log$test_rmse_mean)

xgb_Mr<- xgb.train(data = dtrain,nrounds =bestIter, max.depth=6, eta=0.1,objective="reg:squarederror")

xgb_Mr_importance <- xgb.importance(model=xgb_Mr)
xgb_Mr_importance%>%view()



##Model 8 
xgb_Mr2<- xgboost( data = dtrain, nrounds = bestIter, max.depth=4, eta=0.05, objective="reg:squarederror")

xgb_Mr_importance <- xgb.importance(model = xgb_Mr2)
xgb_Mr_importance %>% view()


##Model 9 
xgb_Mr3<- xgboost(data=dtrain,xgbParam, nrounds=100, eta=0.01, subsample=0.7)

xgb_Mr_importance <- xgb.importance(model = xgb_Mr3)
xgb_Mr_importance %>% view()


##Model 10 
xgbr_Mr4 <- xgb.cv( data = dtrain, nrounds = 100, max.depth=4, nfold = 5,
                    eta=0.05, objective="reg:linear" )

xgbr_Mr4

##########
##Model 11 Boosting linear models

xgb_Lin_Rcv <- xgb.cv( data = dtrain, nrounds=100, nfold = 5, eta=0.3, subsample=1,
                       early_stopping_rounds=10, booster="gblinear", alpha=0.0001)

xgb_Lin_Rcv

xgb_Lin_R1 <- xgboost( data = dtrain, nrounds = xgb_Lin_Rcv$best_iteration,
                       eta=0.3, subsample=1, booster="gblinear", alpha=0.0001 )


xgb.importance(model=xgb_Lin_R1) %>% view()

##Model 12
xgb_Lin_R2 <- xgboost( data = dtrain, nrounds = xgb_Lin_Rcv$best_iteration,
                       eta=0.01, booster="gbtree",min_child_weight=1,
                       colsample_bytree=0.6, max_depth = 5)

xgb.importance(model=xgb_Lin_R2) %>% view()

##Actual Returns
###training 
predXgbRet_Trn <- dfTrnPP %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>%
  mutate(predXgbRet=predict(xgb_Mr,subset(ohlcdfTrn,select=-c(annRet, actualTerm, total_pymnt,actualReturn))))

predXgbRet_Trn <- predXgbRet_Trn %>% mutate(tile=ntile(-predXgbRet, 10))

predXgbRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgPredRet=mean(predXgbRet), numDefaults=sum(loan_status=="Charged Off"),
                                                avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(abs(actualTerm)), totA=sum(grade=="A"),
                                                totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F"))

###testing 
predXgbRet_Tst <- dfTestPP %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>%
  mutate(predXgbRet=predict(xgb_Mr,subset(ohlcdfTst,select=-c(annRet, actualTerm, total_pymnt,actualReturn))))

predXgbRet_Tst <- predXgbRet_Tst %>% mutate(tile=ntile(-predXgbRet, 10))

predXgbRet_Tst %>% group_by(tile) %>% summarise(count=n(), avgPredRet=mean(predXgbRet), numDefaults=sum(loan_status=="Charged Off"),
                                                avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(abs(actualTerm)), totA=sum(grade=="A"),
                                                totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F"))

################
Question 4 
################
#=============
#Question 4 

#============
#Ranger Model
#=============
##Testing Models 

lg_dfTst<-dfTst %>% filter(grade=='C'| grade=='D'| grade== 'E'| grade== 'F'| grade== 'G')
lg_dfTrn<-dfTrn %>% filter(grade=='C'| grade=='D'| grade== 'E'| grade== 'F'| grade== 'G')


rf_M2_lg <- ranger(loan_status ~., data=subset(lg_dfTrn, select=-c(annRet, actualTerm, actualReturn)), num.trees =200,
                   probability=TRUE, importance='permutation')

###Actual Return
lg_scoreTrnRF <- lg_dfTrn %>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>%
  mutate(score=(predict(rf_M2_lg,lg_dfTrn))$predictions[,"Fully Paid"])
lg_scoreTrnRF <- lg_scoreTrnRF %>% mutate(tile=ntile(-score, 10))
lg_scoreTrnRF %>% group_by(tile) %>% summarise(count=n(), avgSc=mean(score),
                                               numDefaults=sum(loan_status=="Charged Off"), avgActRet=mean(actualReturn), minRet=min(actualReturn),
                                               maxRet=max(actualReturn), avgTer=mean(actualTerm), totA=sum(grade=="A"), totB=sum(grade=="B" ),
                                               totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )

importance(rf_M2_lg)

###test the performance
rfPred <- predict(rf_M2_lg, dfTrn)
###calculate the MSE
sqrt(mean(rfPred$predictions - dfTst$loan_status)^2)


###plot train and test to see what the model is getting

par(mfrow =c(1,2))
plot((predict(rf_M2_lg, dfTst))$predictions, dfTst$loan_status)
plot((predict(rf_M2_lg, dfTrn))$predictions, dfTrn$loan_status)

#====================
#GBM model for the lower grades 
#====================


#Training Models

lg_dfTrnPP<-dfTrnPP %>% filter(grade=='C'| grade=='D'| grade== 'E'| grade== 'F'| grade== 'G')
lg_subsetTrn<-subsetTrn %>% filter(grade=='C'| grade=='D'| grade== 'E'| grade== 'F'| grade== 'G')
lg_halfdfTrn<-halfdfTrn %>% filter(grade=='C'| grade=='D'| grade== 'E'| grade== 'F'| grade== 'G')
lg_dfTestPP<-dfTestPP %>% filter(grade=='C'| grade=='D'| grade== 'E'| grade== 'F'| grade== 'G')
lg_subsetTst<-subsetTst %>% filter(grade=='C'| grade=='D'| grade== 'E'| grade== 'F'| grade== 'G')
lg_halfdfTst<-halfdfTst %>% filter(grade=='C'| grade=='D'| grade== 'E'| grade== 'F'| grade== 'G')


##Model 2 preprocessing 
gbm2 <- gbm(formula=unclass(loan_status)-1 ~., data=subset(lg_dfTrnPP, select = -c(annRet, actualTerm, actualReturn, total_pymnt)), distribution = "adaboost", n.trees=300, shrinkage=0.01, interaction.depth = 4, bag.fraction=0.5, cv.folds = 5, n.cores=16) 

##Model 3 taking out high correlated variables
gbm3 <- gbm(formula=unclass(loan_status)-1 ~., data= subset(lg_subsetTrn, select = -c(annRet, actualTerm, actualReturn)), distribution = "bernoulli", n.trees=300, shrinkage=0.01, interaction.depth = 4, bag.fraction=0.5, cv.folds = 3, n.cores=16) 

##Model 4 50/50
gbm4 <- gbm(formula=unclass(loan_status)-1 ~., data= subset(lg_halfdfTrn, select = -c(annRet, actualTerm, actualReturn)), distribution = "adaboost", n.trees=300, shrinkage=0.01, interaction.depth = 4, bag.fraction=0.5, cv.folds = 5, n.cores=16)

###Actual Return 

predXgbRet_Trn <- lg_dfTrnPP %>% select(grade, loan_status, actualReturn, actualTerm) %>%
  mutate(predXgbRet=predict(gbm2,subset(lg_dfTrnPP, select = -c(annRet, actualTerm, actualReturn, total_pymnt))))
predXgbRet_Trn <- predXgbRet_Trn %>% mutate(tile=ntile(-predXgbRet, 10))
predXgbRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgPredRet=mean(predXgbRet), numDefaults=sum(loan_status=="Charged Off"),
                                                avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(actualTerm), totA=sum(grade=="A"),
                                                totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F"))

predXgbRet_Tst <- lg_dfTestPP %>% select(grade, loan_status, actualReturn, actualTerm, total_pymnt) %>%
  mutate(predXgbRet=predict(gbm2,subset( lg_dfTestPP,select=-c(annRet, actualTerm, total_pymnt,actualReturn))))
predXgbRet_Tst <- predXgbRet_Tst %>% mutate(tile=ntile(-predXgbRet, 10))
predXgbRet_Tst %>% group_by(tile) %>% summarise(count=n(), avgPredRet=mean(predXgbRet), numDefaults=sum(loan_status=="Charged Off"),
                                                avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(abs(actualTerm)), totA=sum(grade=="A"),
                                                totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F"))



###Summary 
summary(gbm2)
summary(gbm3)
summary(gbm4)

###CM

pred_2=predict(gbm2,lg_dfTestPP, type="response") 
head(pred_2)
table(pred_2>0.5,lg_dfTestPP$loan_status)

pred_3=predict(gbm3,lg_subsetTst, type="response") 
head(pred_3)
table(pred_3>0.5,lg_subsetTst$loan_status)

pred_4=predict(gbm4,lg_halfdfTst, type="response") 
head(pred_4)
table(pred_4>0.5,lg_halfdfTst$loan_status)

###Performance of the model

print(gbm2)
bestIter<-gbm.perf(gbm2, method='cv')
scores_gbm2<- predict(gbm2, newdata=lg_dfTestPP, n.tree= bestIter, type="response")
head(scores_gbm2)

print(gbm3)
bestIter<-gbm.perf(gbm3, method='cv')
scores_gbm3<- predict(gbm3, newdata=lg_subsetTst, n.tree= bestIter, type="response")
head(scores_gbm3)

print(gbm4)
bestIter<-gbm.perf(gbm4, method='cv')
scores_gbm4<- predict(gbm4, newdata=lg_halfdfTst, n.tree= bestIter, type="response")
head(scores_gbm4)

###MSE 

min(gbm2$cv.error)
sqrt(min(gbm2$cv.error))

min(gbm3$cv.error)
sqrt(min(gbm3$cv.error))

min(gbm4$cv.error)
sqrt(min(gbm4$cv.error))

###ROC AUC Graphs

pred_gbm2=prediction(scores_gbm2,lg_dfTestPP$loan_status, label.ordering = c("Fully Paid", "Charged Off"))
aucPerf_gbm2 <-performance(pred_gbm2, "tpr", "fpr")
aucPerf_gbm12=performance(pred_gbm2, "auc")
aucPerf_gbm2@y.values
plot(aucPerf_gbm2,col=2,legacy.axes=TRUE,print.auc=TRUE,main="ROC(GBM)")
abline(a=0, b= 1)


pred_gbm3=prediction(scores_gbm3,lg_subsetTst$loan_status, label.ordering = c("Fully Paid", "Charged Off"))
aucPerf_gbm3 <-performance(pred_gbm3, "tpr", "fpr")
aucPerf_gbm13=performance(pred_gbm3, "auc")
aucPerf_gbm3@y.values
plot(aucPerf_gbm3,col=4,add=TRUE)

pred_gbm4=prediction(scores_gbm4,lg_halfdfTst$loan_status, label.ordering = c("Fully Paid", "Charged Off"))
aucPerf_gbm4 <-performance(pred_gbm4, "tpr", "fpr")
aucPerf_gbm14=performance(pred_gbm4, "auc")
aucPerf_gbm4@y.values
plot(aucPerf_gbm4,col=5,add=TRUE)


###Relative Influence Bar Graphs for the models
par(oma=c(1,5.0,1,1))
summary(gbm2,6,19,las=2,main="Model 2")
par(oma=c(1,6.0,1,1))
summary(gbm3,7,19,las=2,main="Model 3")
summary(gbm4,7,19,las=2,main="Model 4")


#======make glmnet for the subset===========================================

lg_dfTst<-dfTestPP %>% filter(grade=='C'| grade=='D'| grade== 'E'| grade== 'F'| grade== 'G')
lg_dfTrn<-dfTrnPP %>% filter(grade=='C'| grade=='D'| grade== 'E'| grade== 'F'| grade== 'G')

xD<-lg_dfTrn %>% select(-loan_status, -actualTerm, -annRet, -actualReturn, -total_pymnt)
xDTst <- lg_dfTst %>% select(-loan_status, -actualTerm, -annRet, -actualReturn, -total_pymnt)

sublg_dfTrn$loan_status

sublg_dfTst<-ddTestPP %>% filter(grade=='C'| grade=='D'| grade== 'E'| grade== 'F'| grade== 'G')
sublg_dfTrn<-ddTrnPP %>% filter(grade=='C'| grade=='D'| grade== 'E'| grade== 'F'| grade== 'G')

xDsub<-sublg_dfTrn %>% select(-loan_status, -actualTerm, -annRet, -actualReturn)
xDsubTst <- sublg_dfTst %>% select(-loan_status, -actualTerm, -annRet, -actualReturn)

#lasso, ridge, enet- with the PP reg
set.seed(1234)
glmDefault_cvL<- cv.glmnet(data.matrix(xD), lg_dfTrn$loan_status, family="binomial", nfolds=3, alpha=1, type.measure = "auc",)
glmDefault_cvR<- cv.glmnet(data.matrix(xD), lg_dfTrn$loan_status, family="binomial", nfolds=5, alpha=0, type.measure = "auc")
glmDefault_cvE<- cv.glmnet(data.matrix(xD), lg_dfTrn$loan_status, family="binomial", nfolds=5, alpha=0.5, type.measure = "auc")

#let's do elastic with the subset
glmDefault_cvS <- cv.glmnet(data.matrix(xDsub), sublg_dfTrn$loan_status, family="binomial", nfolds=5, alpha=1, type.measure = "auc")
glmDefault_cvL$lambda

mean<-glmDefault_cvL$cvm
mean(mean) #.669
mean<-glmDefault_cvR$cvm
mean(mean) #.644
mean<-glmDefault_cvE$cvm
mean(mean) #.665
mean<-glmDefault_cvS$cvm
mean(mean) #.600

#get AUC and MAE
a<-assess.glmnet(glmDefault_cvE, data.matrix(xDTst), newy = lg_dfTst$loan_status)
a$mae
.6564/mean(lg_dfTst$loan_status)
a

a<-assess.glmnet(glmDefault_cvR, data.matrix(xDTst), newy = lg_dfTst$loan_status)
a$mae
.6514/mean(lg_dfTst$loan_status)
a

a<-assess.glmnet(glmDefault_cvL, data.matrix(xDTst), newy = lg_dfTst$loan_status)
a$mae
.6556/mean(lg_dfTst$loan_status)
a

a<-assess.glmnet(glmDefault_cvS, data.matrix(xDsubTst), newy = sublg_dfTst$loan_status)
a$mae
.682/mean(sublg_dfTst$loan_status)
a

par(mfrow=c(2,2))
plot(glmDefault_cvL)
plot(glmDefault_cvR)
plot(glmDefault_cvE)
plot(glmDefault_cvS)


confusion.glmnet(glmDefault_cvL, data.matrix(xDTst), newy = lg_dfTst$loan_status)  #.7696
confusion.glmnet(glmDefault_cvR, data.matrix(xDTst), newy = lg_dfTst$loan_status)  #.7711
confusion.glmnet(glmDefault_cvE, data.matrix(xDTst), newy = lg_dfTst$loan_status)  #.7698

confusion.glmnet(glmDefault_cvS, data.matrix(xDsubTst), newy = sublg_dfTst$loan_status)  #.7764


############now do it without PP ############

lg_dfTst<-dfTst %>% filter(grade=='C'| grade=='D'| grade== 'E'| grade== 'F'| grade== 'G')
lg_dfTrn<-dfTrn %>% filter(grade=='C'| grade=='D'| grade== 'E'| grade== 'F'| grade== 'G')

xD<-lg_dfTrn %>% select(-loan_status, -actualTerm, -annRet, -actualReturn, -total_pymnt, -int_rate)
xDTst <- lg_dfTst %>% select(-loan_status, -actualTerm, -annRet, -actualReturn, -total_pymnt, -int_rate)


sublg_dfTst<-subsetTst %>% filter(grade=='C'| grade=='D'| grade== 'E'| grade== 'F'| grade== 'G')
sublg_dfTrn<-subsetTrn %>% filter(grade=='C'| grade=='D'| grade== 'E'| grade== 'F'| grade== 'G')

xDsub<-sublg_dfTrn %>% select(-loan_status, -actualTerm, -annRet, -actualReturn)
xDsubTst <- sublg_dfTst %>% select(-loan_status, -actualTerm, -annRet, -actualReturn)


#lasso, ridge, enet and subset no PP

glmDefault_cvL<- cv.glmnet(data.matrix(xD), lg_dfTrn$loan_status, family="binomial", nfolds=5, alpha=1, type.measure = "auc")
glmDefault_cvR<- cv.glmnet(data.matrix(xD), lg_dfTrn$loan_status, family="binomial", nfolds=5, alpha=0, type.measure = "auc")
glmDefault_cvE<- cv.glmnet(data.matrix(xD), lg_dfTrn$loan_status, family="binomial", nfolds=5, alpha=0.5, type.measure = "auc")

#let's do elastic with the subset
glmDefault_cvS <- cv.glmnet(data.matrix(xDsub), sublg_dfTrn$loan_status, family="binomial", nfolds=5, alpha=1, type.measure = "auc")

mean<-glmDefault_cvL$cvm
mean(mean) 
mean<-glmDefault_cvR$cvm
mean(mean) 
mean<-glmDefault_cvE$cvm
mean(mean) 
mean<-glmDefault_cvS$cvm
mean(mean) 

#get AUC and MAE
a<-assess.glmnet(glmDefault_cvE, data.matrix(xDTst), newy = lg_dfTst$loan_status)
a$mae
.6553/mean(lg_dfTst$loan_status)
a

a<-assess.glmnet(glmDefault_cvR, data.matrix(xDTst), newy = lg_dfTst$loan_status)
a$mae
.6523/mean(lg_dfTst$loan_status)
a

a<-assess.glmnet(glmDefault_cvL, data.matrix(xDTst), newy = lg_dfTst$loan_status)
a$mae
.6556/mean(lg_dfTst$loan_status)
a

a<-assess.glmnet(glmDefault_cvS, data.matrix(xDsubTst), newy = sublg_dfTst$loan_status)
a$mae
.682/mean(sublg_dfTst$loan_status)
a

par(mfrow=c(2,2))
plot(glmDefault_cvL)
plot(glmDefault_cvR)
plot(glmDefault_cvE)
plot(glmDefault_cvS)


confusion.glmnet(glmDefault_cvL, data.matrix(xDTst), newy = lg_dfTst$loan_status)  #.7696
confusion.glmnet(glmDefault_cvR, data.matrix(xDTst), newy = lg_dfTst$loan_status)  #.7707
confusion.glmnet(glmDefault_cvE, data.matrix(xDTst), newy = lg_dfTst$loan_status)  #.7698

confusion.glmnet(glmDefault_cvS, data.matrix(xDsubTst), newy = sublg_dfTst$loan_status)  #.7764


>>>>>>> e3d1b0ee8744770dd68041514a82be1ef12d259f




