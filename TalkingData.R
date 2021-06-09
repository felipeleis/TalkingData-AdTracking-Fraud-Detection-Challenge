# Fraud risk is everywhere, but for companies that advertise online, click fraud can
# happen at an overwhelming volume, resulting in misleading click data and wasted money.
# Ad channels can drive up costs by simply clicking on the ad at a large scale. With
# over 1 billion smart mobile devices in active use every month, China is the largest
# mobile market in the world and therefore suffers from huge volumes of fradulent traffic.
#
# TalkingData, China’s largest independent big data service platform, covers over 70%
# of active mobile devices nationwide. They handle 3 billion clicks per day, of which 90%
# are potentially fraudulent. Their current approach to prevent click fraud for app
# developers is to measure the journey of a user’s click across their portfolio, and flag
# IP addresses who produce lots of clicks, but never end up installing apps. With this
# information, they've built an IP blacklist and device blacklist.

# While successful, they want to always be one step ahead of fraudsters and have turned to
# the Kaggle community for help in further developing their solution. In their 2nd
# competition with Kaggle, you’re challenged to build an algorithm that predicts whether
# a user will download an app after clicking a mobile app ad. To support your modeling,
# they have provided a generous dataset covering approximately 200 million clicks over
# 4 days! Each row of the training data contains a click record, with the following features:
# 
# ip: ip address of click.
# app: app id for marketing.
# device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.).
# os: os version id of user mobile phone.
# channel: channel id of mobile ad publisher.
# click_time: timestamp of click (UTC).
# attributed_time: if user download the app for after clicking an ad, this is the time of the app download.
# is_attributed: the target that is to be predicted, indicating the app was downloaded.
# 
# Note that ip, app, device, os, and channel are encoded.

library(dplyr)
library(ggplot2)
library(reshape2)
library(tidyr)
library(e1071)
library(caret)
library(readr)
library(data.table)
library(devtools)

# Reading the file with the 'readr' package, because it is relatively too large for 'utils'.
df <- read_csv("train_sample.csv")
df
glimpse(df)
dim(df)
summary(df)

# The entire dataset is numeric with two date-like columns.

# There are NA values in the 'attributed_time' column, which are probably people
# who clicked on the ad and did not download the app.
any(is.na(df))

# Is that column the only one with NA values?
any(is.na(df[,-which(names(df)=="attributed_time")]))
# Yes.

sum(is.na(df[,which(names(df)=="attributed_time")]))/length(df$attributed_time)
round(prop.table(table(df$is_attributed)) * 100, digits = 1)

# In fact, 99.8% of that column is composed of NA values.

# NA values in that column are not really a problem. Actually,
# those are the ones who clicked on the ad and did not download it.
# This information is likely important for what we want to predict (click fraud).

# Creating a column with the time difference between click and download:
df$time_diff = df$attributed_time - df$click_time

filter(df, is.na(attributed_time)==FALSE)
filter(df, df$is_attributed==1)

# How does the time difference between click and download relate with 'ip'?
ip_vs_time <- df %>%
  filter(is.na(attributed_time)==FALSE) %>% # Those who downloaded the app
              group_by(ip) %>%
              summarise(media = mean(time_diff),
                        std = sd(time_diff),
                        n = n())
ip_vs_time

# There are a lot of ips with unique access for which it is not possible
# to calculate the associated standard deviation.

# Some ips with a low mean 'time_diff' accessed the link more than once
# and these 'ip's are:
filter(ip_vs_time, n!=1)

# The number of downloads is always either 1 or 3 for each ip.
# The number of downloads does not seem to be relevant.

library(cowplot)
main_plot <- ggplot(ip_vs_time, aes(ip, media, ymin = media-std, ymax = media+std)) +
  geom_line() +
  theme_classic() +
  geom_pointrange(color = 'red', size=0.2) +
  xlab("IP") +
  ylab("Mean time difference between\nclick and download (seconds)")
inset_plot <- ggplot(ip_vs_time, aes(ip, media, ymin = media-std, ymax = media+std)) +
  geom_line() +
  theme_classic() +
  geom_pointrange(color = 'red', size=0.5) +
  coord_cartesian(xlim = c(5300,5360), ylim = c(0,3000)) +
  xlab("") + ylab("")
ggdraw(main_plot) +
draw_plot(inset_plot, x = 0.4, y = 0.7, width = .25, height = .3)

# How about those that did not download the app?
ip_no_dl <- df %>%
  filter(is.na(attributed_time)==TRUE) %>% # Those who did not download the app
  group_by(ip) %>%
  summarise(n = n()) %>%
  arrange(desc(n))
ip_no_dl

# The 'ip's that downloaded the app three times (5348 and 5314)
# are the ones that clicked on the ad the most.

ggplot(ip_no_dl, aes(ip, n)) +
  geom_step() +
  theme_classic() +
  xlab("IP") + ylab("Number of ad clicks\nwithout download")

# Putting it in a log scale (log(y)) for better visualization:
ggplot(ip_no_dl, aes(ip, n)) +
  geom_step() +
  theme_classic() +
  scale_y_log10() +
  xlab("IP") + ylab("Number of ad clicks\nwithout download") +
  annotation_logticks()

# A lot of 'ip's coded with low numbers have accessed the link many
# times and did not download the app.

# How are the variables correlated?
library(corrplot)

# Removing columns with NA so as to plot correlations
# and not have problems with machine learning models:
df_no_NA = df[,-which(names(df) %in% c("attributed_time", "time_diff"))]
df_no_NA

# Which columns are numeric?
apply(df_no_NA, 2, is.numeric)

# Transforming them into numeric columns to plot the correlations.
df_no_NA$click_time = as.POSIXct(df_no_NA$click_time)
df_numeric = data.frame(sapply(df_no_NA[,1:ncol(df_no_NA)], as.numeric))
apply(df_numeric, 2, is.numeric)

corrplot(cor(df_numeric, method="pearson"), method = 'circle')
# Only 'ip' and 'app' are very weakly and positively correlated with the target variable 
# ('is_attributed'). Naturally, 'device' and 'os' correlate with the app and with each other.

# Transforming target variable into a two-level factor:
df_numeric$is_attributed = as.factor(df_numeric$is_attributed)

# Splitting into train and test sets:
lines = sample(1:nrow(df_numeric), 0.7*nrow(df_numeric))
train = df_numeric[lines,]
test = df_numeric[-lines,]

# Because there were way too many NAs (is_attributed = 0),
# we have an unbalanced dataset problem:
table(train$is_attributed)
table(test$is_attributed)

# Applying Random Oversampling to balance the dataset's class variables:
library(ROSE)
rose_train = ROSE(is_attributed~., data = train, na.action = NULL , seed = 1)$data
rose_test = ROSE(is_attributed~., data = test, na.action = NULL , seed = 1)$data

table(rose_train$is_attributed)
table(rose_test$is_attributed)

# Creating a Naive-Bayes model:
glimpse(rose_train)
glimpse(rose_test)

model = train(
  rose_train[,-ncol(rose_train)],
  rose_train[,ncol(rose_train)],
  method='naive_bayes',
  trControl = trainControl(method="repeatedcv", number=10, repeats=3))
model

# Making predictions on test data:
pred = predict(model, rose_test[,-ncol(rose_test)])

# Confusion Matrix: 
confusionMatrix(pred, rose_test$is_attributed)

# Generating ROC curve and calculating AUC:
library(ROCR)
library(pROC)

pred_prob = predict(model, rose_test[,-ncol(rose_test)], type = 'prob')

pred_real <- prediction(pred_prob[,2], rose_test$is_attributed)
perf <- performance(pred_real, "tpr","fpr") 

roc.curve(rose_test$is_attributed, pred_prob[,2], plotit = T,
          col = "red", lwd = 2)
legend("bottomright", c("NB"),
       col = c('red'),
       lwd = 2, bty="n", inset=c(0, 0),
       xpd = TRUE, horiz = TRUE)

# Changing the model to Linear Discriminant Analysis (LDA):
model2 = train(rose_train[,-ncol(rose_train)], rose_train[,ncol(rose_train)], method='lda')
model2

pred2 = predict(model2, rose_test[,-ncol(rose_test)], type='prob')

pred_real2 <- prediction(pred2[,2], rose_test$is_attributed)
perf2 <- performance(pred_real2, "tpr","fpr")

roc.curve(rose_test$is_attributed, pred_prob[,2], plotit = T,
          col = "red", lwd = 2)
roc.curve(rose_test$is_attributed, pred2[,2], plotit = T,
          col = "black", lwd = 2, add=TRUE)
legend("bottomright", c("NB", "LDA"),
       col = c('red', 'black'),
       lwd = 2, bty="n", inset=c(0, 0),
       xpd = TRUE, horiz = TRUE)

# Changing the model to Decision Tree (rpart):
model3 = train(rose_train[,-ncol(rose_train)], rose_train[,ncol(rose_train)], method='rpart')
model3

pred3 = predict(model3, rose_test[,-ncol(rose_test)], type='prob')

pred_real3 <- prediction(pred3[,2], rose_test$is_attributed)
perf3 <- performance(pred_real3, "tpr","fpr") 

roc.curve(rose_test$is_attributed, pred_prob[,2], plotit = T,
          col = "red", lwd = 2)
roc.curve(rose_test$is_attributed, pred2[,2], plotit = T,
          col = "black", lwd = 2, add=TRUE)
roc.curve(rose_test$is_attributed, pred3[,2], plotit = T,
          col = "blue", lwd = 2, add=TRUE)
legend("bottomright", c("NB", "LDA", 'DT'),
       col = c('red', 'black', 'blue'),
       lwd = 2, bty="n", inset=c(0, 0),
       xpd = TRUE, horiz = TRUE)

# The Naive-Bayes model has an excellent performance, even when compared with
# LDA or Decision Tree.

# Testing Random Forest:
library(randomForest)
model4 <- randomForest(is_attributed ~ . , 
                               data = rose_train, 
                               ntree = 100, 
                               nodesize = 10)
model4

pred4 = predict(model4, rose_test[,-ncol(rose_test)], type='prob')

pred_real4 <- prediction(pred4[,2], rose_test$is_attributed)
perf4 <- performance(pred_real4, "tpr","fpr") 

roc.curve(rose_test$is_attributed, pred_prob[,2], plotit = T,
          col = "red", lwd = 2)
roc.curve(rose_test$is_attributed, pred2[,2], plotit = T,
          col = "black", lwd = 2, add=TRUE)
roc.curve(rose_test$is_attributed, pred3[,2], plotit = T,
          col = "blue", lwd = 2, add=TRUE)
roc.curve(rose_test$is_attributed, pred4[,2], plotit = T,
          col = "green", lwd = 2, add=TRUE)
legend("bottomright", c("NB", "LDA", 'DT', 'RF'),
       col = c('red', 'black', 'blue', 'green'),
       lwd = 2, bty="n", inset=c(0, 0),
       xpd = TRUE, horiz = TRUE)

# Random Forest performs really well, much like Naive-Bayes.

varImpPlot(model4)

# 'app', 'os', and 'ip' are the most important variables for Random Forest.
# As to Naive-Bayes:

varImp(model)

# 'app', 'ip' and 'channel' are the most important.
# 'app' and 'ip' definitively should be in the model.

# Testing Naive-Bayes with a new column: number of clicks per ip.
df_n <- df_numeric %>%
  group_by(ip) %>%
  mutate(n=n()) %>%
  relocate(n, .before = is_attributed)

# Are there any duplicated rows in the dataset?
dim(df_n)
dim(unique(df_n))

# There is only one duplicated row (same value in all columns, even time stamp!).
# Let's not ignore this information as it might be important.
duplicated_row = which(duplicated(df_n))
df_n[duplicated_row,]
df_n[df_n$ip==871,]

# It is quite weird that the duplicated row, relative
# to ip 871, has even the same click_time.

# Reconstructing the NB model with this added column:
lines2 = sample(1:nrow(df_n), 0.7*nrow(df_n))
train2 = df_n[lines2,]
test2 = df_n[-lines2,]

rose_train2 = ROSE(is_attributed~., data = train2, na.action = NULL , seed = 1)$data
rose_test2 = ROSE(is_attributed~., data = test2, na.action = NULL , seed = 1)$data

table(rose_train2$is_attributed)
table(rose_test2$is_attributed)

model_NB = train(rose_train2[,-ncol(rose_train2)], rose_train2[,ncol(rose_train2)], method='naive_bayes')
model_NB

# Making predictions on test data:
pred_NB = predict(model_NB, rose_test2[,-ncol(rose_test2)])

confusionMatrix(pred_NB, rose_test2$is_attributed)

pred_NB = predict(model_NB, rose_test2[,-ncol(rose_test2)], type='prob')

pred_real_NB <- prediction(pred_NB[,2], rose_test2$is_attributed)
perf_NB <- performance(pred_real_NB, "tpr","fpr") 

roc.curve(rose_test2$is_attributed, pred_NB[,2], plotit = T, col = "red")

# Naive-Bayes and Random Forest are two models that perform really well in this dataset.
# The chosen model could be Naive-Bayes with 6 predictor variables: 
# 'ip', 'app', 'device', 'os', 'channel', and 'click_time'.
# 'app' and 'ip' are two very important features for both Random Forest and Naive-Bayes.
