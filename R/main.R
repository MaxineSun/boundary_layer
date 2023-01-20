#install.packages("xlsx")
#install.packages("lme4")
#install.packages('caTools') 
library(caTools)
library(xlsx)
library(lme4)

data <- read.xlsx("data_aggregated_without_zeros.xlsx", sheetIndex = 1) 
data <- data[1:185, 1:5]
data[1:185, 3] = log10(data[1:185, 3])
data[1:185, 4] = log10(data[1:185, 4])
set.seed(4) 
split = sample.split(data$thickness, SplitRatio = 145) 
training_set = subset(data, split == TRUE) 
test_set = subset(data, split == FALSE)

model <- lmer(thickness~Voltage+espr+(1+Voltage+espr|c_bulk)+(1+Voltage+espr|c_max)+(1+Voltage+espr+c_bulk|c_max),data=training_set,REML=TRUE)
predict <- predict(model, test_set[, 1:4])
test_MSE = sum((predict - test_set[, 5])^2)/40
