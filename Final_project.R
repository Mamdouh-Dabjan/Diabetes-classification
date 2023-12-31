# Install and load required libraries
packages <- c("tidyverse", "e1071", "cowplot", "caret", "randomForest", "xgboost", 
              "lattice", "readr", "doParallel", "iterators", "parallel", "dplyr"
              ,"rpart","FactoMineR")
#packages <- setdiff(packages, rownames(installed.packages()))
if(length(packages)) install.packages(packages)
lapply(packages, library, character.only = TRUE)

# Set working directory and read the dataset
setwd("C:/College/10 -fall 2023/Data Mining/Final Projcet")
df <- read_csv("diabetes_prediction_dataset.csv")
diabetes_label <- df$diabetes  # Just the 'diabetes' column


# Check for missing values
colSums(is.na(df))

df_features <- df %>% select(-diabetes)  # All columns except 'diabetes'
diabetes_label <- df$diabetes  # Just the 'diabetes' column

# Identify numeric and non-numeric columns
numeric_col <- names(df_features)[sapply(df_features, is.numeric)]
non_numeric_col <- names(df_features)[!names(df_features) %in% numeric_col]

# Data Preprocessing
df_features$smoking_history <- as.character(df_features$smoking_history)
df_features$smoking_history[df_features$smoking_history == "not_current"] <- "former"
df_features$smoking_history[df_features$smoking_history == "ever"] <- "never"
df_features$smoking_history <- as.factor(df_features$smoking_history)

# Create 'forced_glucose_intake' feature
df_features$forced_glucose_intake <- ifelse((df_features$blood_glucose_level > 140) & (df_features$hypertension == 1) & (df_features$age >= 40), 1, 0)

# Encode non-numeric columns using Label Encoding
df_features[non_numeric_col] <- lapply(df_features[non_numeric_col], function(x) as.numeric(as.factor(x)))

# Ensure the diabetes label encoding is correct

# Select only numeric columns for clustering
df_numeric <- df_features %>% select_if(is.numeric)

# Scale the features (excluding the label)
df_scaled <- as.data.frame(scale(df_numeric ))

# Splitting the dataset into training and testing sets
set.seed(123)  # For reproducibility
partition <- createDataPartition(df$diabetes, p = 0.7, list = FALSE)

# Apply preProcess only after splitting
training_set <- df_scaled[partition, ]
testing_set <- df_scaled[-partition, ]

pre_proc_val <- preProcess(training_set, method = c("center", "scale"))
training_set_scaled <- predict(pre_proc_val, training_set)
testing_set_scaled <- predict(pre_proc_val, testing_set)

X_train <- training_set_scaled
y_train <- as.factor(diabetes_label[partition])
X_test <- testing_set_scaled
y_test <- as.factor(diabetes_label[-partition])

#--------------------------------------------------------------------------done
# Univariate analysis functions
univariate_analysis_numeric <- function(col) {
  hist_plot <- ggplot(df, aes_string(x = col)) +
    geom_histogram(fill = 'skyblue', bins = 20) +
    labs(title = paste("Histogram of", col))
  
  box_plot <- ggplot(df, aes_string(x = col)) +
    geom_boxplot() +
    labs(title = paste("Boxplot diagram of", col))
  
  plot_grid(hist_plot, box_plot, ncol = 2)
}

univariate_analysis_cat <- function(col) {
  count_plot <- ggplot(df, aes_string(x = col, fill = "..count..")) +
    geom_bar(position = "dodge") +
    labs(title = paste("Countplot for", col))
  
  pie_plot <- ggplot(df, aes(x = factor(1), fill = ..count..)) +
    geom_bar() +
    coord_polar(theta = "y") +
    labs(title = paste("Pie plot for", col))
  
  plot_grid(count_plot, pie_plot, ncol = 2)
}

# Execute univariate analysis
lapply(numeric_col, univariate_analysis_numeric)
lapply(non_numeric_col, univariate_analysis_cat)

#=====================================================================done
# Random Forest model training and evaluation
rf_model <- randomForest(x = X_train, y = y_train, ntree = 100)
rf_predictions <- predict(rf_model, newdata = X_test)
rf_conf_matrix <- confusionMatrix(rf_predictions, y_test)
print("Random Forest Model Evaluation:")
print(rf_conf_matrix)

#-----------------------------------------------------------------done 
#XGBOOST
y_train <- as.numeric(as.character(y_train))
y_test <- as.numeric(as.character(y_test))

# Train XGBoost model
xgb_model <- xgboost(data = as.matrix(X_train), label = as.numeric(y_train), nrounds = 10, objective = "binary:logistic")

# Make predictions on the test set
y_pred <- predict(xgb_model, as.matrix(X_test))

# Convert probabilities to binary predictions
y_pred_binary <- ifelse(y_pred > 0.5, 1, 0)
y_pred_binary <- as.numeric(as.character(y_pred_binary))
# Evaluate the model
y_pred_binary <- as.factor(y_pred_binary)
levels(y_pred_binary) <- c("0", "1")

# Convert y_test to a factor with levels "0" and "1" if it's not already
y_test <- as.factor(y_test)
levels(y_test) <- c("0", "1")

# Calculate the confusion matrix
conf_matrix <- confusionMatrix(data = y_pred_binary, reference = y_test)
# Print classification report
print(conf_matrix)
#-------------------------------------------------------------------------------done
# SVM model training and evaluation (using radial kernel as an example)
svm_model <- train(x = X_train, y = y_train, method = "svmRadial", trControl = trainControl(method = "cv", number = 10))
svm_predictions <- predict(svm_model, newdata = X_test)
svm_conf_matrix <- confusionMatrix(svm_predictions, y_test)
print("SVM Model Evaluation:")
print(svm_conf_matrix)
#-------------------------------------------------------------------------done
# Clustering (K-means) and purity calculation
df$forced_glucose_intake <- ifelse((df$blood_glucose_level > 140) & (df$hypertension == 1) & (df$age >= 40), 1, 0)
df[non_numeric_col] <- lapply(df[non_numeric_col], function(x) as.numeric(as.factor(x)))
df_numeric <- df %>% select_if(is.numeric)
df_scaled <- as.data.frame(scale(df_numeric ))
kmeans_result <- kmeans(df_scaled, centers = 2, nstart = 25)
df$cluster <- as.factor(kmeans_result$cluster)
contingency_table <- table(df$diabetes, df$cluster)
cluster_purities <- apply(contingency_table, 2, function(cluster_column) {
  max(cluster_column) / sum(cluster_column)
})
overall_purity <- sum(apply(contingency_table, 2, max)) / sum(contingency_table)
print("Clustering Purity Evaluation:")
print("Purity for each cluster:")
print(cluster_purities)
print(paste("Overall purity:", overall_purity))
#============================================================================done
# Bagging model using Random Forest as a base learner
bagging_model <- randomForest(x = X_train, y = as.factor(y_train), ntree = 100, type = "classification", bagging = TRUE)
bagging_predictions <-predict(bagging_model, newdata = X_test)
levels(bagging_predictions)
levels(y_test)
levels(bagging_predictions) <- levels(y_test)
conf_matrix <- confusionMatrix(bagging_predictions, y_test)
print("Bagging Model Evaluation:")
print(conf_matrix)
#-----------------------------------------------------------(Pruned)------done
y_train_bagging = as.factor(y_train)
y_test_bagging = as.factor(y_test)
pruned_tree_model <- rpart(y_train_bagging ~ ., data = X_train, control = rpart.control(cp = 0.01)) # Adjust 'cp' for tree complexity
pruned_tree_predictions <- predict(pruned_tree_model, X_test, type = "class")
pruned_tree_conf_matrix <- confusionMatrix(pruned_tree_predictions, y_test_bagging)
print("Pruned Tree Model Evaluation:")
print(pruned_tree_conf_matrix)
#------------------------------------------------------------(unpured)------done
# Convert y_train to factor if it is not already
y_train <- factor(y_train)

# Train an unpruned Random Forest model (for classification)
rf_unpruned <- randomForest(x = X_train, y = y_train, ntree = 100)

# Make predictions on the test set
y_pred_unpruned <- predict(rf_unpruned, newdata = X_test)

# Ensure y_test is a factor and has the same levels as y_train
y_test <- factor(y_test, levels = levels(y_train))

# Evaluate the model
conf_matrix <- confusionMatrix(factor(y_pred_unpruned), y_test)
print(conf_matrix)
#-------------------------------------------------------------------------done
# SVM model with a different kernel (e.g., polynomial kernel)
# Define the parameter grid and train for svmLinear
tuneGridLinear <- expand.grid(C = seq(0.1, 10, length.out = 10))
svm_model_linear <- train(
  x = X_train, y = y_train, method = "svmLinear",
  trControl = train_control, tuneLength = 10,
  tuneGrid = tuneGridLinear, metric = "Accuracy"
)

# Define the parameter grid and train for svmPoly
tuneGridPoly <- expand.grid(
  degree = seq(1, 5, length.out = 5),
  scale = seq(0.1, 1, length.out = 5),
  C = seq(0.1, 10, length.out = 10)
)
svm_model_poly <- train(
  x = X_train, y = y_train, method = "svmPoly",
  trControl = train_control, tuneLength = 10,
  tuneGrid = tuneGridPoly, metric = "Accuracy"
)

# Define the parameter grid and train for svmRadial
tuneGridRadial <- expand.grid(
  sigma = seq(0.01, 1, length.out = 10),
  C = seq(0.1, 10, length.out = 10)
)
svm_model_radial <- train(
  x = X_train, y = y_train, method = "svmRadial",
  trControl = train_control, tuneLength = 10,
  tuneGrid = tuneGridRadial, metric = "Accuracy"
)

print_classification_report <- function(model, X_test, y_test) {
  predictions <- predict(model, newdata = X_test)
  conf_matrix <- confusionMatrix(predictions, y_test)
  print(conf_matrix)
}

# Print classification report for each kernel
cat("Classification Report for Linear Kernel:\n")
print_classification_report(svm_model_linear, X_test, y_test)

cat("\nClassification Report for Polynomial Kernel:\n")
print_classification_report(svm_model_poly, X_test, y_test)

cat("\nClassification Report for Radial Kernel:\n")
print_classification_report(svm_model_radial, X_test, y_test)
#-----------------------------------------------------------------(SVM updated)
registerDoParallel(cores = detectCores())

# Define the parameter grid
tuneGrid <- expand.grid(
  C = seq(0.1, 10, length.out = 20), # Cost parameter
  sigma = seq(0.01, 1, length.out = 20) # Sigma parameter (1/gamma)
)

# Define training control with random search
train_control <- trainControl(
  method = "cv",
  number = 10,
  search = "random",
  allowParallel = TRUE
)

# Fit the SVC model with the radial kernel
set.seed(42)
svm_model <- train(
  x = X_train,
  y = y_train,
  method = "svmRadial", # Radial basis function kernel
  trControl = train_control,
  tuneLength = 10, # Number of different parameter sets to try
  tuneGrid = tuneGrid,
  metric = "Accuracy"
)

# Print the best parameters
print("Best Parameters for SVC: ")
print(svm_model$bestTune)

# Evaluate the model on the test set
predictions <- predict(svm_model, newdata = X_test)
conf_matrix <- confusionMatrix(predictions, y_test)

# Print SVC accuracy
print(paste("SVC Accuracy: ", round(conf_matrix$overall['Accuracy'] * 100, 2), "%"))

# Print classification report
print(conf_matrix)

#========================================================(Random forest updated)

# Define the parameter grid
tuneGrid <- expand.grid(
  .mtry = c(sqrt(ncol(X_train)), log2(ncol(X_train)), ncol(X_train)),
  .ntree = seq(10, 200, length.out = 5),
  .nodesize = seq(1, 20, length.out = 5)
)

# Define training control
trainControl <- trainControl(
  method = "cv",
  number = 5,
  search = "random"
)

# Fit the Random Forest model
rf_model <- train(
  x = X_train,
  y = y_train,
  method = "rf",
  trControl = trainControl,
  tuneLength = 10 # Number of different parameter sets to try
)

# Print the best parameters
print(paste("Best Parameters: ", toString(rf_model$bestTune)))

# Evaluate the model on the test set
predictions <- predict(rf_model, newdata = X_test)
conf_matrix <- confusionMatrix(predictions, y_test)

# Print test accuracy
print(paste("Test Accuracy: ", round(conf_matrix$overall['Accuracy'] * 100, 2), "%"))

# Print classification report
print(conf_matrix)

#-------------------------------------------------------------(XGBOOST Updated)done
tuneGrid <- expand.grid(
  nrounds = seq(50, 500, by = 50),
  max_depth = seq(3, 15, by = 1),
  eta = c(0.01, 0.05, 0.1, 0.2, 0.3),
  gamma = seq(0, 0.5, by = 0.05),
  colsample_bytree = seq(0.4, 1, length.out = 5),
  min_child_weight = seq(1, 10, by = 1),
  subsample = seq(0.4, 1, length.out = 5)
)

# Define training control for random search
train_control <- trainControl(
  method = "cv",
  number = 5,
  allowParallel = TRUE,
  verboseIter = TRUE,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  search = "random"  # Enable random search
)

# Fit the XGBoost model
xgb_model <- train(
  x = as.matrix(X_train),
  y = y_train,
  method = "xgbTree",
  trControl = train_control,
  tuneGrid = tuneGrid,
  tuneLength = 2,  # Number of models to sample and train
  metric = "Accuracy"
)
print("Best Parameters for XGBoost: ")
print(xgb_model$bestTune)

y_pred <- predict(xgb_model, as.matrix(X_test))
y_pred <- factor(y_pred, levels = c("No", "Yes"))
conf_matrix <- confusionMatrix(data = y_pred, reference = y_test)
print(conf_matrix)
#=========================================================(PCA Done)
# Remove columns with zero variance
install.packages("FactoMineR")
library(FactoMineR)

pca <- PCA(X_train, scale.unit = TRUE, ncp = 5)  # ncp is number of dimensions
X_train_pca <- predict(pca, X_train)
X_test_pca <- predict(pca, X_test)

# Extract proportion of variance explained by each PC
variance_explained <- pca$eig / sum(pca$eig)

# Print the proportion of variance explained by each PC
print(variance_explained)

# Access the contribution of variables to each dimension
print("Contribution of variables to each dimension:")
print(pca$var$contrib)

summary(pca)

