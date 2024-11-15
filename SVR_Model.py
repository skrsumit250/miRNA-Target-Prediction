import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np

# Function to perform k-mer encoding
def kmer_encoding(sequences, k=3, *, vectorizer=None):
    seqs = [" ".join([seq[i:i+k] for i in range(len(seq)-k+1)]) for seq in sequences]
    if vectorizer is None:
        vectorizer = CountVectorizer()
        kmer_features = vectorizer.fit_transform(seqs)
    else:
        kmer_features = vectorizer.transform(seqs)
    return pd.DataFrame(kmer_features.toarray(), columns=vectorizer.get_feature_names_out())

# Load data
dataA = pd.read_csv("Train.csv")
dataB = pd.read_csv("Test.csv")

# Encode the miRNA_Sequence and gene_Sequence for training data
vectorizer_mirna = CountVectorizer()
vectorizer_gene = CountVectorizer()

vectorizer_mirna.fit(dataA['miRNA_Sequence'])
vectorizer_gene.fit(dataA['gene_Sequence'])

X_train_mirna = kmer_encoding(dataA['miRNA_Sequence'], k=3, vectorizer=vectorizer_mirna)
X_test_mirna = kmer_encoding(dataB['miRNA_Sequence'], k=3, vectorizer=vectorizer_mirna)

X_train_gene = kmer_encoding(dataA['gene_Sequence'], k=3, vectorizer=vectorizer_gene)
X_test_gene = kmer_encoding(dataB['gene_Sequence'], k=3, vectorizer=vectorizer_gene)

# Normalize all features
scaler = StandardScaler()
X_train_mirna = scaler.fit_transform(X_train_mirna)
X_test_mirna = scaler.transform(X_test_mirna)
X_train_gene = scaler.fit_transform(X_train_gene)
X_test_gene = scaler.transform(X_test_gene)
X_train_length = scaler.fit_transform(dataA[['gene_length']])
X_test_length = scaler.transform(dataB[['gene_length']])

# Concatenate all features for training data
X_train = np.hstack((X_train_mirna, X_train_gene, X_train_length))
X_test = np.hstack((X_test_mirna, X_test_gene, X_test_length))

# Prepare the target
y_train = dataA['Target_Score']

# Create SVM regressor
svm_model = SVR()

# Hyperparameter tuning using Grid Search
param_grid = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto']
}
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best parameters
print("Best parameters found: ", grid_search.best_params_)

# Train the model with best parameters
best_svm_model = grid_search.best_estimator_
best_svm_model.fit(X_train, y_train)
print("Model Fitted with Best Parameters")

# Predict target scores for dataB
y_pred = best_svm_model.predict(X_test)
print("Predicted Target Scores for dataB:", y_pred)

# Evaluate model performance on the training set
y_train_pred = best_svm_model.predict(X_train)
mse = mean_squared_error(y_train, y_train_pred)
print("Mean Squared Error (Training Set):", mse)

# Custom accuracy calculation
threshold = 0.1  # Define a threshold for acceptable error
accuracy = np.mean(np.abs(y_train - y_train_pred) < threshold)
print("Custom Training Accuracy:", accuracy)
