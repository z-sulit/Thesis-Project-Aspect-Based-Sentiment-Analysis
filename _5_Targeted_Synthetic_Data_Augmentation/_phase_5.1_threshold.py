import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample

# We extend the batches beyond 150 to ensure we actually see where it plateaus
batch_sizes = [50, 100, 150, 200, 300, 400, 500, 600, 700, 800]
f1_scores = []

# Isolate one of your minority classes to map its threshold
target_class = 'Store Operations and Accessibility'

# Create binary targets: 1 if it's the target aspect, 0 otherwise
y_train_binary = (df_train['TargetAspect'] == target_class).astype(int)
y_val_binary = (df_val['TargetAspect'] == target_class).astype(int)

# Use a fast proxy model to map the learning curve
proxy_model = make_pipeline(
    TfidfVectorizer(max_features=5000), 
    LogisticRegression(max_iter=1000, class_weight='balanced')
)

for size in batch_sizes:
    # Stratified sampling ensures both classes exist even in the 50-instance batch
    df_subset = resample(df_train, n_samples=size, stratify=y_train_binary, random_state=42)
    X_subset = df_subset['sentence']
    y_subset = (df_subset['TargetAspect'] == target_class).astype(int)
    
    # Train proxy model and predict against the isolated validation set
    proxy_model.fit(X_subset, y_subset)
    y_pred = proxy_model.predict(df_val['sentence'])
    
    # Track F1 score specifically for the target class
    score = f1_score(y_val_binary, y_pred)
    f1_scores.append(score)


plt.figure(figsize=(9, 5))
plt.plot(batch_sizes, f1_scores, marker='o', linestyle='-', color='b')
plt.title(f'Learning Curve: {target_class}')
plt.xlabel('Number of Authentic Training Instances')
plt.ylabel('F1 Score')
plt.grid(True)
plt.show()