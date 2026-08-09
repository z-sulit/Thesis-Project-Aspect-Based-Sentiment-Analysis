import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import emoji
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('/Dataset/Thesis_master_reviews_combined.csv')

# Standardize identifiers to prevent case-mismatch grouping errors
df['shop_type'] = df['shop_type'].astype(str).str.lower()
df['place_name'] = df['place_name'].astype(str).str.lower()

# 1. Key Dataset Overview & 2. Establishment Volume & Rating Visualizations
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(data=df, x='shop_type', ax=axes[0])
axes[0].set_title('Review Volume by Shop Type')

sns.histplot(data=df, x='rating', hue='shop_type', multiple='dodge', bins=5, ax=axes[1])
axes[1].set_title('Rating Distribution')
plt.show()

# 3. Textual Profiling
df['word_count'] = df['clean_review'].str.split().str.len()
print("=== Textual Profiling ===")
print(df.groupby('shop_type')['word_count'].describe())

# 4. Vocabulary & N-Gram Analysis
def get_top_ngrams(corpus, n=2, top_k=10):
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_k]

print("\n=== Top Bigrams (Coffee) ===")
print(get_top_ngrams(df[df['shop_type'] == 'coffee']['clean_review'].dropna(), n=2))
print("\n=== Top Bigrams (Matcha) ===")
print(get_top_ngrams(df[df['shop_type'] == 'matcha']['clean_review'].dropna(), n=2))

def calculate_ttr(text_series):
    full_text = " ".join(text_series.dropna())
    tokens = full_text.split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

print("\n=== Type-Token Ratio (TTR) ===")
print(f"Coffee TTR: {calculate_ttr(df[df['shop_type'] == 'coffee']['clean_review']):.4f}")
print(f"Matcha TTR: {calculate_ttr(df[df['shop_type'] == 'matcha']['clean_review']):.4f}")

# 5. Code-Switching Density (Hybrid Regex Implementation)
# Strict exact matches for short words to prevent false positives (e.g., 'jud', 'pud', 'man')
strict_markers = ['jud', 'pud', 'sad', 'man', 'na', 'pa', 'din', 'rin']
strict_pattern = r'\b(' + '|'.join(strict_markers) + r')\b'

# Partial matches for root words to catch affixes (e.g., 'lami' catches 'kalami', 'lamian')
root_markers = ['lami', 'kaayo', 'sarap', 'mahal', 'init', 'tugnaw', 'balik', 'grabe', 'mura', 'pangit', 'bati']
root_pattern = r'(' + '|'.join(root_markers) + r')'

combined_pattern = f"({strict_pattern}|{root_pattern})"
df['code_switched'] = df['clean_review'].str.contains(combined_pattern, case=False, na=False)

print("\n=== Code-Switching Density ===")
print(df.groupby('shop_type')['code_switched'].mean() * 100)

# 6. Emoji Frequency Analysis
def extract_emojis(text):
    if pd.isna(text): return []
    return [res['emoji'] for res in emoji.emoji_list(text)]

all_emojis = [e for text in df['display_review'] for e in extract_emojis(text)]
emoji_counts = Counter(all_emojis)

print("\n=== Top 10 Emojis ===")
for em, count in emoji_counts.most_common(10):
    print(f"{em} ({emoji.demojize(em)}): {count}")