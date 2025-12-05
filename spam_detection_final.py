"""
ТӨСЛИЙН АЖИЛ: ИМЭЙЛ СПАМ ИЛРҮҮЛЭХ СИСТЕМ

ЗОРИЛГО: Имэйл мессежүүдийг спам эсвэл хэвийн гэж ангилах
ЗАГВАРУУД: Naive Bayes, Decision Tree, Logistic Regression

ӨГӨГДЛИЙН ЭХ СУРВАЛЖ:
- Dataset: spam_dataset.csv
- Тайлбар: message_content (имэйлийн текст), is_spam (0=хэвийн, 1=спам)
- Эх сурвалж: Synthetic Email Spam Dataset / Educational Use
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import joblib
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Arial']

print("ИМЭЙЛ СПАМ ИЛРҮҮЛЭХ СИСТЕМ")

# 1. ӨГӨГДӨЛ УНШИЖ АВАХ
print("\n1. Өгөгдөл уншиж байна...")
df = pd.read_csv('spam_dataset.csv')
print(f"   Нийт: {len(df)} мөр, {df.shape[1]} багана")
print(f"   Missing values: {df.isnull().sum().sum()}")
print(f"   Хэвийн: {(df['is_spam']==0).sum()}, Спам: {(df['is_spam']==1).sum()}")

# 2. ТЕКСТ ЦЭВЭРЛЭХ
print("\n2. Текст цэвэрлэж байна...")

def clean_text(text):
    """Текстийг цэвэрлэх: lowercase, тоо/тусгай тэмдэгт устгах"""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())

df['message_clean'] = df['message_content'].apply(clean_text)
print(f"   ✓ Текст цэвэрлэгдсэн")

# 3. ӨГӨГДӨЛ ХУВААХ
print("\n3. Өгөгдөл хуваарилж байна...")
X = df['message_clean']
y = df['is_spam']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Сургалт: {len(X_train)}, Тест: {len(X_test)}")

# 4. TF-IDF ВЕКТОРЖУУЛАЛТ
print("\n4. TF-IDF векторжуулалт хийж байна...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)  # fit зөвхөн сургалт дээр
X_test_tfidf = vectorizer.transform(X_test)
print(f"   ✓ Матриц: {X_train_tfidf.shape}")

# 5. ЗАГВАРУУДЫГ СУРГАХ
print("\n5. Загваруудыг сургаж байна...")
results = {}
models = {}

# 5.1 Naive Bayes
print("   5.1 Naive Bayes...")
train_start = time.time()
nb_grid = GridSearchCV(MultinomialNB(), {'alpha': [0.1, 0.5, 1.0]}, 
                       cv=5, scoring='f1')
nb_grid.fit(X_train_tfidf, y_train)
nb_model = nb_grid.best_estimator_
train_time = time.time() - train_start

pred_start = time.time()
nb_pred = nb_model.predict(X_test_tfidf)
pred_time = time.time() - pred_start

results['Naive Bayes'] = {
    'accuracy': accuracy_score(y_test, nb_pred),
    'precision': precision_score(y_test, nb_pred),
    'recall': recall_score(y_test, nb_pred),
    'f1_score': f1_score(y_test, nb_pred),
    'predictions': nb_pred,
    'train_time': train_time,
    'pred_time': pred_time
}
models['Naive Bayes'] = nb_model
print(f"       Accuracy: {results['Naive Bayes']['accuracy']:.4f}, "
      f"F1: {results['Naive Bayes']['f1_score']:.4f}, "
      f"Хурд: {pred_time*1000:.2f}ms")

# 5.2 Decision Tree
print("   5.2 Decision Tree...")
train_start = time.time()
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), 
                       {'max_depth': [5, 10, 15]}, cv=5, scoring='f1')
dt_grid.fit(X_train_tfidf, y_train)
dt_model = dt_grid.best_estimator_
train_time = time.time() - train_start

pred_start = time.time()
dt_pred = dt_model.predict(X_test_tfidf)
pred_time = time.time() - pred_start

results['Decision Tree'] = {
    'accuracy': accuracy_score(y_test, dt_pred),
    'precision': precision_score(y_test, dt_pred),
    'recall': recall_score(y_test, dt_pred),
    'f1_score': f1_score(y_test, dt_pred),
    'predictions': dt_pred,
    'train_time': train_time,
    'pred_time': pred_time
}
models['Decision Tree'] = dt_model
print(f"       Accuracy: {results['Decision Tree']['accuracy']:.4f}, "
      f"F1: {results['Decision Tree']['f1_score']:.4f}, "
      f"Хурд: {pred_time*1000:.2f}ms")

# 5.3 Logistic Regression
print("   5.3 Logistic Regression...")
train_start = time.time()
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), 
                       {'C': [0.1, 1.0, 10.0]}, cv=5, scoring='f1')
lr_grid.fit(X_train_tfidf, y_train)
lr_model = lr_grid.best_estimator_
train_time = time.time() - train_start

pred_start = time.time()
lr_pred = lr_model.predict(X_test_tfidf)
pred_time = time.time() - pred_start

results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, lr_pred),
    'precision': precision_score(y_test, lr_pred),
    'recall': recall_score(y_test, lr_pred),
    'f1_score': f1_score(y_test, lr_pred),
    'predictions': lr_pred,
    'train_time': train_time,
    'pred_time': pred_time
}
models['Logistic Regression'] = lr_model
print(f"       Accuracy: {results['Logistic Regression']['accuracy']:.4f}, "
      f"F1: {results['Logistic Regression']['f1_score']:.4f}, "
      f"Хурд: {pred_time*1000:.2f}ms")

# 6. ҮР ДҮНГ ХАРЬЦУУЛАХ
print("\n6. Үр дүнг харьцуулж байна...")
results_df = pd.DataFrame(results).T
print("\n" + str(results_df[['accuracy', 'precision', 'recall', 'f1_score']].round(4)))

print("\n  ХУРДНЫ ХАРЬЦУУЛАЛТ:")
print(f"{'Загвар':<20} {'Сургалт (сек)':<15} {'Таамаглал (ms)':<15}")
print("-" * 50)
for model_name in results.keys():
    train_t = results[model_name]['train_time']
    pred_t = results[model_name]['pred_time'] * 1000
    print(f"{model_name:<20} {train_t:<15.3f} {pred_t:<15.2f}")

fastest_model = min(results.keys(), key=lambda x: results[x]['pred_time'])
print(f"\n⚡ Хамгийн хурдан: {fastest_model} "
      f"({results[fastest_model]['pred_time']*1000:.2f}ms)")

best_model_name = results_df['f1_score'].idxmax()
print(f"🏆 Хамгийн сайн F1: {best_model_name} "
      f"(F1={results_df['f1_score'].max():.4f})")

# 7. ДЭЛГЭРЭНГҮЙ ТАЙЛАН
print("\n7. Дэлгэрэнгүй тайлан:")
best_pred = results[best_model_name]['predictions']
print(f"\nClassification Report ({best_model_name}):")
print(classification_report(y_test, best_pred, target_names=['Хэвийн', 'Спам']))

cm = confusion_matrix(y_test, best_pred)
print(f"Confusion Matrix:\n{cm}")
print(f"Зөв таасан: {cm[0][0]+cm[1][1]}/{len(y_test)}")

# 8. ГРАФИК ЗУРАХ
print("\n8. График зурж байна...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Accuracy харьцуулалт
ax1 = axes[0, 0]
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
bars = ax1.bar(model_names, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'], 
               alpha=0.7, edgecolor='black')
ax1.set_ylabel('Accuracy')
ax1.set_title('Загваруудын нарийвчлал', fontweight='bold')
ax1.set_ylim([0.95, 1.0])
ax1.grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom')

# Бүх метрик
ax2 = axes[0, 1]
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
x = np.arange(len(model_names))
width = 0.2
for i, metric in enumerate(metrics):
    values = [results[m][metric] for m in model_names]
    ax2.bar(x + i*width, values, width, label=metric.capitalize(), alpha=0.8)
ax2.set_ylabel('Утга')
ax2.set_title('Бүх метрикүүд', fontweight='bold')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(model_names, rotation=15, ha='right')
ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0.95, 1.0])

# Confusion Matrix
ax3 = axes[1, 0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax3,
            xticklabels=['Хэвийн', 'Спам'], yticklabels=['Хэвийн', 'Спам'])
ax3.set_ylabel('Бодит утга')
ax3.set_xlabel('Таамагласан утга')
ax3.set_title(f'Confusion Matrix ({best_model_name})', fontweight='bold')

# Өгөгдлийн харьцаа
ax4 = axes[1, 1]
spam_counts = df['is_spam'].value_counts()
ax4.pie(spam_counts, labels=['Хэвийн', 'Спам'], 
        colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%', startangle=90)
ax4.set_title('Өгөгдлийн харьцаа', fontweight='bold')

plt.suptitle('Имэйл Спам Илрүүлэх Үр Дүн', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('spam_results.png', dpi=300, bbox_inches='tight')
print("   ✓ График хадгалагдлаа: spam_results.png")

# 9. ЗАГВАРУУДЫГ ХАДГАЛАХ
print("\n9. Загваруудыг хадгалж байна...")
joblib.dump(vectorizer, 'vectorizer.joblib')
for name, model in models.items():
    filename = f"{name.lower().replace(' ', '_')}.joblib"
    joblib.dump(model, filename)
print("   ✓ Бүх загвар хадгалагдлаа")

# 10. ШИНЭ ИМЭЙЛ ТЕСТ
print("\n10. Шинэ имэйл тест хийж байна...")
test_emails = [
    "Hello, meeting reminder for tomorrow at 10 AM.",
    "Congratulations! You won $1000000! Click now!",
    "Please review the attached document.",
    "FREE MONEY!!! Act now! Limited offer!"
]

best_model = models[best_model_name]
for i, email in enumerate(test_emails, 1):
    email_clean = clean_text(email)
    email_tfidf = vectorizer.transform([email_clean])
    prediction = best_model.predict(email_tfidf)[0]
    probability = best_model.predict_proba(email_tfidf)[0]
    
    print(f"\n   Имэйл #{i}: {email[:50]}...")
    print(f"   Таамаглал: {'🚫 СПАМ' if prediction == 1 else '✅ ХЭВИЙН'} "
          f"(Спам: {probability[1]:.1%})")

# ДҮГНЭЛТ
print("\n" + "="*70)
print("ДҮГНЭЛТ")
print("="*70)
print(f"✓ Өгөгдөл: {len(df)} имэйл ({(df['is_spam']==0).sum()} хэвийн, "
      f"{(df['is_spam']==1).sum()} спам)")
print(f"✓ Preprocessing: Текст цэвэрлэлт, TF-IDF векторжуулалт")
print(f"✓ Загварууд: Naive Bayes, Decision Tree, Logistic Regression")
print(f"✓ Hyperparameter tuning: GridSearchCV + 5-fold CV")
print(f"✓ Хамгийн сайн: {best_model_name} (Accuracy: "
      f"{results_df['accuracy'].max()*100:.2f}%)")
print(f"✓ Бүх загвар өндөр нарийвчлалтай ажилласан")
print("\n📁 Файлууд: spam_results.png, vectorizer.joblib, загваруудын .joblib")
print("="*70)
