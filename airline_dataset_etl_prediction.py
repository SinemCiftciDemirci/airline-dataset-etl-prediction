import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import time

# DeprecationWarnings'ları filtrele
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Analiz başlangıç zamanı
start_time = time.time()
print(f"Analiz başlangıç zamanı: {time.ctime(start_time)}\n")

# Veri setini yükle
data = pd.read_excel("Airline Dataset Updated - v2.xlsx")

# Departure Date sütununu datetime formatına dönüştür
data['Departure Date'] = pd.to_datetime(data['Departure Date'], errors='coerce')


# Verilerin toplam boyutunu kontrol et
print(f"Total instances in entire dataset: {len(data)}\n")

# Yıllık veriyi hesapla
year_data = data.groupby(['Airport Name', 'Flight Status']).size().reset_index(name='Count')

# Yaş için istatistiksel özet
nums = [0, 15, 30, 50, 70, 120]  # Define the age groups
gap = ['0-15', '16-30', '31-50', '51-70', '71+']  # Define group names
data['ages'] = pd.cut(data['Age'], bins=nums, labels=gap)  # Create a new age group column
age_summary = data['ages'].value_counts()

# Cinsiyet (Gender) için istatistiksel özet
gender_summary = data['Gender'].value_counts()

# Nationality için istatistiksel özet
nationality_summary = data['Nationality'].value_counts().nlargest(20)

# Ülke (Country Name) için istatistiksel özet
country_summary = data['Country Name'].value_counts().nlargest(20)

# Her bir Flight Status için en yüksek sayıya sahip havaalanlarını bul
top_airports = year_data.groupby('Flight Status', as_index=False).apply(lambda x: x.nlargest(5, 'Count')).reset_index(drop=True)

# Yaş (Age) grafiği
age_gaps = age_summary.sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=age_gaps.index, y=age_gaps.values, hue=age_gaps.index, palette="Spectral", legend=False)
plt.title("Travellers based on their age")
plt.xlabel("Age Group")
plt.ylabel("Count")
plt.show()

# Cinsiyet (Gender) grafiği
plt.figure(figsize=(8, 6))
gender_summary.plot(kind='bar', color=['turquoise', 'violet'])
plt.title('Travelers by Genders')
plt.xlabel('Gender')
plt.ylabel('Passenger Count')
plt.show()

# Ulus (Nationality) grafiği
plt.figure(figsize=(12, 8))
nationality_summary.plot(kind='bar')
plt.title('Travelers by Nationality')
plt.xlabel('Nationality')
plt.ylabel('Passenger Count')
plt.show()

# Ülke (Country Name) grafiği
plt.figure(figsize=(12, 8))
country_summary.plot(kind='bar')
plt.title('Travelers by Country Name')
plt.xlabel('Country Name')
plt.ylabel('Passenger Count')
plt.show()

# En çok gecikmiş uçuşları görselleştir
plt.figure(figsize=(10, 6))
plt.bar(top_airports[top_airports['Flight Status'] == 'Delayed']['Airport Name'], top_airports[top_airports['Flight Status'] == 'Delayed']['Count'], color='red')
plt.xlabel('Airport Name')
plt.ylabel('Number of Delayed Flights')
plt.title('Top Airports with the Most Delayed Flights')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# En çok zamanında uçuşları görselleştir
plt.figure(figsize=(10, 6))
plt.bar(top_airports[top_airports['Flight Status'] == 'On Time']['Airport Name'], top_airports[top_airports['Flight Status'] == 'On Time']['Count'], color='green')
plt.xlabel('Airport Name')
plt.ylabel('Number of On Time Flights')
plt.title('Top Airports with the Most On Time Flights')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# En çok iptal edilmiş uçuşları görselleştir
plt.figure(figsize=(10, 6))
plt.bar(top_airports[top_airports['Flight Status'] == 'Cancelled']['Airport Name'], top_airports[top_airports['Flight Status'] == 'Cancelled']['Count'], color='blue')
plt.xlabel('Airport Name')
plt.ylabel('Number of Cancelled Flights')
plt.title('Top Airports with the Most Cancelled Flights')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Hedef sütunu etiketle
label_encoder = LabelEncoder()
data['Flight Status'] = label_encoder.fit_transform(data['Flight Status'])

# Cancelled & Delayed ve On Time verilerini etiketle
on_time_data = data[data['Flight Status'] == 0]
delayed_data = data[data['Flight Status'] == 1]
cancelled_data = data[data['Flight Status'] == 1]

first_half_data = pd.concat([cancelled_data[cancelled_data['Departure Date'].dt.month <= 6], 
                             on_time_data[on_time_data['Departure Date'].dt.month <= 6], 
                             delayed_data[delayed_data['Departure Date'].dt.month <= 6]])

second_half_data = pd.concat([cancelled_data[cancelled_data['Departure Date'].dt.month > 6], 
                              on_time_data[on_time_data['Departure Date'].dt.month > 6], 
                              delayed_data[delayed_data['Departure Date'].dt.month > 6]])

# Verilerin boyutunu kontrol et
print(f"Total instances in first half data: {len(first_half_data)}")
print(f"Total instances in second half data: {len(second_half_data)}\n")

# Nümerik ve kategorik verileri seç
numerical_features = ['Age']
categorical_features = ['Gender', 'Nationality', 'Airport Name', 'Airport Country Code', 'Country Name', 
                        'Airport Continent', 'Continents', 'Arrival Airport', 'Pilot Name']

# Gereksiz kategorik özellikleri çıkarmak için kullanmadığınız özellikleri çıkarın
selected_categorical_features = ['Airport Name', 'Arrival Airport', 'Pilot Name']

# Kategorik verileri one-hot encode yap
first_half_data = pd.get_dummies(first_half_data, columns=selected_categorical_features)

# Model için özellikler ve hedef değişkeni ayır
feature_columns = list(first_half_data.columns[first_half_data.columns.str.contains('|'.join(selected_categorical_features))])
X = first_half_data[feature_columns]
y = first_half_data['Flight Status']

# Veri setini eğitim ve test kümelerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Total instances in train set: {len(y_train)}")
print(f"Total instances in test set: {len(y_test)}\n")
print(f"Total instances in X: {len(X)}")
print(f"Total instances in y: {len(y)}\n")

# Sınıflandırıcıları tanımla
classifiers = {
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=2000),
    'KNN': KNeighborsClassifier(),
    'ANN': MLPClassifier(max_iter=2000),
    'Decision Tree': DecisionTreeClassifier(),
}

# Değerleri saklamak için boş listeler oluşturalım
models = []
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []

# Her bir sınıflandırıcı için modeli eğit ve test verisi üzerinde değerlendir
for name, clf in classifiers.items():
    model_start_time = time.time()  # Model eğitimi ve değerlendirmesi başlangıç zamanı

    # Modeli eğit
    clf.fit(X_train, y_train)
    
    # Modeli kullanarak tahmin yap
    y_pred = clf.predict(X_test)

    # Metrikleri hesapla
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Değerleri listelere ekle
    models.append(name)
    accuracy_list.append(acc)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)

    # Confusion matrix'teki toplam instance sayısı
    total_instances = conf_matrix.sum()

    # Model eğitim ve değerlendirme süresini hesapla
    model_end_time = time.time()  # Model eğitimi ve değerlendirmesi bitiş zamanı
    model_elapsed_time = (model_end_time - model_start_time) / 60  # Dakika cinsinden model eğitim ve değerlendirme süresi

    print(f"Model: {name}\n")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall: {rec:.3f}")
    print(f"F1 Score: {f1:.3f}\n")
    print(f"Confusion Matrix:\n{conf_matrix}\n")
    print(f"Total Instances: {total_instances}\n")
    print(f"Model Training and Evaluation Time: {model_elapsed_time:.2f} minutes\n")

# Grafik oluşturma
plt.figure(figsize=(12, 6))

# Accuracy
plt.subplot(2, 2, 1)
plt.bar(models, accuracy_list, color='blue', alpha=0.5)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Models')
plt.xticks(rotation=45)

# Precision
plt.subplot(2, 2, 2)
plt.bar(models, precision_list, color='red', alpha=0.5)
plt.xlabel('Models')
plt.ylabel('Precision')
plt.title('Precision of Models')
plt.xticks(rotation=45)

# Recall
plt.subplot(2, 2, 3)
plt.bar(models, recall_list, color='green', alpha=0.5)
plt.xlabel('Models')
plt.ylabel('Recall')
plt.title('Recall of Models')
plt.xticks(rotation=45)

# F1 Score
plt.subplot(2, 2, 4)
plt.bar(models, f1_list, color='purple', alpha=0.5)
plt.xlabel('Models')
plt.ylabel('F1 Score')
plt.title('F1 Score of Models')
plt.xticks(rotation=45)

# Analiz bitiş zamanı ve toplam süre
end_time = time.time()
elapsed_time = (end_time - start_time)/60

print(f"Analiz başlangıç zamanı: {time.ctime(start_time)}")
print(f"Analiz bitiş zamanı: {time.ctime(end_time)}")
print(f"Toplam analiz süresi: {elapsed_time:.2f} dakika")

plt.tight_layout()
plt.show()
