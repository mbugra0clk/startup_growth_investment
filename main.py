import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = "C:/ML/startup_growth_investment/startup_growth_investment_data.csv"
df = pd.read_csv(path)
print(df)

# Eksik verileri kontrol etme
missing_values = df.isnull().sum()
missing_values

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Kullanılacak modeller
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Decision Tree": DecisionTreeRegressor(max_depth=5),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "SVR": SVR(kernel='rbf')
}

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Model için giriş değişkenleri (X) ve hedef değişken (y)
X = df[["Funding Rounds", "Investment Amount (USD)", "Number of Investors", "Growth Rate (%)", "Year Founded"]]
y = df["Valuation (USD)"]

# Veriyi eğitim ve test setlerine ayırma (%80 eğitim, %20 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendirme (Bazı modeller daha iyi çalışır)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Lineer regresyon modelini eğitme
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Modelin test seti üzerindeki performansını değerlendirme
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)  # Ortalama mutlak hata
r2 = r2_score(y_test, y_pred)  # R-kare skoru (model başarısı)

mae, r2


# Sonuçları saklamak için bir sözlük
results = {}

# ROC eğrisi için threshold belirleyelim (örneğin, medyan)
threshold = np.median(y_test)
y_test_binary = (y_test >= threshold).astype(int)

plt.figure(figsize=(8, 6))

# Modelleri eğit ve değerlendir
for name, model in models.items():
    # Modeli eğit
    model.fit(X_train_scaled, y_train)

    # Test verisiyle tahmin yap
    y_pred = model.predict(X_test_scaled)

    # Metri̇kleri̇ hesapla
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # ROC eğrisi için tahminleri ikiliye çevir
    fpr, tpr, _ = roc_curve(y_test_binary, y_pred)
    roc_auc = auc(fpr, tpr)

    # Sonuçları kaydet
    results[name] = {"MAE": mae, "R² Score": r2, "AUC": roc_auc}

    # ROC eğrisini çiz
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# Rastgele tahmin çizgisi
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

# ROC eğrisi grafiği
plt.xlabel('False Positive Rate (Yanlış Pozitif Oranı)')
plt.ylabel('True Positive Rate (Doğru Pozitif Oranı)')
plt.title('ROC Eğrisi Karşılaştırması')
plt.legend(loc='lower right')
plt.show()

# Sonuçları göster
import pandas as pd
results_df = pd.DataFrame(results).T
print(results_df)


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Kullanılacak sütunları belirleyelim
target = "Valuation (USD)"
features = ["Industry", "Funding Rounds", "Investment Amount (USD)", "Number of Investors",
            "Country", "Year Founded", "Growth Rate (%)"]

# Giriş ve çıkış değişkenlerini ayır
X = df[features]
y = df[target]

# Kategorik ve sayısal değişkenleri ayır
categorical_features = ["Industry", "Country"]
numerical_features = ["Funding Rounds", "Investment Amount (USD)", "Number of Investors", "Year Founded", "Growth Rate (%)"]

# Ön işleme pipeline'ı (Kategorik değişkenler için One-Hot Encoding, Sayısal değişkenler için Standart Ölçekleme)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi işleyelim
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Ön işlenmiş verinin boyutunu kontrol edelim
X_train_transformed.shape, X_test_transformed.shape

''''''

# Modelleri oluştur
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1)
}

# Sonuçları saklamak için sözlük
results = {}

# Modelleri eğit ve test et
for name, model in models.items():
    # Modeli eğit
    model.fit(X_train_transformed, y_train)

    # Test verisi ile tahmin yap
    y_pred = model.predict(X_test_transformed)

    # Performans metrikleri
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Sonuçları sakla
    results[name] = {"MAE": mae, "R² Score": r2}

# Sonuçları göster
results_df = pd.DataFrame(results).T
results_df


# Yeni girişimin özelliklerini içeren veri
new_startup = pd.DataFrame({
    "Industry": ["Fintech"],
    "Funding Rounds": [5],
    "Investment Amount (USD)": [500000000],  # 500 Milyon Dolar
    "Number of Investors": [30],
    "Country": ["USA"],
    "Year Founded": [2023],
    "Growth Rate (%)": [150]
})

# Yeni girişimi ölçeklendirme ve encoding işlemi
new_startup_transformed = preprocessor.transform(new_startup)

# Ridge Regression modelini kullanarak tahmin yapalım
predicted_valuation = models["Ridge Regression"].predict(new_startup_transformed)[0]
predicted_valuation1 = models["Lasso Regression"].predict(new_startup_transformed)[0]
predicted_valuation2 = models["Linear Regression"].predict(new_startup_transformed)[0]

print(f"Şuanki Tahmini Değerlemesi: ${predicted_valuation:,.2f}")
print(f"Şuanki Tahmini Değerlemesi: ${predicted_valuation1:,.2f}")
print(f"Şuanki Tahmini Değerlemesi: ${predicted_valuation2:,.2f}")

# Varsayılan yıllık büyüme oranı (%30)
growth_rate = 0.30  # %30 büyüme oranı

# Kaç yıl sonraki değerleme?
years = 2

# 2 yıl sonraki tahmini değerleme hesaplama
future_valuation = predicted_valuation * ((1 + growth_rate) ** years)

print(f"5 yıl sonraki tahmini değerleme: ${future_valuation:,.2f}")
