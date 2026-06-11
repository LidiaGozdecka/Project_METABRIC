import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer

# =====================================================
# DANE
# =====================================================

dane3 = pd.read_csv("dane3.csv", sep=";")

with open("eda_variables.pkl", "rb") as f:
    eda_vars = pickle.load(f)

zmienne_do_boxow = eda_vars["zmienne_do_boxow"]
zmienne_genowe = eda_vars["zmienne_genowe"]
zmienne_kliniczne = eda_vars["zmienne_kliniczne"]
kategoryczne_zmienne = eda_vars["kategoryczne_zmienne"]
numeryczne_zmienne = eda_vars["numeryczne_zmienne"]
kolumny_mar = eda_vars["kolumny_mar"]
kolumny_mcar = eda_vars["kolumny_mcar"]
zmienne_do_pca = eda_vars["zmienne_do_pca"]

target = "death_from_cancer"

wynik_ist = pd.read_csv(
    "wynik_po_korekcie.csv",
    sep=";"
)

mar = wynik_ist["missing in:"].unique()

# =====================================================
# CECHY NUMERYCZNE MAR
# =====================================================

mar_num_clin = [
    col for col in mar
    if (
        col in numeryczne_zmienne
        and not col.endswith("_mut")
        and col not in zmienne_do_pca
        and col != "Unnamed: 0"
    )
]

# =====================================================
# CECHY KATEGORYCZNE MAR
# =====================================================

mar_cat_clin = [
    col for col in mar
    if (
        col in kategoryczne_zmienne
        and not col.endswith("_mut")
        and col not in zmienne_do_pca
        and col != target
    )
]

# =====================================================
# CECHY NUMERYCZNE MCAR
# =====================================================

mcar_num_clin = [
    col for col in dane3.columns
    if (
        col not in mar
        and col not in kategoryczne_zmienne
        and not col.endswith("_mut")
        and col not in zmienne_do_pca
    )
]

mcar_num_clin = [
    col for col in mcar_num_clin
    if col not in [
        "patient_id",
        "mahalanobis",
        "therapy_combo",
        "outlier",
        "age_group",
        "death_from_cancer",
        "Unnamed: 0"
    ]
]

mcar_num_clin += ["nottingham_prognostic_index"]

# =====================================================
# CECHY KATEGORYCZNE MCAR
# =====================================================

mcar_cat_clin = [
    col for col in dane3.columns
    if (
        col not in mar
        and col in kategoryczne_zmienne
        and not col.endswith("_mut")
        and col not in zmienne_do_pca
    )
]

mcar_cat_clin = [
    col for col in mcar_cat_clin
    if col not in [
        "patient_id",
        "mahalanobis",
        "chemotherapy",
        "radio_therapy",
        "hormone_therapy",
        "age_group",
        "nottingham_prognostic_index",
        "death_from_cancer",
        "overall_survival"
    ]
]

# =====================================================
# LISTY CECH
# =====================================================

feature_names_clin = (
    mcar_num_clin +
    mar_num_clin +
    mcar_cat_clin +
    mar_cat_clin
)

# =====================================================
# TRAIN / VAL / TEST
# =====================================================

X = dane3[feature_names_clin].copy()
y = dane3[target].copy()
maska=y.notna()
X=X[maska]
y=y[maska]

X_train_clin, X_temp_clin, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    stratify=y,
    random_state=2026
)

X_val_clin, X_test_clin, y_val, y_test = train_test_split(
    X_temp_clin,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=2026
)

# =====================================================
# PREPROCESSING
# =====================================================
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import OrdinalEncoder

encoder=OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1
)
mcar_num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

mar_num_pipe = Pipeline([
    ("imputer", IterativeImputer(
        random_state=2026,
        max_iter=20
    ))
])

mcar_cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ('encoder', encoder)
])

mar_cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ('encoder', encoder)
])

preprocessor_clin = ColumnTransformer(
    transformers=[
        ("mcar_num", mcar_num_pipe, mcar_num_clin),
        ("mar_num", mar_num_pipe, mar_num_clin),
        ("mcar_cat", mcar_cat_pipe, mcar_cat_clin),
        ("mar_cat", mar_cat_pipe, mar_cat_clin)
    ],
    remainder="drop"
)

X_train_clin_final = preprocessor_clin.fit_transform(
    X_train_clin
)

X_val_clin_final = preprocessor_clin.transform(
    X_val_clin
)

X_test_clin_final = preprocessor_clin.transform(
    X_test_clin
)

cols = (
    mcar_num_clin +
    mar_num_clin +
    mcar_cat_clin +
    mar_cat_clin
)

# # =====================================================
# # XGBOOST
# # =====================================================
#
from sklearn.preprocessing import LabelEncoder
# from xgboost import XGBClassifier
# from sklearn.metrics import classification_report
#
le_clin = LabelEncoder()

y_train_clin = le_clin.fit_transform(y_train)
y_val_clin = le_clin.transform(y_val)
y_test_clin = le_clin.transform(y_test)

# xgb_clin = XGBClassifier(
#     objective="multi:softprob",
#     num_class=len(np.unique(y_train_clin)),
#     n_estimators=1500,
#     reg_lambda=10,
#     reg_alpha=4,
#     max_depth=5,
#     learning_rate=0.01,
#     subsample=0.6,
#     colsample_bytree=0.6,
#     random_state=2026,
#     n_jobs=-1,
#     eval_metric="mlogloss",
#     early_stopping_rounds=300
# )
#
# xgb_clin.fit(
#     X_train_clin_final,
#     y_train_clin,
#     eval_set=[
#         (X_train_clin_final, y_train_clin),
#         (X_val_clin_final, y_val_clin)
#     ],
#     verbose=100
# )
#
# # =====================================================
# # PREDYKCJE
# # =====================================================
#
# y_pred_val_clin = xgb_clin.predict(
#     X_val_clin_final
# )
#
# # =====================================================
# # METRYKI
# # =====================================================
#
# classification_report_xgb_clin = classification_report(
#     y_val_clin,
#     y_pred_val_clin,
#     output_dict=True
# )
#
# classification_report_xgb_clin = (
#     pd.DataFrame(classification_report_xgb_clin)
#     .T
# )
#
# classification_report_xgb_clin.to_csv(
#     "metryki_xgb_clin.csv",
#     sep=";"
# )
#
# # =====================================================
# # PARAMETRY
# # =====================================================
#
# parametry_xgb_clin = pd.DataFrame(
#     [xgb_clin.get_params()]
# )
#
# parametry_xgb_clin = parametry_xgb_clin.dropna(
#     axis=1,
#     how="all"
# )
#
# parametry_xgb_clin.to_csv(
#     "parametry_xgb_clin.csv",
#     sep=";"
# )
#
# # =====================================================
# # BEST SCORE
# # =====================================================
#
# best_score_xgb_clin = pd.DataFrame(
#     [xgb_clin.best_score]
# )
#
# best_score_xgb_clin.to_csv(
#     "total_metryki_xgb_clin.csv",
#     sep=";"
# )
#
# # =====================================================
# # FEATURE IMPORTANCE
# # =====================================================
#
# wagi_cech_xgb_clin = pd.DataFrame({
#     "cecha": cols,
#     "wagi": xgb_clin.feature_importances_
# }).sort_values(
#     "wagi",
#     ascending=False
# )
#
# wagi_cech_xgb_clin.to_csv(
#     "wagi_cech_xgb_clin.csv",
#     sep=";"
# )
#
# # =====================================================
# # ZAPIS MODELU
# # =====================================================
#
# with open("xgb_clin.pkl", "wb") as f:
#     pickle.dump(xgb_clin, f)




import pickle
from sklearn.metrics import classification_report
with open('xgb_clin.pkl', 'rb') as f:
    model=pickle.load(f)

# y_pred_test=model.predict(X_test_clin_final)
#
# classification_report_xgb_clin_test=classification_report(y_test_clin, y_pred_test, output_dict=True)
#
# classification_report_xgb_clin_test=pd.DataFrame(classification_report_xgb_clin_test).T
# classification_report_xgb_clin_test.to_csv('metryki_xgb_clin_test.csv', sep=';')



from sklearn.metrics import confusion_matrix
y_pred_val=model.predict(X_val_clin_final)
macierz_val_cat=confusion_matrix(y_val_clin, y_pred_val)
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('AGG')
plt.figure(figsize=(10,4))
k=['Died of Disease', 'Died of other causes', 'Living']
sns.heatmap(macierz_val_cat, annot=True, fmt='d', xticklabels=k,
    yticklabels=k, cbar=False)


plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('mac_val_xg_clin.png')
plt.close()