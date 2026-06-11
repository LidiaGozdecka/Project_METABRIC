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

preprocessor_clin = ColumnTransformer(
    transformers=[
        (
            "mar_num",
            IterativeImputer(random_state=2026),
            mar_num_clin
        ),
        (
            "mcar_num",
            StandardScaler(),
            mcar_num_clin
        ),
        (
            "mar_cat",
            SimpleImputer(strategy="most_frequent"),
            mar_cat_clin
        ),
        (
            "mcar_cat",
            "passthrough",
            mcar_cat_clin
        )
    ],
    remainder="drop",
    verbose_feature_names_out=False
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

# =====================================================
# CATBOOST
# =====================================================

num_clin = mcar_num_clin + mar_num_clin
cat_clin = mcar_cat_clin + mar_cat_clin

cat_features_clin = list(
    range(
        len(num_clin),
        len(num_clin) + len(cat_clin)
    )
)

for idx in cat_features_clin:
    X_train_clin_final[:, idx] = (
        X_train_clin_final[:, idx].astype(str)
    )

    X_val_clin_final[:, idx] = (
        X_val_clin_final[:, idx].astype(str)
    )

    X_test_clin_final[:, idx] = (
        X_test_clin_final[:, idx].astype(str)
    )

print(X_train_clin_final.shape)
print(X_val_clin_final.shape)
print(X_test_clin_final.shape)

print("Numeryczne:", len(num_clin))
print("Kategoryczne:", len(cat_clin))
print("Razem:", len(feature_names_clin))


from catboost import CatBoostClassifier
from catboost import Pool

# model = CatBoostClassifier(
#      loss_function="MultiClass",
#      eval_metric="TotalF1",
#      custom_metric=[
#          "Accuracy"
#      ],
#      iterations=1000,
#      depth=4,
#      learning_rate=0.03,
#      l2_leaf_reg=10,
#      random_strength=5,
#      min_data_in_leaf=15,
#      border_count=20,
#      random_seed=2026,
#      verbose=100
#  )
#
# train_pool=Pool(
#     X_train_clin_final,
#     y_train,
#     feature_names=feature_names_clin,
#     cat_features=cat_features_clin
#
# )
val_pool = Pool(
     X_val_clin_final,
     y_val,
     feature_names=feature_names_clin,
     cat_features=cat_features_clin
)
# model.fit(
#     train_pool,
#     eval_set=val_pool,
#     use_best_model=True
# )
# imp_clin=list(model.get_feature_importance())
#
# feat_clin=feature_names_clin
# import pandas as pd
# wagi_cech_clin=pd.DataFrame({
#     'wagi':imp_clin,
#     'cechy':feat_clin
# }).sort_values('wagi', ascending=False)
#
# wagi_cech_clin.to_csv('wagi_catboost_clin.csv', sep=';')
# model.save_model("catboost_model_clin.cbm")
#
#
# results_clin=model.get_evals_result()
#
#
#
# dane = {}
#
# for zbior, metryki in results_clin.items():
#     prefix = 'train' if zbior == 'learn' else 'val'
#
#     for nazwa, wartosci in metryki.items():
#         dane[f'{prefix}_{nazwa}'] = wartosci
#
# wyniki_model_clin = pd.DataFrame(dane)
#
# wyniki_model_clin.to_csv('wyniki_model_clin.csv', sep=';')
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('AGG')
#
# plt.figure(figsize=(12,6))
#
# for col in wyniki_model_clin.columns:
#     plt.plot(
#         wyniki_model_clin.index,
#         wyniki_model_clin[col],
#         label=col
#     )
#
# plt.xlabel("Liczba drzew")
# plt.ylabel("Wartość metryki")
# plt.title("Metryki podczas uczenia")
# plt.legend()
# plt.grid(True)
# plt.savefig('Metryki_over_drzewa_clin.png')
# plt.close()
# from sklearn.metrics import classification_report
# #
# y_pred_val2 = model.predict(val_pool)
#
# classification_report_test_clin = classification_report(
#      y_val,
#      y_pred_val2,
#      output_dict=True
# )
#
# classification_report_test_clin = pd.DataFrame(
#     classification_report_test_clin
# ).T
#
# classification_report_test_clin.to_csv('metryki_walidcja_clin.csv', sep=';')
#
# # #print(print(model.get_best_iteration()))
# best_score_clin=model.get_best_score()
# best_score_clin=pd.DataFrame(best_score_clin)
# best_score_clin.to_csv('total_metryki_clin.csv', sep=';')
#
# params_clin=model.get_params()
# params_clin=pd.DataFrame(params_clin)
#
#
# import os
# os.environ["PATH"] += r";C:\Program Files\Graphviz\bin"
# graph_clin = model.plot_tree(tree_idx=0, pool=train_pool)
#
# graph_clin.render(
#     filename="cat_tree_clin",
#     format="png",
#     cleanup=True
# )
from catboost import CatBoostClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use("AGG")

# =====================================================
# WCZYTANIE MODELU
# =====================================================

model = CatBoostClassifier()
model.load_model("catboost_model_clin.cbm")
print(model.is_fitted())
print(model.get_best_iteration())
print(model.get_best_score())
# =====================================================
# PREDYKCJE
# =====================================================

y_pred_val_clin = model.predict(val_pool)
y_pred_val_clin = np.ravel(y_pred_val_clin)

# =====================================================
# MACIERZ POMYŁEK
# =====================================================

macierz_val_cat_clin = confusion_matrix(
    y_val,
    y_pred_val_clin
)

plt.figure(figsize=(10, 4))

sns.heatmap(
    macierz_val_cat_clin,
    annot=True,
    fmt="d",
    xticklabels=np.unique(y_val),
    yticklabels=np.unique(y_val),
    cbar=False
)

plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("mac_val_cat_clin.png")
plt.close()

# =====================================================
# METRYKI
# =====================================================

classification_report_clin = classification_report(
    y_val,
    y_pred_val_clin,
    output_dict=True
)

classification_report_clin = (
    pd.DataFrame(classification_report_clin)
    .T
)

classification_report_clin.to_csv(
    "metryki_walidcja_clin.csv",
    sep=";"
)

# =====================================================
# BEST SCORE ZAPISANY W MODELU
# =====================================================
best_score_clin=model.get_best_score()
best_score_clin = pd.DataFrame(
    best_score_clin
)

best_score_clin.to_csv(
    "total_metryki_clin.csv",
    sep=";"
)