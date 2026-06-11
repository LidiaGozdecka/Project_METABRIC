import  pandas as pd
from IPython.core.pylabtools import figsize
from PIL.ImageChops import difference
from sklearn.compose import ColumnTransformer

dane3=pd.read_csv('dane3.csv', sep=';')

import pickle
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


#split
#↓
#imputacja na train
#↓
#ICA na train
#↓
#transformacja test
#↓
#model

target='death_from_cancer'

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
wynik_ist=pd.read_csv('wynik_po_korekcie.csv', sep=';')


mar=wynik_ist['missing in:'].unique()
from sklearn.preprocessing import OneHotEncoder
mar_num = [
    col for col in mar
    if not col.endswith('_mut')
    and col in numeryczne_zmienne
]
mar_num=[
    col for col in mar_num
    if col!='Unnamed: 0'
]

mar_cat=[
    col for col in mar
    if col in kategoryczne_zmienne
    or col.endswith('_mut')
]

mcar_num = [
    col for col in dane3.columns
    if not col.endswith('_mut')
    and col not in kategoryczne_zmienne
    and col not in mar
]
mcar_num = [
    col for col in mcar_num
    if col not in [
        'patient_id',
        'mahalanobis',
        'therapy_combo',
        'outlier',
        'age_group',
        'death_from_cancer',
        'Unnamed: 0'

    ]
]

mcar_num=mcar_num +['nottingham_prognostic_index']
mcar_cat=[
    col for col in dane3.columns
    if col not in mar
    and col in kategoryczne_zmienne
]
mcar_cat = [
    col for col in mcar_cat
    if col not in [
        'patient_id',
        'mahalanobis',
        'chemotherapy',
        'radio_therapy',
        'hormone_therapy',
        'age_group',
        'nottingham_prognostic_index',
        'death_from_cancer',
        'overall_survival'
    ]
]




mcar_num = [
    col for col in mcar_num
    if col not in zmienne_do_pca
]
mcar_cat = [
    col for col in mcar_cat
    if col not in zmienne_do_pca
]
mar_num = [
    col for col in mar_num
    if col not in zmienne_do_pca
]
mar_cat = [
    col for col in mar_cat
    if col not in zmienne_do_pca
]
mar_cat=[
    col for col in mar_cat
    if col !=target
]





X=dane3.copy()
X[mcar_cat]=X[mcar_cat].astype('category')
X[mar_cat]=X[mar_cat].astype('category')
X=X.drop(columns=target)
X=X.drop(columns='patient_id')
X=X.drop(columns='overall_survival')
X=X.drop(columns='age_group')
X=X.drop(columns='mahalanobis')
X=X.drop(columns=['chemotherapy',
        'radio_therapy',
        'hormone_therapy'])
X=X.drop(columns='Unnamed: 0')
y=dane3[target]
maska=y.notna()
y=y[maska]
X=X[maska]


X_train_val, X_test, y_train_val, y_test=train_test_split(X,y,test_size=0.2,random_state=2026, stratify=y )
zmienne_do_pca=list(zmienne_do_pca)
X_train, X_val, y_train, y_val=train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=2026, stratify=y_train_val)
X_train_geny = X_train[zmienne_do_pca]
X_val_geny=X_val[zmienne_do_pca]
X_test_geny = X_test[zmienne_do_pca]

X_train_reszta = X_train.drop(columns=zmienne_do_pca)
X_val_reszta=X_val.drop(columns=zmienne_do_pca)
X_test_reszta = X_test.drop(columns=zmienne_do_pca)

from sklearn.pipeline import Pipeline

gen_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

X_train_geny = gen_pipe.fit_transform(X_train_geny)
X_val_geny=gen_pipe.transform(X_val_geny)
X_test_geny = gen_pipe.transform(X_test_geny)
from sklearn.decomposition import FastICA
ica = FastICA(
    n_components=60,
    random_state=2026
)

X_train_ica = ica.fit_transform(X_train_geny)
X_val_ica=ica.transform(X_val_geny)
X_test_ica = ica.transform(X_test_geny)



mcar_num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

mar_num_pipe = Pipeline([
    ("imputer", IterativeImputer(
        random_state=2026,
        max_iter=20
    )),
    ('scaler', StandardScaler())
])

mcar_cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

mar_cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent"))
])


preprocessor = ColumnTransformer(
    transformers=[
        ("mcar_num", mcar_num_pipe, mcar_num),
        ("mar_num", mar_num_pipe, mar_num),
        ("mcar_cat", mcar_cat_pipe, mcar_cat),
        ("mar_cat", mar_cat_pipe, mar_cat)
    ],
    remainder="drop"
)
X_train_reszta=preprocessor.fit_transform(X_train_reszta)
X_val_reszta=preprocessor.transform(X_val_reszta)
X_test_reszta=preprocessor.transform(X_test_reszta)


import numpy as np
X_train_final = np.hstack([
    X_train_reszta,
    X_train_ica
])

X_val_final=np.hstack([
    X_val_reszta,
    X_val_ica
])

X_test_final = np.hstack([
    X_test_reszta,
    X_test_ica
])

ica_cols = [f"IC{i+1}" for i in range(60)]
num=mcar_num+mar_num
cat=mcar_cat+mar_cat


cols = (
    mcar_num +
    mar_num +
    mcar_cat +
    mar_cat+
    ica_cols
)





cat_features = list(
     range(
         len(mar_num)+len(mcar_num),
         len(mar_num)+len(mcar_num)+ len(cat)
     )
 )


for idx in cat_features:
     X_train_final[:, idx] = X_train_final[:, idx].astype(str)
     X_val_final[:, idx]=X_val_final[:,idx].astype(str)
     X_test_final[:, idx] = X_test_final[:, idx].astype(str)


from catboost import CatBoostClassifier


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
# model.fit(
#     X_train_final,
#     y_train,
#     cat_features=cat_features,
#     eval_set=(X_val_final, y_val)
# )
#
# imp=list(model.get_feature_importance())
#
# feat=num+cat
# import pandas as pd
# wagi_cech=pd.DataFrame({
#    'wagi':imp,
#    'cechy':feat
# }).sort_values('wagi', ascending=False)
#
#
#  #model.save_model("catboost_model1.cbm")
#
#
# results=model.get_evals_result()
# #
# #
# from sklearn.metrics import classification_report
# #
# y_pred_val = model.predict(X_val_final)
# #
# classification_report_test1 = classification_report(
#     y_val,
#     y_pred_val,
#     output_dict=True
# )
#
# classification_report_test1 = pd.DataFrame(
#     classification_report_test1
# ).T
# classification_report_test1.to_csv('metryki_walidcja1.csv', sep=';')
#
# print(print(model.get_best_iteration()))
# best_score=model.get_best_score()
# best_score=pd.DataFrame(best_score)
#
# params=model.get_params()
# params=pd.DataFrame(params)




















# drugi model ten sam

# from catboost import CatBoostClassifier
from catboost import Pool
#
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
#     X_train_final,
#     y_train,
#     feature_names=mcar_num+mar_num + mcar_cat+mar_cat+ica_cols,
#     cat_features=cat_features
#
# )
val_pool = Pool(
     X_val_final,
     y_val,
     feature_names=mcar_num+mar_num + mcar_cat+mar_cat+ica_cols,
     cat_features=cat_features
)
# model.fit(
#     train_pool,
#     eval_set=val_pool,
#     use_best_model=True
# )
# imp2=list(model.get_feature_importance())
#
# feat=mcar_num+mar_num+mcar_cat+mar_cat+ica_cols
# import pandas as pd
# wagi_cech2=pd.DataFrame({
#     'wagi':imp2,
#     'cechy':feat
# }).sort_values('wagi', ascending=False)
#
# wagi_cech2.to_csv('wagi_catboost2.csv', sep=';')
# model.save_model("catboost_model2.cbm")
#
#
# results2=model.get_evals_result()
#
#
#
# dane = {}
#
# for zbior, metryki in results2.items():
#     prefix = 'train' if zbior == 'learn' else 'val'
#
#     for nazwa, wartosci in metryki.items():
#         dane[f'{prefix}_{nazwa}'] = wartosci
#
# wyniki_model2 = pd.DataFrame(dane)
#
# wyniki_model2.to_csv('wyniki_model2.csv', sep=';')
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('AGG')
#
# plt.figure(figsize=(12,6))
#
# for col in wyniki_model2.columns:
#     plt.plot(
#         wyniki_model2.index,
#         wyniki_model2[col],
#         label=col
#     )
#
# plt.xlabel("Liczba drzew")
# plt.ylabel("Wartość metryki")
# plt.title("Metryki podczas uczenia")
# plt.legend()
# plt.grid(True)
# plt.savefig('Metryki_over_drzewa2.png')
# plt.close()
# from sklearn.metrics import classification_report
# #
# y_pred_val2 = model.predict(X_val_final)
# #
# classification_report_test2 = classification_report(
#     y_val,
#     y_pred_val2,
#     output_dict=True
# )
#
# classification_report_test2 = pd.DataFrame(
#     classification_report_test2
# ).T
#
# classification_report_test2.to_csv('metryki_walidcja2.csv', sep=';')
#
# #print(print(model.get_best_iteration()))
# best_score2=model.get_best_score()
# best_score2=pd.DataFrame(best_score2)
# best_score2.to_csv('total_metryki2.csv', sep=';')
#
# params2=model.get_params()
# params2=pd.DataFrame(params2)
# params2.to_csv('parametry2.csv', sep=';')
#
# import os
# os.environ["PATH"] += r";C:\Program Files\Graphviz\bin"
# graph2 = model.plot_tree(tree_idx=0, pool=train_pool)
#
# graph2.render(
#     filename="cat_tree1",
#     format="png",
#     cleanup=True
# )
from catboost import CatBoostClassifier
from catboost import Pool

test_pool = Pool(
    X_test_final,
    cat_features=cat_features
)

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
model=CatBoostClassifier()
model.load_model('catboost_model2.cbm')
#y_pred_test=model.predict(test_pool)
# classification_report_c1_test=classification_report(
#     y_test,
#     y_pred_test,
#     output_dict=True
# )
# classification_report_c1_test=pd.DataFrame(classification_report_c1_test).T
# classification_report_c1_test.to_csv('metryki_c_test.csv', sep=';')
y_pred_val=model.predict(val_pool)
macierz_val_cat=confusion_matrix(y_val, y_pred_val)
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.use('AGG')
plt.figure(figsize=(10,4))
sns.heatmap(macierz_val_cat, annot=True, fmt='d', xticklabels=np.unique(y_val),
    yticklabels=np.unique(y_val), cbar=False)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('mac_val_cat.png')
plt.close()