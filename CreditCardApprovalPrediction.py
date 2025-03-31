
#imports
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import RobustScaler
from scipy.stats import shapiro


#Settings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#functions

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T.round(2))

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def base_models(X, y, scoring="roc_auc"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   #('KNN', KNeighborsClassifier()),
                   #("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   #('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(eval_metric='logloss', enable_categorical=True)),
                   ('LightGBM', LGBMClassifier()),
                   #('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def calculate_ratio(group):
    total_count = len(group)
    x_count = (group == 'X').sum()
    c_count = (group == 'C').sum()
    zero_count = (group == '0').sum()
    one_count = (group == '1').sum()
    two_count = (group == '2').sum()
    three_count = (group == '3').sum()
    four_count = (group == '4').sum()
    five_count = (group == '5').sum()
    ratio_x = x_count / total_count
    ratio_c = c_count / total_count
    ratio_zero = zero_count / total_count
    ratio_one = one_count / total_count
    ratio_two = two_count / total_count
    ratio_three = three_count / total_count
    ratio_four = four_count / total_count
    ratio_five = five_count / total_count
    return pd.Series({'X_RATIO': ratio_x, 'C_RATIO': ratio_c, 'ZERO_RATIO': ratio_zero, 'ONE_RATIO': ratio_one,'TWO_RATIO': ratio_two, 'THREE_RATIO': ratio_three, 'FOUR_RATIO': ratio_four, 'FIVE_RATIO': ratio_five,})

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show(block=True)

def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

#Read to data

df_app = pd.read_csv("projectdatasets/application_record.csv")
df_cre = pd.read_csv("projectdatasets/credit_record.csv")

#EDA

check_df(df_app)
check_df(df_cre)

df_app.head()
df_cre.head()

toplam_unique_degerler = df_cre["ID"].nunique()
print("Toplam unique değer sayısı:", toplam_unique_degerler)

df_cre = df_cre[df_cre['STATUS'] != 'X']

# Feature Engineering

status_ratios = df_cre.pivot_table(index='ID', columns='STATUS', values='STATUS', aggfunc='count', fill_value=0)

status_ratios['C_RATIO'] = status_ratios['C'] / (status_ratios['C'] + status_ratios['0'] + status_ratios['1'] + status_ratios['2'] + status_ratios['3'] + status_ratios['4'] + status_ratios['5'])
status_ratios['ZERO_RATIO'] = status_ratios['0'] / (status_ratios['C'] + status_ratios['0'] + status_ratios['1'] + status_ratios['2'] + status_ratios['3'] + status_ratios['4'] + status_ratios['5'])
status_ratios['ONE_RATIO'] = status_ratios['1'] / (status_ratios['C'] + status_ratios['0'] + status_ratios['1'] + status_ratios['2'] + status_ratios['3'] + status_ratios['4'] + status_ratios['5'])
status_ratios['TWO_RATIO'] = status_ratios['2'] / (status_ratios['C'] + status_ratios['0'] + status_ratios['1'] + status_ratios['2'] + status_ratios['3'] + status_ratios['4'] + status_ratios['5'])
status_ratios['THREE_RATIO'] = status_ratios['3'] / (status_ratios['C'] + status_ratios['0'] + status_ratios['1'] + status_ratios['2'] + status_ratios['3'] + status_ratios['4'] + status_ratios['5'])
status_ratios['FOUR_RATIO'] = status_ratios['4'] / (status_ratios['C'] + status_ratios['0'] + status_ratios['1'] + status_ratios['2'] + status_ratios['3'] + status_ratios['4'] + status_ratios['5'])
status_ratios['FIVE_RATIO'] = status_ratios['5'] / (status_ratios['C'] + status_ratios['0'] + status_ratios['1'] + status_ratios['2'] + status_ratios['3'] + status_ratios['4'] + status_ratios['5'])
status_ratios['TOTAL_RATIO'] = status_ratios['C_RATIO'] + status_ratios['ZERO_RATIO'] + status_ratios['ONE_RATIO'] + status_ratios['TWO_RATIO'] + status_ratios['THREE_RATIO'] + status_ratios['FOUR_RATIO'] + status_ratios['FIVE_RATIO']
status_ratios['CUSTOMER_TYPE_RATIO'] = (status_ratios['C_RATIO'] + status_ratios['ZERO_RATIO']) / (status_ratios['C_RATIO'] + status_ratios['ZERO_RATIO'] + status_ratios['ONE_RATIO'] + status_ratios['TWO_RATIO'] + status_ratios['THREE_RATIO'] + status_ratios['FOUR_RATIO'] + status_ratios['FIVE_RATIO'])

len(status_ratios[status_ratios['C_RATIO'] >= 0.85])          # C + 0 rasyosu %90 üzeri olanları risksiz müşteri olarak tanımlıyorum. adet: 26680


status_ratios['CUSTOMER_TYPE'] = status_ratios['C_RATIO'] < 0.85
status_ratios['CUSTOMER_TYPE'] = status_ratios['CUSTOMER_TYPE'].astype(int)

status_ratios[status_ratios['CUSTOMER_TYPE'] == 1]['CUSTOMER_TYPE'].count()           # 35702 riskli olanlar "1"
status_ratios[status_ratios['CUSTOMER_TYPE'] == 0]['CUSTOMER_TYPE'].count()           # 5747 risksiz olanlar "0"

status_ratios["ID"] = status_ratios.index       #ID için pivot tablo yaptığımdan ID satırını kullanabilmek için kopya bir kolon oluşturdum.

df_app = df_app.drop_duplicates(subset='ID', keep='first')      # Burada df_app içindeki tekrarlanan ID bilgilerini sildim.

df_app['CUSTOMER_TYPE_MATCHED'] = df_app['ID'].map(status_ratios.set_index('ID')['CUSTOMER_TYPE'])  #başvuru datasına eşleşen IDler için müşteri tiplerini girdim.

df = df_app.dropna()  #NaN değerleri sildim
df.reset_index()      #index bilgisini düzelttim

df[df['CUSTOMER_TYPE_MATCHED'] == 1]['CUSTOMER_TYPE_MATCHED'].count()           # 19799 riskli olanlar "1"
df[df['CUSTOMER_TYPE_MATCHED'] == 0]['CUSTOMER_TYPE_MATCHED'].count()           #  2938 risksiz olanlar "0"

# Encoding

    # Doğduğu zamandan bu yana geçen gün sayısını yaş olarak düzenledim.
df["AGE"] = df["DAYS_BIRTH"] / 365 * -1
df["AGE"] = df["AGE"].round()

    # işe başladığı zamandan bu yana geçen gün sayısını ay olarak düzenledim.
df["MONTHS_EMPLOYED"] = df["DAYS_EMPLOYED"] / 30 * -1
df["MONTHS_EMPLOYED"] = df["MONTHS_EMPLOYED"].round()

df.drop("DAYS_EMPLOYED", axis=1, inplace=True) # eski gün esası doğum tarihi ve işe başlangıç zamanlarını sildim.
df.drop("DAYS_BIRTH", axis=1, inplace=True)

    # evi arabası var ise 1 yok ise 0.
df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].apply(lambda x: 1 if x == 'Y' else 0)
df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].apply(lambda x: 1 if x == 'Y' else 0)

    # Cinsiyet için 0 erkek 1 kadın.
df['CODE_GENDER'] = df['CODE_GENDER'].apply(lambda x: 1 if x == 'F' else 0)

    # 1 evli 0 yalnız
df['NAME_FAMILY_STATUS'] = df['NAME_FAMILY_STATUS'].apply(lambda x: 1 if x in ['Married', 'Civil marriage'] else 0)

    # NAME_INCOME_TYPE label encoding yapıyorum. 0=Student, 1=Pensioner, 2=Working, 3=Commercial associate, 4=State servant (önem düzeyi görecedir değiştirilebilir)
income_type_mapping = {
    'Student': 0,
    'Pensioner': 1,
    'Working': 2,
    'Commercial associate': 3,
    'State servant': 4
}
df['NAME_INCOME_TYPE_ENCODED'] = df['NAME_INCOME_TYPE'].map(income_type_mapping)
df.drop("NAME_INCOME_TYPE", axis=1, inplace=True)

    # NAME_EDUCATION_TYPE label encoding yapıyorum. 0=Lower secondary , 1=Secondary / secondary special , 2=Higher education , 3=Incomplete higher , 4=Academic degree
education_type_mapping = {
    'Lower secondary': 0,
    'Secondary / secondary special': 1,
    'Higher education': 2,
    'Incomplete higher': 3,
    'Academic degree': 4
}
df['NAME_EDUCATION_TYPE_ENCODED'] = df['NAME_EDUCATION_TYPE'].map(education_type_mapping)
df.drop("NAME_EDUCATION_TYPE", axis=1, inplace=True)

    # ortalama kazanç altında ve üstünde kalan meslek gruplarını belirledim ve altında kalanlara 0 üstünde kalanlara 1 olarak tanımladım.
df.groupby('OCCUPATION_TYPE')['AMT_INCOME_TOTAL'].mean().round(2)
df['AMT_INCOME_TOTAL'].mean()

# Ortalamadan Düşük Kazanç Grubu: 0

# Cleaning staff
# Cooking staff
# Laborers
# Low-skill Laborers
# Medicine staff
# Sales staff
# Secretaries
# Security staff
# Waiters/barmen staff

# Ortalamadan Yüksek Kazanç Grubu: 1
# Accountants
# Drivers
# High skill tech staff
# IT staff
# Managers
# Private service staff
# Realty agents
# Core staff               çok yakın olduğundan inisiyatifen aldım
# HR staff                 çok yakın olduğundan inisiyatifen aldım


df['OCCUPATION_TYPE_ENCODED'] = df['OCCUPATION_TYPE'].apply(
    lambda x: 1 if x in ['Accountants', 'Core staff', 'HR staff', 'Drivers', "High skill tech staff", "IT staff",
                         "Managers", "Private service staff", "Realty agents"] else 0)
df.drop("OCCUPATION_TYPE", axis=1, inplace=True)

    # ev tiplerini iki gruba ayırdım: 0 = ['Rented apartment', 'With parents', 'Office apartment'], 1 = ['House / apartment', 'Municipal apartment', 'Co-op apartment']
df['HOUSING_TYPE_ENCODED'] = df['NAME_HOUSING_TYPE'].apply(lambda x: 1 if x in ['House / apartment', 'Municipal apartment', 'Co-op apartment'] else 0)
df.drop("NAME_HOUSING_TYPE", axis=1, inplace=True)

# Yeni değişkenler üretme

df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
df['CHILD_INDICATOR'] = df['CNT_CHILDREN'].apply(lambda x: 1 if x > 0 else 0)
df["PARENT_SIT"] = df.apply(lambda row: 1 if row['CNT_FAM_MEMBERS'] - row['CNT_CHILDREN'] == 1 else 0, axis=1)

df[df['CNT_CHILDREN'] == 19]['CNT_FAM_MEMBERS']
# Değişken türleri tespit etme ve aşırı değerleri baskılama

df['CNT_FAM_MEMBERS'] = df["CNT_FAM_MEMBERS"].replace(15, 8)        #aile üyesi sayısını maks 8 yaptım.
df['CNT_FAM_MEMBERS'] = df["CNT_FAM_MEMBERS"].replace(20, 8)
df['CNT_FAM_MEMBERS'] = df["CNT_FAM_MEMBERS"].replace(9, 8)

df['CNT_CHILDREN'] = df["CNT_CHILDREN"].replace(14, 7)
df['CNT_CHILDREN'] = df["CNT_CHILDREN"].replace(19, 7)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col, 0.25, 0.75))

df[num_cols].describe().T

replace_with_thresholds(df, "AMT_INCOME_TOTAL")             # gelirler ve çalışma süresini baskıladım.
replace_with_thresholds(df, "MONTHS_EMPLOYED")
replace_with_thresholds(df, "INCOME_PER_PERSON")

# shapiro testi normal dağılım kontrolü (normal değil)
stat, p = shapiro(df['AMT_INCOME_TOTAL'])
alpha = 0.05

if p > alpha:
    print("Veri normal dağılıma uyar (H0 reddedilmedi)")
else:
    print("Veri normal dağılıma uymuyor (H0 reddedildi)")


# herhangi bir sütunda unique değerleri bulmak için kullandığım liste
unique_values = df['CNT_CHILDREN'].unique()
print("Benzersiz değişkenler:", unique_values)

#Standartlaştırma

num_cols_new = ['AMT_INCOME_TOTAL', 'AGE', 'MONTHS_EMPLOYED', "INCOME_PER_PERSON"]        #ID değişkeni hariç numerik değişkenleri standartlaştırıyorum.

scaler = RobustScaler()
df[num_cols_new] = scaler.fit_transform(df[num_cols_new])


##################################################### Modeller #########################################################

y = df["CUSTOMER_TYPE_MATCHED"]
X = df.drop(["CUSTOMER_TYPE_MATCHED", "ID"], axis=1)

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

    # Lojistik regresyon
log_model = LogisticRegression().fit(X, y)
y_pred = log_model.predict(X)
print(classification_report(y, y_pred))
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

    #GBM
gbm_model = GradientBoostingClassifier().fit(X, y)
y_pred = gbm_model.predict(X)
print(classification_report(y, y_pred))
y_prob = gbm_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

    #CART
cart_model = DecisionTreeClassifier().fit(X, y)
y_pred = cart_model.predict(X)
print(classification_report(y, y_pred))
y_prob = cart_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

    #KNN
knn_model = KNeighborsClassifier().fit(X, y)
y_pred = knn_model.predict(X)
print(classification_report(y, y_pred))
y_prob = knn_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

    #Random Forest
rf_model = RandomForestClassifier().fit(X, y)
y_pred = rf_model.predict(X)
print(classification_report(y, y_pred))
y_prob = rf_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

    #Adaboost
ada_model = AdaBoostClassifier().fit(X, y)
y_pred = ada_model.predict(X)
print(classification_report(y, y_pred))
y_prob = ada_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

    #LightGBM
lgb_model = LGBMClassifier().fit(X, y)
y_pred = lgb_model.predict(X)
print(classification_report(y, y_pred))
y_prob = lgb_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

    #CatBoost
cat_model = CatBoostClassifier(verbose=False).fit(X, y)
y_pred = cat_model.predict(X)
print(classification_report(y, y_pred))
y_prob = cat_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)


######################################################
# Model Validation: Holdout
######################################################
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20)
#CART
cart_model = DecisionTreeClassifier().fit(X_train, y_train)
y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

#roc_auc %89

#RF
rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)

#roc_auc %89

######################################################
# Model Validation: 10-Fold Cross Validation
######################################################
#CART
cart_model = DecisionTreeClassifier().fit(X, y)

cv_results = cross_validate(cart_model,
                            X, y,
                            cv=3,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# Accuracy: 0.6594525966537104

cv_results['test_precision'].mean()
# Precision: 0.7729773966656376

cv_results['test_recall'].mean()
# Recall: 0.46305170970195975

cv_results['test_f1'].mean()
# F1-score: 0.5742285698458109

cv_results['test_roc_auc'].mean()
# AUC: 0.62897401704435

#RF
rf_model = RandomForestClassifier().fit(X, y)

cv_results = cross_validate(rf_model,
                            X, y,
                            cv=3,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# Accuracy: 0.6594525966537104

cv_results['test_precision'].mean()
# Precision: 0.7729773966656376

cv_results['test_recall'].mean()
# Recall: 0.46305170970195975

cv_results['test_f1'].mean()
# F1-score: 0.5742285698458109

cv_results['test_roc_auc'].mean()
# AUC: 0.62897401704435

######################## Seçilen modeller için hiperparametre optimizasyonu

#Random Forest
rf_final = RandomForestClassifier(max_depth=8, min_samples_split=20, n_estimators=300).fit(X, y)
y_pred = rf_final.predict(X)
print(classification_report(y, y_pred))
y_prob = rf_final.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

#CART
cart_model = DecisionTreeClassifier(max_depth=8, min_samples_split=19).fit(X, y)
y_pred = cart_model.predict(X)
print(classification_report(y, y_pred))
y_prob = cart_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)

cart_params = {'max_depth': range(1, 20), "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None], "max_features": [5, 7, "auto"], "min_samples_split": [15, 20], "n_estimators": [200, 300]}

classifiers = [("CART", DecisionTreeClassifier(), cart_params), ("RF", RandomForestClassifier(), rf_params)]

hyperparameter_optimization(X, y, cv=3, scoring="f1")

############### Feature Importance#############

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(rf_final, X)



########### sonuç tahmin fonksiyonu ################

def predict_result(df_app):
    df_app = df_app.drop_duplicates(subset='ID', keep='first')
    df_salla = df_app.drop("CUSTOMER_TYPE_MATCHED", axis=1, inplace=True)
    df_p = df_app.dropna()
    df_p.reset_index()
    df_p["AGE"] = df_p["DAYS_BIRTH"] / 365 * -1
    df_p["AGE"] = df_p["AGE"].round()
    df_p["MONTHS_EMPLOYED"] = df_p["DAYS_EMPLOYED"] / 30 * -1
    df_p["MONTHS_EMPLOYED"] = df_p["MONTHS_EMPLOYED"].round()
    df_p.drop("DAYS_EMPLOYED", axis=1, inplace=True)  # eski gün esası doğum tarihi ve işe başlangıç zamanlarını sildim.
    df_p.drop("DAYS_BIRTH", axis=1, inplace=True)
    df_p['FLAG_OWN_CAR'] = df_p['FLAG_OWN_CAR'].apply(lambda x: 1 if x == 'Y' else 0)
    df_p['FLAG_OWN_REALTY'] = df_p['FLAG_OWN_REALTY'].apply(lambda x: 1 if x == 'Y' else 0)
    df_p['CODE_GENDER'] = df_p['CODE_GENDER'].apply(lambda x: 1 if x == 'F' else 0)
    df_p['NAME_FAMILY_STATUS'] = df_p['NAME_FAMILY_STATUS'].apply(lambda x: 1 if x in ['Married', 'Civil marriage'] else 0)
    income_type_mapping = {
        'Student': 0,
        'Pensioner': 1,
        'Working': 2,
        'Commercial associate': 3,
        'State servant': 4
    }
    df_p['NAME_INCOME_TYPE_ENCODED'] = df_p['NAME_INCOME_TYPE'].map(income_type_mapping)
    df_p.drop("NAME_INCOME_TYPE", axis=1, inplace=True)
    education_type_mapping = {
        'Lower secondary': 0,
        'Secondary / secondary special': 1,
        'Higher education': 2,
        'Incomplete higher': 3,
        'Academic degree': 4
    }
    df_p['NAME_EDUCATION_TYPE_ENCODED'] = df_p['NAME_EDUCATION_TYPE'].map(education_type_mapping)
    df_p.drop("NAME_EDUCATION_TYPE", axis=1, inplace=True)
    df_p['OCCUPATION_TYPE_ENCODED'] = df_p['OCCUPATION_TYPE'].apply(
        lambda x: 1 if x in ['Accountants', 'Core staff', 'HR staff', 'Drivers', "High skill tech staff", "IT staff",
                             "Managers", "Private service staff", "Realty agents"] else 0)
    df_p.drop("OCCUPATION_TYPE", axis=1, inplace=True)
    df_p['HOUSING_TYPE_ENCODED'] = df_p['NAME_HOUSING_TYPE'].apply(
        lambda x: 1 if x in ['House / apartment', 'Municipal apartment', 'Co-op apartment'] else 0)
    df_p.drop("NAME_HOUSING_TYPE", axis=1, inplace=True)
    df_p["INCOME_PER_PERSON"] = df_p["AMT_INCOME_TOTAL"] / df_p["CNT_FAM_MEMBERS"]
    df_p['CHILD_INDICATOR'] = df_p['CNT_CHILDREN'].apply(lambda x: 1 if x > 0 else 0)
    df_p["PARENT_SIT"] = df_p.apply(lambda row: 1 if row['CNT_FAM_MEMBERS'] - row['CNT_CHILDREN'] == 1 else 0, axis=1)
    df_p['CNT_FAM_MEMBERS'] = df_p["CNT_FAM_MEMBERS"].replace(15, 8)
    df_p['CNT_FAM_MEMBERS'] = df_p["CNT_FAM_MEMBERS"].replace(20, 8)
    df_p['CNT_FAM_MEMBERS'] = df_p["CNT_FAM_MEMBERS"].replace(9, 8)
    df_p['CNT_CHILDREN'] = df_p["CNT_CHILDREN"].replace(14, 7)
    df_p['CNT_CHILDREN'] = df_p["CNT_CHILDREN"].replace(19, 7)
    replace_with_thresholds(df_p, "AMT_INCOME_TOTAL")
    replace_with_thresholds(df_p, "MONTHS_EMPLOYED")
    replace_with_thresholds(df_p, "INCOME_PER_PERSON")
    num_cols_new = ['AMT_INCOME_TOTAL', 'AGE', 'MONTHS_EMPLOYED', "INCOME_PER_PERSON"]
    scaler = RobustScaler()
    df_p[num_cols_new] = scaler.fit_transform(df_p[num_cols_new])
    print(df_p)
    return df_p

df_p = predict_result(df_app)
Xpre = df_p.drop(["ID"], axis=1)
pre_result = rf_final.predict(Xpre)

sayac = np.count_nonzero(pre_result == 1)

print(f"pre_result dizisinde {sayac} adet 1 bulunuyor.")

df_app = df_app.drop_duplicates(subset='ID', keep='first')
df_test = df_app.dropna()