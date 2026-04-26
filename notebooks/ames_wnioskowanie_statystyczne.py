import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Wczytanie danych - dostosuj sciezke, jezeli plik jest w innym miejscu
DATA_PATH = "../data/raw/AmesHousing.csv"
df = pd.read_csv(DATA_PATH)

# Zmienna grupujaca: czy nieruchomosc ma kominek
# W Ames Housing brak wartosci w 'Fireplace Qu' oznacza brak kominka.
df["Has_Fireplace"] = df["Fireplace Qu"].notna()

with_fireplace = df.loc[df["Has_Fireplace"], "SalePrice"].dropna()
without_fireplace = df.loc[~df["Has_Fireplace"], "SalePrice"].dropna()

alpha = 0.05
confidence = 1 - alpha


def mean_ci_t(sample, confidence=0.95):
    """Przedzial ufnosci t-Studenta dla sredniej jednej populacji."""
    sample = pd.Series(sample).dropna()
    n = len(sample)
    mean = sample.mean()
    sem = stats.sem(sample)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * sem
    return mean, mean - margin, mean + margin


def welch_ci_diff(x, y, confidence=0.95):
    """Przedzial ufnosci dla roznicy srednich mu_x - mu_y, wariant Welcha."""
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    nx, ny = len(x), len(y)
    mean_diff = x.mean() - y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se = np.sqrt(vx / nx + vy / ny)
    df_welch = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    t_crit = stats.t.ppf((1 + confidence) / 2, df=df_welch)
    return mean_diff, mean_diff - t_crit * se, mean_diff + t_crit * se, df_welch

# 1. Statystyki opisowe i estymacja punktowa
summary = df.groupby("Has_Fireplace")["SalePrice"].agg(["count", "mean", "median", "std", "min", "max"])
summary.index = ["bez kominka", "z kominkiem"]
print("Statystyki opisowe:")
print(summary.round(2))

# 2. Estymacja przedzialowa srednich
mean_yes, ci_yes_low, ci_yes_high = mean_ci_t(with_fireplace, confidence)
mean_no, ci_no_low, ci_no_high = mean_ci_t(without_fireplace, confidence)
print("\n95% przedzialy ufnosci dla sredniej SalePrice:")
print(f"Z kominkiem:   srednia = {mean_yes:.2f}, CI = [{ci_yes_low:.2f}, {ci_yes_high:.2f}]")
print(f"Bez kominka:   srednia = {mean_no:.2f}, CI = [{ci_no_low:.2f}, {ci_no_high:.2f}]")

# 3. Przedzial ufnosci dla roznicy srednich
mean_diff, diff_low, diff_high, df_welch = welch_ci_diff(with_fireplace, without_fireplace, confidence)
print("\n95% przedzial ufnosci dla roznicy srednich: z kominkiem - bez kominka")
print(f"roznica = {mean_diff:.2f}, CI = [{diff_low:.2f}, {diff_high:.2f}], df = {df_welch:.2f}")

# 4. Test hipotezy - test t Welcha dla dwoch niezaleznych prob
# H0: mu_z_kominkiem = mu_bez_kominka
# H1: mu_z_kominkiem > mu_bez_kominka
try:
    test = stats.ttest_ind(with_fireplace, without_fireplace, equal_var=False, alternative="greater")
except TypeError:
    # Dla starszych wersji scipy bez argumentu alternative
    test_two_sided = stats.ttest_ind(with_fireplace, without_fireplace, equal_var=False)
    test = type("TestResult", (), {})()
    test.statistic = test_two_sided.statistic
    test.pvalue = test_two_sided.pvalue / 2 if test_two_sided.statistic > 0 else 1 - test_two_sided.pvalue / 2

print("\nTest t Welcha, H1: srednia cena domow z kominkiem jest wieksza")
print(f"t = {test.statistic:.4f}, p-value = {test.pvalue:.6g}")

if test.pvalue < alpha:
    print("Wniosek: odrzucamy H0 na poziomie istotnosci 0.05.")
else:
    print("Wniosek: brak podstaw do odrzucenia H0 na poziomie istotnosci 0.05.")

# 5. Wykres do raportu
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x="Has_Fireplace", y="SalePrice")
plt.xticks([0, 1], ["Bez kominka", "Z kominkiem"])
plt.xlabel("Obecność kominka")
plt.ylabel("Cena sprzedaży [USD]")
plt.title("Porównanie cen nieruchomości z kominkiem i bez kominka")
plt.tight_layout()
plt.savefig("fireplace_saleprice_boxplot.png", dpi=300)
plt.show()
