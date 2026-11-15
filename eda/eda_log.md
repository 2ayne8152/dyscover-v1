# EDA log

## Structural overview

Shape: (6052, 21)  (11 numeric, 9 categorical, 1 target)

Data types:

```
         n_columns
float64         21
```

Summary of numeric features (with skew):

```
                                  count    mean     std    min     25%     50%     75%      max  skew
age                              6052.0   46.23   18.46  18.00   30.00   45.00   61.00    80.00  0.21
income_poverty_ratio             5208.0    2.58    1.64   0.00    1.17    2.20    4.16     5.00  0.27
weight_kg                        5975.0   80.91   21.96  32.60   65.60   77.60   92.40   254.30  1.10
body_mass_index                  5960.0   28.92    7.16  14.20   24.00   27.80   32.50    92.30  1.37
waist_circumference_cm           5793.0   97.64   16.67  56.40   85.80   96.30  107.80   178.00  0.59
sedentary_minutes_per_day        6010.0  327.85  199.76   0.00  180.00  300.00  480.00  1320.00  0.89
sleep_hours_weekday              6003.0    7.60    1.65   2.00    6.50    7.50    8.50    14.00 -0.02
waist_height_ratio               5782.0    0.59    0.10   0.37    0.51    0.58    0.65     1.11  0.60
systolic_bp_mean                 5464.0  122.49   18.40  76.33  109.33  119.67  132.00   218.67  0.95
diastolic_bp_mean                5464.0   74.25   11.46  41.33   66.33   73.33   81.00   135.00  0.59
alcohol_drinks_on_drinking_days  5683.0    1.93    2.21   0.00    0.00    1.00    3.00    15.00  2.39
```

## Leftover special codes (7, 9, 77, 99, 777, 999, 7777, 9999)

None found in categorical columns.

## Leakage check

Potential leakage columns: None

## Missingness

```
                                 count    pct
income_poverty_ratio               844  13.95
systolic_bp_mean                   588   9.72
diastolic_bp_mean                  588   9.72
education_level                    372   6.15
marital_status                     370   6.11
alcohol_drinks_on_drinking_days    369   6.10
history_cvd                        367   6.06
waist_height_ratio                 270   4.46
waist_circumference_cm             259   4.28
body_mass_index                     92   1.52
weight_kg                           77   1.27
sleep_hours_weekday                 49   0.81
sedentary_minutes_per_day           42   0.69
history_high_cholesterol            27   0.45
taking_cholesterol_meds             15   0.25
taking_bp_meds                       6   0.10
history_high_bp                      5   0.08
smoking_status                       2   0.03
age                                  0   0.00
sex                                  0   0.00
target                               0   0.00
```

No columns are missing more than 50% of their values.

## Duplicates and leftover ID columns

Duplicate rows: 0
Leftover ID columns: None

## Target distribution

```
        count    pct
target              
0.0      3433  56.73
1.0      2619  43.27
```

## Outliers (IQR rule)

```
                            feature  count   pct
10  alcohol_drinks_on_drinking_days    172  2.84
3                   body_mass_index    168  2.78
6               sleep_hours_weekday    149  2.46
8                  systolic_bp_mean    145  2.40
2                         weight_kg    144  2.38
9                 diastolic_bp_mean     93  1.54
4            waist_circumference_cm     88  1.45
7                waist_height_ratio     84  1.39
5         sedentary_minutes_per_day     50  0.83
1              income_poverty_ratio      0  0.00
0                               age      0  0.00
```
