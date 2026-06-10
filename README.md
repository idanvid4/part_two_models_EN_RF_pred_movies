# חיזוי דירוגי סרטים ב-IMDb

> **מטלת סיכום - חלק 2: למידת מכונה**
> **מגישים:** מאיה הלוי (212074736), עידן וידר (315143305)

---

## תיאור הפרויקט

פרויקט זה חוזה את הציון הממוצע של סרטים ב-IMDb (`averageRating`) **לפני יציאתם לאקרנים**, על בסיס מטה-דאטה הזמינה בשלב הקדם-שחרור (ז'אנר, אורך, צוות שחקנים, במאי, מדינה, שפה ועוד).

המודל הסופי הוא **Random Forest Regressor** עם R² של 0.2944 ו-RMSE של 1.0873 (10-Fold Cross Validation).

---

## מבנה הקבצים

```
part_two_models_EN_RF_pred_movies/
├── README.md                       # הקובץ הזה
├── requirements.txt                # רשימת ספריות עם גרסאות
├── ML_project2.ipynb               # המחברת הראשית (קוד מלא)
├── report.pdf                      # הדוח המלא (8 עמודים)
├── dataset.csv                     # סט הנתונים מחלק 1
├── model.pkl                       # המודל לתחרות (נטען ב-test-time)
├── elastic_net_model.pkl           # מודל Elastic Net מאומן
├── random_forest_model.pkl         # מודל Random Forest מאומן
├── imdb_enrich_cache.pkl           # cache: מדינה/שפה/במאי מ-IMDb
├── oscar_wins_cache.pkl            # cache: זכיות אוסקר מ-Wikidata
└── actor_movies_cache.pkl          # cache: היסטוריית סרטים של שחקנים
```

---

## דרישות מערכת

- **Python**: 3.10 ומעלה
- **זיכרון RAM**: לפחות 8GB (12GB מומלץ)
- **שטח דיסק**: ~400MB (כולל קבצי cache)

---

## התקנה

### 1. שכפול ה-repository

```bash
git clone https://github.com/idanvid4/part_two_models_EN_RF_pred_movies.git
cd part_two_models_EN_RF_pred_movies
```

### 2. יצירת סביבת עבודה (מומלץ)

```bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows
```

### 3. התקנת הספריות

```bash
pip install -r requirements.txt
```

---

## הרצה

### הרצת המחברת המלאה

```bash
jupyter notebook ML_project2.ipynb
```

ואז בתוך Jupyter: **Kernel → Restart & Run All**

חשוב: כל קבצי ה-cache (`*.pkl`) מצורפים ל-repository. המחברת מזהה אותם וטוענת אותם אוטומטית - אין צורך לבנות אותם מחדש. כך זמן ההרצה מתקצר משמעותית.

### שימוש במודל לתחרות (test-time)

המודל הסופי שמור ב-`model.pkl`. רצף ההפעלה במעמד הבדיקה:

```python
import pandas as pd
import joblib

df_2025 = pd.read_csv("test.csv")
X = prepare_data(df_2025)              # מוגדרת במחברת
model = joblib.load("model.pkl")
y_pred = model.predict(X)
```

הערה חשובה: הפונקציה `prepare_data` טוענת את קבצי ה-cache באמצעות path יחסי, ומשתמשת ב-`.get()` כך שגם סרטים שאינם קיימים ב-cache לא יגרמו לקריסה (הערכים החסרים מטופלים על ידי `SimpleImputer` בתוך ה-Pipeline).

---

## תיאור המודל

### המודל הראשי: Random Forest Regressor

היפר-פרמטרים אופטימליים (נמצאו ב-Grid Search עם 5-Fold CV):

| היפר-פרמטר | ערך |
|------------|------|
| `n_estimators` | 300 |
| `max_depth` | 25 |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 1 |
| `max_features` | 'sqrt' |
| `random_state` | 42 |

### המודל לתחרות: Elastic Net

`model.pkl` מכיל את מודל ה-Elastic Net (כנדרש בסעיף התחרות):

| היפר-פרמטר | ערך |
|------------|------|
| `alpha` | 0.001 |
| `l1_ratio` | 0.05 |
| `max_iter` | 20000 |
| `random_state` | 42 |

### ביצועי המודלים (10-Fold Cross Validation)

| מדד | Random Forest | Elastic Net |
|-----|----------------|--------------|
| **RMSE** | 1.0873 ± 0.0153 | 1.1229 ± 0.0140 |
| **MAE**  | 0.8290 ± 0.0118 | 0.8582 ± 0.0111 |
| **R²**   | 0.2944 ± 0.0124 | 0.2474 ± 0.0126 |

### מבנה ה-Pipeline

```
Input DataFrame (prepare_data)
  |
ColumnTransformer:
  - Numeric (key) -> SimpleImputer(median) -> StandardScaler -> PolynomialFeatures(deg=2)
  - Numeric (other) -> SimpleImputer(median) -> StandardScaler
  - Categorical -> SimpleImputer(most_frequent) -> OneHotEncoder
  - Director -> OneHotEncoder(max_categories=30)
  - Multi-label -> MultiLabelTopN(top_n=25)
  |
Model (Random Forest / Elastic Net)
  |
Prediction: float (1.0-10.0)
```

כל שלבי העיבוד המקדים מתבצעים **בתוך** ה-Pipeline, כך שב-Cross-Validation הם נלמדים רק על fold האימון - מניעת data leakage.

### 18 הפיצ'רים

**נומריים (10):** `startYear`, `runtimeMinutes`, `num_actors`, `num_genres`, `screen_time_per_actor` (Ratio), `director_oscar_wins`, `lead_actors_oscar_wins`, `lead_actors_oscar_wins_max`, `actors_recent_movies_sum`, `actors_recent_movies_max`

**בינאריים (8):** `has_director`, `is_sequel`, `has_oscar_winner`, `actors_very_active` (Binning), `missing_actors`, `missing_language`, `missing_country` (MNAR), `decade`

**Multi-label (3):** `genres`, `Country`, `Language`

**טקסט (1):** `director_name`

---

## העשרת נתונים - קבצי Cache

קבצי ה-cache נוצרו מ-3 מקורות חיצוניים ומצורפים ל-repository:

| קובץ | מקור | תוכן |
|------|------|------|
| `imdb_enrich_cache.pkl` | IMDb (title.akas, title.crew, name.basics) | מדינה, שפה, במאי |
| `oscar_wins_cache.pkl` | Wikidata SPARQL | זכיות אוסקר |
| `actor_movies_cache.pkl` | IMDb (title.principals) | היסטוריית סרטים של שחקנים |

### שיקולים מתודולוגיים

- **גישה שמרנית למדינה**: שדה `region` ב-IMDb מציין טריטוריית הפצה ולא מדינת הפקה. השתמשנו רק בשורות עם `isOriginalTitle=1` או `types='original'`. אם אין סימון אמין - מחזירים None (עדיף NaN מפורש על פני ניחוש שגוי).
- **Strict Whitelist**: רשימות מוגדרות מראש של ~160 מדינות, ~140 שפות, ו-28 ז'אנרים. כל ערך שאינו ברשימה נדחה.
- **מניעת דליפת עתיד**: כל הפיצ'רים ההיסטוריים (אוסקרים, פעילות שחקנים) מחושבים רק על אירועים שקדמו לשנת יציאת הסרט.

---

## תקלות נפוצות

### "FileNotFoundError: title.akas.tsv"
המחברת מנסה לבנות cache מחדש כי קובץ ה-pkl חסר. ודאו שכל שלושת קבצי ה-cache (`imdb_enrich_cache.pkl`, `oscar_wins_cache.pkl`, `actor_movies_cache.pkl`) נמצאים באותה תיקייה של המחברת.

### תוצאות שונות בכל הרצה
ודאו ש-`random_state=42` בכל המקומות הרלוונטיים (כבר מוגדר במחברת).

### "DtypeWarning: Columns have mixed types"
אזהרה לא קריטית. ניתן להוסיף `low_memory=False` ל-`pd.read_csv` כדי להסתירה.

---

## הערות

- הקוד רץ end-to-end על `dataset.csv` ללא שגיאות.
- שימוש ב-`random_state=42` בכל מקום רלוונטי לצורך reproducibility.
- הטרנספורמר המותאם `MultiLabelTopN` מוגדר בתא הראשון (imports), כך שטעינת `model.pkl` עובדת גם בהרצה חלקית.
