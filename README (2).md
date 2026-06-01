# חיזוי דירוגי סרטים ב-IMDb

> **מטלה אקדמית - למידת מכונה**  
> **מגישים:** מאיה הלוי (212074736), עידן וידר (315143305)

---

## תיאור הפרויקט

פרויקט זה מטרתו לחזות את הציון הממוצע של סרטים ב-IMDb (`averageRating`) **לפני יציאתם לאקרנים**, על בסיס מטה-דאטה הזמינה בשלב הקדם-שחרור (מטה-דאטה זמינים: שם הסרט, ז'אנר, אורך, צוות שחקנים, במאי, מדינה, שפה וכו').

המודל הסופי הוא **Random Forest Regressor** עם R² של 0.2944 ו-RMSE של 1.0873 (10-Fold Cross Validation).

---

## מבנה הקבצים

```
project/
├── README.md                              # הקובץ הזה
├── requirements.txt                       # רשימת ספריות
├── notebook_final.ipynb                   # המחברת הראשית
├── עבודה_למידת_מכונה_מתוקן.ipynb         # גרסה עברית של המחברת
├── דוח_עבודת_למידת_מכונה_סופי.docx       # הדוח המלא (8 עמודים)
├── dataset.csv                            # סט הנתונים המקורי
├── elastic_net_model.pkl                  # מודל Elastic Net מאומן
├── random_forest_model.pkl                # מודל Random Forest מאומן (הראשי)
├── imdb_enrich_cache.pkl                  # cache להעשרת נתונים מ-IMDb
├── oscar_wins_cache.pkl                   # cache לזכיות אוסקר מ-Wikidata
├── actor_movies_cache.pkl                 # cache להיסטוריית סרטים של שחקנים
└── data/                                  # קבצי IMDb לבניית cache (אופציונלי)
    ├── title.akas.tsv.gz
    ├── title.crew.tsv.gz
    ├── title.principals.tsv.gz
    └── name.basics.tsv.gz
```

---

## דרישות מערכת

- **Python**: 3.9 ומעלה
- **זיכרון RAM**: לפחות 8GB (12GB מומלץ)
- **שטח דיסק**: ~500MB (כולל קבצי cache)

---

## התקנה

### 1. שכפול הפרויקט

```bash
git clone <repository-url>
cd movie-rating-prediction
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

### אופציה 1: שימוש במודל המאומן (מהיר)

המודלים כבר מאומנים ונשמרו לקבצי `.pkl`. כדי לבצע חיזויים על דאטה חדש:

```python
import pickle
import pandas as pd

# טעינת הדאטה החדש
df_new = pd.read_csv('new_movies.csv')

# טעינת המודל
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# בנייה של הפיצ'רים
# (יש להריץ את prepare_data מהמחברת על הדאטה החדש)
X = prepare_data(df_new)

# חיזוי
predictions = model.predict(X)
print(predictions)
```

### אופציה 2: הרצת המחברת המלאה

לאימון מחדש של המודלים או בדיקה מלאה של התהליך:

```bash
jupyter notebook notebook_final.ipynb
```

ואז בתוך Jupyter: **Kernel → Restart & Run All**

⏱️ **זמן הרצה משוער**: כשעה (כולל בניית caches מ-IMDb בפעם הראשונה)

### אופציה 3: ⚡ הרצה מהירה (cache קיים)

אם קבצי ה-cache כבר קיימים בתיקייה (imdb_enrich_cache.pkl, oscar_wins_cache.pkl, actor_movies_cache.pkl), המחברת תזהה אותם ולא תבנה אותם מחדש. זמן ההרצה יקטן ל-10 דקות בלבד.

---

## תיאור המודל

### המודל הראשי: Random Forest Regressor

| היפר-פרמטר | ערך |
|------------|------|
| `n_estimators` | 300 |
| `max_depth` | 30 |
| `min_samples_split` | 5 |
| `min_samples_leaf` | 2 |
| `max_features` | 'sqrt' |
| `random_state` | 42 |

### ביצועי המודל (10-Fold Cross Validation)

| מדד | Random Forest | Elastic Net |
|-----|----------------|--------------|
| **RMSE** | 1.0873 ± 0.0153 | 1.1229 ± 0.0140 |
| **MAE**  | 0.8290 ± 0.0118 | 0.8582 ± 0.0111 |
| **R²**   | 0.2944 ± 0.0124 | 0.2474 ± 0.0126 |

### Pipeline המודל

```
Input DataFrame
  ↓
ColumnTransformer:
  - Numeric → SimpleImputer(median) → StandardScaler
  - Categorical → SimpleImputer(most_frequent) → OneHotEncoder
  - Multi-label → MultiLabelTopN (Top 25)
  - Director → OneHotEncoder (Top 30)
  ↓
Random Forest Regressor
  ↓
Prediction: float (0-10)
```

### 18 הפיצ'רים שבהם משתמש המודל

**נומריים (10):**
- `startYear`, `runtimeMinutes`, `num_actors`, `num_genres`
- `screen_time_per_actor` ← פיצ'ר הנדסי (Ratio)
- `director_oscar_wins`, `lead_actors_oscar_wins`, `lead_actors_oscar_wins_max`
- `actors_recent_movies_sum`, `actors_recent_movies_max`

**בינאריים (8):**
- `has_director`, `is_sequel`, `has_oscar_winner`, `actors_very_active`
- `missing_actors`, `missing_language`, `missing_country` ← פיצ'רי MNAR
- `decade`

**Multi-label (3):**
- `genres`, `Country`, `Language`

**טקסט (1):**
- `director_name` (Top 30 הבמאים)

---

## פורמט הקלט

הדאטה צריך להגיע כקובץ CSV עם העמודות הבאות:

```
tconst, primaryTitle, startYear, runtimeMinutes, genres,
Language, Country, lead_actors_ids
```

**הערה**: אם חסרים ערכים בעמודות מסוימות, המודל יטפל בהם אוטומטית (פיצ'רי `missing_*` יסומנו).

---

## פורמט הפלט

המודל מחזיר רשימה של תחזיות מספריות:

```python
[6.42, 7.18, 5.23, ...]  # ציון משוער ב-IMDb (סקאלה 1-10)
```

---

## איך בנינו את ה-Cache

קבצי ה-cache נוצרו מ-3 מקורות:

1. **`imdb_enrich_cache.pkl`** - מ-IMDb (title.akas, title.crew, name.basics)
2. **`oscar_wins_cache.pkl`** - מ-Wikidata SPARQL API
3. **`actor_movies_cache.pkl`** - מ-IMDb (title.principals)

### שיקולים מתודולוגיים

- **Strict Whitelist**: הסתמכנו על רשימות מוגדרות מראש של 160 מדינות, 140 שפות, ו-28 ז'אנרים
- **גישה שמרנית להשלמת מדינה**: השתמשנו רק בשורות עם `isOriginalTitle=1` או `types='original'`. עדיף NaN מאשר ניחוש שגוי.
- **מניעת דליפת עתיד**: כל הפיצ'רים ההיסטוריים (אוסקרים, פעילות שחקנים) חושבו על אירועים שקדמו לשנת יציאת הסרט

---

## תקלות נפוצות

### "FileNotFoundError: imdb_enrich_cache.pkl"
**פתרון**: המחברת תיצור את ה-cache בריצה הראשונה. ודאי שקבצי `title.akas.tsv.gz` ו-`name.basics.tsv.gz` נמצאים בתיקייה.

### "MemoryError" בעת בניית actor_movies_cache
**פתרון**: קובץ `title.principals.tsv.gz` גדול (~3GB מפורש). הרצי על מחשב עם לפחות 12GB RAM.

### "ValueError: Input contains NaN" בעת חיזוי
**פתרון**: ה-Pipeline אמור לטפל ב-NaN אוטומטית. אם השגיאה מופיעה, ייתכן שחסרות עמודות בקלט - ודאי שיש בו את כל העמודות הנדרשות.

### תוצאות שונות בכל הרצה
**פתרון**: ודאי ש-`random_state=42` בשני המקומות (train_test_split וגם RandomForest).

---

## רישיון

הפרויקט נוצר במסגרת אקדמית. השימוש בקבצי IMDb כפוף לתנאי השימוש שלהם.

---

## תודות

- **IMDb** - על דאטה גולמית פתוחה
- **Wikidata** - על מאגר נתוני האוסקרים
- **Scikit-Learn** - על ספרייה מצוינת ללמידת מכונה
