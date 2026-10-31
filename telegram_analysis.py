# =====================================================================
#  Аналитика Telegram-канала по CSV-выгрузке (Google Colab)
#  Запуск: загрузите Messgasas.csv в /content или склонируйте репозиторий —
#  файл в папке data/ найдётся сам
# =====================================================================
import os
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "wordcloud", "emoji"], check=False)

import ast
import codecs
import re
from collections import Counter

import emoji
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from wordcloud import WordCloud

CSV_NAME = "Messgasas.csv"
DATA_CANDIDATES = ["/content", "/content/sample_data", "/content/telegram-channel-analysis/data", "data", "."]
PATH = next((os.path.join(d, CSV_NAME) for d in DATA_CANDIDATES if os.path.exists(os.path.join(d, CSV_NAME))), None)
if PATH is None:
    raise FileNotFoundError(f"Не найден {CSV_NAME} в " + ", ".join(DATA_CANDIDATES))
print(f"📂 Данные: {os.path.abspath(PATH)}")
OUT_DIR = "/content" if os.path.isdir("/content") else "."
DAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
MONTHS = ["", "янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
TEMPLATE = "plotly_white"
figures = []  # все интерактивные графики попадут в общий HTML-отчёт


def show(fig, title=None):
    """Показать график и запомнить его для HTML-отчёта."""
    fig.update_layout(template=TEMPLATE, title=title or fig.layout.title.text)
    fig.show()
    figures.append(fig)


def header(text):
    print("\n" + "═" * 70 + f"\n  {text}\n" + "═" * 70)


def short(text, n=80):
    text = re.sub(r"\*\*|\[|\]\(.*?\)|https?://\S+|#\w+", "", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return (text[:n] + "…") if len(text) > n else (text or "📷 медиа без подписи")


# =====================================================================
#  1. Загрузка и подготовка
# =====================================================================
df = pd.read_csv(PATH, sep=";", encoding="utf-8-sig")
df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Europe/Moscow").dt.tz_localize(None)
for col in ["views", "comments", "reactions_total", "forwards"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
df["text"] = df["message"].where(df["message"] != "MediaMessage", "").fillna("")

# Альбом = несколько сообщений с одним временем публикации -> один пост
posts = (
    df.groupby("date")
    .agg(
        views=("views", "max"),
        reactions=("reactions_total", "sum"),
        comments=("comments", "sum"),
        forwards=("forwards", "max"),
        text=("text", lambda s: " ".join(t for t in s if t)),
        items=("message_id", "count"),
        forwarded=("forwarded", "max"),
        raw_reactions=("reactions", lambda s: list(s.dropna())),
    )
    .reset_index()
)
posts["month"] = posts["date"].dt.to_period("M").astype(str)
posts["weekday"] = posts["date"].dt.dayofweek
posts["hour"] = posts["date"].dt.hour
posts["text_len"] = posts["text"].str.len()
posts["has_text"] = posts["text_len"] > 0
posts["interactions"] = posts["reactions"] + posts["comments"] + posts["forwards"]
posts["er"] = posts["interactions"] / posts["views"].replace(0, np.nan) * 100
posts["preview"] = posts["text"].apply(short)
posts["format"] = np.select(
    [~posts["has_text"], posts["items"] > 1],
    ["только медиа", "текст + альбом"],
    "текст / одно фото",
)

members = int(df["channel_members_count"].iloc[0])
channel = df["channel_name"].iloc[0]

# =====================================================================
#  2. Паспорт канала
# =====================================================================
header(f"📊 ПАСПОРТ КАНАЛА «{channel}»")
period_days = (posts["date"].max() - posts["date"].min()).days
kpi = {
    "Подписчиков": f"{members:,}",
    "Период": f"{posts['date'].min():%d.%m.%Y} — {posts['date'].max():%d.%m.%Y} ({period_days} дн.)",
    "Сообщений / постов": f"{len(df):,} / {len(posts):,}",
    "Постов в неделю": f"{len(posts) / period_days * 7:.1f}",
    "Всего просмотров": f"{posts['views'].sum():,.0f}",
    "Медиана просмотров": f"{posts['views'].median():,.0f}",
    "Охват (просмотры/подписчики)": f"{posts['views'].median() / members:.0%}",
    "Средний ER": f"{posts['er'].mean():.2f}%",
    "Всего реакций": f"{posts['reactions'].sum():,.0f}",
    "Всего пересылок": f"{posts['forwards'].sum():,.0f}",
}
for k, v in kpi.items():
    print(f"  {k:<32}{v}")

# =====================================================================
#  3. Динамика: просмотры, тренд и активность
# =====================================================================
header("📈 ДИНАМИКА")
posts_sorted = posts.sort_values("date")
posts_sorted["rolling"] = posts_sorted["views"].rolling(15, min_periods=3).median()

fig = make_subplots(specs=[[{"secondary_y": True}]])
monthly = posts.groupby("month").agg(posts=("date", "size"), median_views=("views", "median"),
                                     er=("er", "mean"), reactions=("reactions", "sum"))
fig.add_bar(x=monthly.index, y=monthly["posts"], name="Постов в месяц", marker_color="#c7d2fe", opacity=0.7)
fig.add_scatter(x=monthly.index, y=monthly["median_views"], name="Медиана просмотров",
                mode="lines+markers", line=dict(color="#dc2626", width=3), secondary_y=True)
fig.update_yaxes(title_text="Постов", secondary_y=False)
fig.update_yaxes(title_text="Просмотры", secondary_y=True)
show(fig, "Активность и охват по месяцам")

fig = px.scatter(posts_sorted, x="date", y="views", color="format", hover_data={"preview": True},
                 opacity=0.6, log_y=True)
fig.add_scatter(x=posts_sorted["date"], y=posts_sorted["rolling"], name="Скользящая медиана (15 постов)",
                line=dict(color="black", width=2))
show(fig, "Каждый пост на шкале времени (наведите мышь)")

print(monthly.round(2).to_string())
best_m, worst_m = monthly["median_views"].idxmax(), monthly["median_views"].idxmin()
print(f"\n  🏆 Лучший месяц: {best_m} (медиана {monthly.loc[best_m, 'median_views']:.0f})")
print(f"  📉 Худший месяц: {worst_m} (медиана {monthly.loc[worst_m, 'median_views']:.0f})")

# =====================================================================
#  4. Календарь публикаций (как на GitHub)
# =====================================================================
cal = posts.groupby(posts["date"].dt.normalize()).size()
cal = cal.reindex(pd.date_range(cal.index.min(), cal.index.max()), fill_value=0)
cal_df = pd.DataFrame({"count": cal.values}, index=cal.index)
cal_df["week"] = cal_df.index.to_period("W").start_time
cal_df["weekday"] = cal_df.index.dayofweek
grid = cal_df.pivot_table(index="weekday", columns="week", values="count", aggfunc="sum")
fig = go.Figure(go.Heatmap(z=grid.values, x=grid.columns, y=DAYS, colorscale="Greens",
                           hovertemplate="Неделя с %{x|%d.%m.%Y}<br>%{y}: %{z} постов<extra></extra>"))
fig.update_yaxes(autorange="reversed")
show(fig, "Календарь публикаций")

# =====================================================================
#  5. Когда публиковать: тепловая карта день × час
# =====================================================================
header("⏰ ЛУЧШЕЕ ВРЕМЯ ДЛЯ ПУБЛИКАЦИЙ")
heat = posts.pivot_table(index="weekday", columns="hour", values="views", aggfunc="median")
cnt = posts.pivot_table(index="weekday", columns="hour", values="views", aggfunc="size")
heat = heat.where(cnt >= 3)  # ячейки с < 3 постами ненадёжны
fig = go.Figure(go.Heatmap(z=heat.values, x=[f"{h}:00" for h in heat.columns], y=[DAYS[i] for i in heat.index],
                           colorscale="YlOrRd", customdata=cnt.reindex_like(heat).values,
                           hovertemplate="%{y} %{x}<br>Медиана: %{z:.0f}<br>Постов: %{customdata}<extra></extra>"))
fig.update_yaxes(autorange="reversed")
show(fig, "Медиана просмотров: день недели × час (МСК)")

slots = (posts.groupby(["weekday", "hour"]).agg(posts=("views", "size"), median=("views", "median"))
         .query("posts >= 3").sort_values("median", ascending=False))
print("  Топ-5 слотов (минимум 3 поста в слоте):")
for (wd, h), r in slots.head(5).iterrows():
    print(f"   {DAYS[wd]} {h:02d}:00 — медиана {r['median']:.0f} просмотров ({int(r['posts'])} постов)")

# =====================================================================
#  6. Вирусные посты и аутсайдеры
# =====================================================================
header("🚀 ВИРУСНЫЕ ПОСТЫ")
# Сравниваем с медианой соседних постов, чтобы учесть рост канала
posts_sorted["virality"] = posts_sorted["views"] / posts_sorted["views"].rolling(21, center=True, min_periods=5).median()
viral = posts_sorted.nlargest(10, "virality")
for _, r in viral.iterrows():
    print(f"  ×{r['virality']:.1f}  {r['date']:%d.%m.%Y}  👁 {r['views']:>6,.0f}  {r['preview']}")

print("\n  😴 Самые недооценённые посты:")
for _, r in posts_sorted.nsmallest(5, "virality").iterrows():
    print(f"  ×{r['virality']:.2f}  {r['date']:%d.%m.%Y}  👁 {r['views']:>6,.0f}  {r['preview']}")

fig = px.scatter(posts, x="views", y="interactions", size="forwards", color="format", log_x=True,
                 hover_data={"preview": True, "date": "|%d.%m.%Y"}, size_max=40)
show(fig, "Просмотры vs взаимодействия (размер = пересылки)")

# =====================================================================
#  7. Рейтинги
# =====================================================================
header("🏅 ТОП-5 ПО КАЖДОЙ МЕТРИКЕ")
for col, icon in [("views", "👁"), ("reactions", "❤️"), ("forwards", "🔁"), ("comments", "💬"), ("er", "⚡ ER%")]:
    print(f"\n  {icon} {col}")
    for _, r in posts.nlargest(5, col).iterrows():
        print(f"    {r[col]:>8,.1f}  {r['date']:%d.%m.%Y}  {r['preview']}")

# =====================================================================
#  8. Реакции: настоящие эмодзи
# =====================================================================
header("😍 РЕАКЦИИ")


def to_emoji(name):
    if name.startswith("\\U") or name.startswith("\\u"):
        return codecs.decode(name, "unicode_escape")
    return emoji.emojize(name, language="alias") if name.startswith(":") else name


rows = []
for _, p in posts.iterrows():
    for raw in p["raw_reactions"]:
        try:
            for name, n in ast.literal_eval(raw).items():
                rows.append({"month": p["month"], "emoji": to_emoji(name), "n": n})
        except (ValueError, SyntaxError):
            pass
react = pd.DataFrame(rows)
total_react = react.groupby("emoji")["n"].sum().sort_values(ascending=False)
for e, n in total_react.head(10).items():
    bar = "█" * int(40 * n / total_react.iloc[0])
    print(f"  {e}  {bar} {n} ({n / total_react.sum():.0%})")

fig = px.treemap(total_react.head(15).reset_index(), path=["emoji"], values="n")
fig.update_traces(textinfo="label+value+percent root", textfont_size=22)
show(fig, "Карта реакций")

top_e = total_react.head(6).index
mix = react[react["emoji"].isin(top_e)].pivot_table(index="month", columns="emoji", values="n", aggfunc="sum").fillna(0)
mix = mix.div(mix.sum(axis=1), axis=0) * 100
fig = px.area(mix, labels={"value": "% реакций", "month": ""})
show(fig, "Как менялся набор реакций")

# =====================================================================
#  9. Контент: формат, длина, темы
# =====================================================================
header("📝 КОНТЕНТ")
fmt = posts.groupby("format").agg(постов=("views", "size"), медиана_просмотров=("views", "median"),
                                  ER=("er", "mean"), пересылки=("forwards", "mean")).round(2)
print(fmt.to_string())

tp = posts[posts["has_text"]].copy()
tp["длина"] = pd.cut(tp["text_len"], [0, 150, 300, 600, 1000, 100_000],
                     labels=["<150", "150–300", "300–600", "600–1000", "1000+"])
fig = px.box(tp, x="длина", y="views", points="all", hover_data=["preview"], log_y=True)
show(fig, "Длина текста и просмотры")

TOPICS = {
    "🥊 спорт": r"соревнован|турнир|чемпионат|спорт|бокс|самбо|стрельб|кубок|первенств",
    "🎭 культура": r"фестивал|концерт|конкурс|творч|музык|театр",
    "🎓 присяга/выпуск": r"присяг|выпуск|диплом|первокурс",
    "🔬 наука": r"конференц|научн|форум|семинар|олимпиад",
    "🎖 патриотика": r"побед|ветеран|памят|герой|наследник",
    "🎉 праздники": r"поздравля|с днём|с днем|с новым годом|праздник",
    "🚪 абитуриентам": r"открытых дверей|абитуриент|поступ|приём|прием",
}
low = posts["text"].str.lower()
for t, pat in TOPICS.items():
    posts[t] = low.str.contains(pat, regex=True)
posts["тема"] = posts[list(TOPICS)].idxmax(axis=1).where(posts[list(TOPICS)].any(axis=1), "прочее")
topic_stats = (posts.groupby("тема").agg(постов=("views", "size"), медиана=("views", "median"),
                                         ER=("er", "mean"), пересылки=("forwards", "sum"))
               .sort_values("медиана", ascending=False).round(2))
print("\n" + topic_stats.to_string())

fig = px.sunburst(posts, path=["тема", "format"], values="views", color="er", color_continuous_scale="RdYlGn")
show(fig, "Темы и форматы (площадь = просмотры, цвет = ER)")

theme_month = posts.pivot_table(index="month", columns="тема", values="views", aggfunc="size").fillna(0)
show(px.bar(theme_month, labels={"value": "постов", "month": ""}), "О чём канал писал по месяцам")

# =====================================================================
#  10. Облако слов и хэштеги
# =====================================================================
header("☁️ СЛОВА")
STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от
меня еще нет о из ему когда даже ну ли если уже или ни быть был него до вас вам ведь там потом себя ничего ей может они
тут где есть надо ней для мы тебя их чем была сам без чего раз тоже себе под будет тогда кто этот того потому этого
какой ним здесь этом один мой тем чтобы нее сейчас были куда можно при два об другой после над больше тот через эти нас
про всего них какая много три эту моя свою этой перед также года году который которые которых которая также
подробнее читайте официальном сайте университета университет университете подписывайтесь наши страницы telegram max
vk https ru рф уважаемые подписчики канала имени россии мосу мвд кикотя московского московском российской федерации
среди состоялся сегодня""".split())
clean = low.str.replace(r"https?://\S+|#\w+|\[.*?\]\(.*?\)", " ", regex=True)
tokens = clean.apply(lambda t: [w for w in re.findall(r"[а-яё]{4,}", t) if w not in STOP])
words = Counter(w for ts in tokens for w in ts)
token_sets = tokens.apply(set)
wc = WordCloud(width=1400, height=600, background_color="white", colormap="viridis",
               max_words=150).generate_from_frequencies(words)
plt.figure(figsize=(16, 7))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Облако слов канала", fontsize=18)
plt.show()
wc.to_file(f"{OUT_DIR}/wordcloud.png")

# Какие слова «продают» пост: средние просмотры постов со словом
word_views = []
for w, n in words.most_common(300):
    if n >= 8:
        mask = token_sets.apply(lambda ts: w in ts)
        word_views.append((w, n, posts.loc[mask, "views"].median()))
wv = pd.DataFrame(word_views, columns=["слово", "упоминаний", "медиана просмотров"]).sort_values("медиана просмотров")
print("  🔝 Слова-магниты (посты с ними смотрят больше):")
print(wv.tail(10).iloc[::-1].to_string(index=False))
print("\n  🔻 Слова-тормоза:")
print(wv.head(10).to_string(index=False))

# =====================================================================
#  11. Регулярность публикаций
# =====================================================================
header("📅 РЕГУЛЯРНОСТЬ")
gaps = posts_sorted["date"].diff().dt.total_seconds() / 86400
i = gaps.idxmax()
print(f"  Средний интервал между постами: {gaps.mean():.1f} дн., медиана: {gaps.median():.1f} дн.")
print(f"  Самый долгий перерыв: {gaps.max():.0f} дн. (до {posts_sorted.loc[i, 'date']:%d.%m.%Y})")
active_days = set(posts["date"].dt.date)
streak = best = 0
for d in pd.date_range(min(active_days), max(active_days)):
    streak = streak + 1 if d.date() in active_days else 0
    best = max(best, streak)
print(f"  Самая длинная серия дней подряд с постами: {best}")
print(f"  Дней с постами: {len(active_days)} из {period_days} ({len(active_days) / period_days:.0%})")

# Влияет ли частота на охват: посты в «загруженные» дни
per_day = posts.groupby(posts["date"].dt.date)["views"].transform("size")
posts["posts_that_day"] = per_day.clip(upper=5)
print("\n  Медиана просмотров в зависимости от числа постов в тот же день:")
print(posts.groupby("posts_that_day")["views"].agg(["size", "median"]).round(0).to_string())

# =====================================================================
#  12. Машинное обучение: что влияет на просмотры
# =====================================================================
header("🤖 МОДЕЛЬ: ЧТО ВЛИЯЕТ НА ПРОСМОТРЫ")
posts["days_from_start"] = (posts["date"] - posts["date"].min()).dt.days
features = ["hour", "weekday", "days_from_start", "text_len", "items", "has_text", "forwarded",
            "posts_that_day"] + list(TOPICS)
X = posts[features].astype(float)
y = np.log1p(posts["views"])
model = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=42)
r2 = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2")
model.fit(X, y)
print(f"  Точность модели (R², кросс-валидация): {r2.mean():.2f} ± {r2.std():.2f}")
names = {"hour": "час", "weekday": "день недели", "days_from_start": "дата (рост канала)",
         "text_len": "длина текста", "items": "фото в альбоме", "has_text": "есть текст",
         "forwarded": "пересланный", "posts_that_day": "постов в тот день"}
imp = pd.Series(model.feature_importances_, index=[names.get(f, f) for f in features]).sort_values()
show(px.bar(imp, orientation="h", labels={"value": "важность", "index": ""}), "Что сильнее всего влияет на просмотры")

# Прогноз: лучший слот для «типичного» поста сегодня
typical = X.median().to_dict()
typical["days_from_start"] = posts["days_from_start"].max()
grid_rows = [{**typical, "weekday": wd, "hour": h} for wd in range(7) for h in range(8, 23)]
pred = pd.DataFrame(grid_rows)
pred["прогноз"] = np.expm1(model.predict(pred[features]))
best_slot = pred.loc[pred["прогноз"].idxmax()]
print(f"  🎯 Модель советует публиковать: {DAYS[int(best_slot['weekday'])]} в {int(best_slot['hour'])}:00 "
      f"(прогноз ≈ {best_slot['прогноз']:.0f} просмотров)")


def predict_views(text, weekday, hour, photos=1):
    """Прогноз просмотров для нового поста. weekday: 0=Пн … 6=Вс."""
    row = {**typical, "weekday": weekday, "hour": hour, "text_len": len(text), "items": photos,
           "has_text": bool(text), "posts_that_day": 1}
    for t, pat in TOPICS.items():
        row[t] = bool(re.search(pat, text.lower()))
    return float(np.expm1(model.predict(pd.DataFrame([row])[features]))[0])


demo = "Курсанты Университета заняли первое место на чемпионате по самбо"
print(f"  Пример: «{demo}» в Пт 17:00 → ≈ {predict_views(demo, 4, 17, photos=5):.0f} просмотров")

# =====================================================================
#  13. Поиск по постам
# =====================================================================
def find_posts(query, n=10):
    """Найти посты по слову и показать их статистику. Пример: find_posts('бокс')"""
    m = posts[low.str.contains(query.lower(), regex=False)].sort_values("views", ascending=False)
    print(f"\n🔎 «{query}»: найдено {len(m)} постов, медиана просмотров {m['views'].median():.0f} "
          f"(по каналу {posts['views'].median():.0f})")
    for _, r in m.head(n).iterrows():
        print(f"   {r['date']:%d.%m.%Y}  👁 {r['views']:>6,.0f}  ❤️ {r['reactions']:>4.0f}  {r['preview']}")
    return m


header("🔎 ПОИСК")
for q in ["бокс", "присяг", "открытых дверей"]:
    find_posts(q, n=3)

# =====================================================================
#  14. Автоматические выводы
# =====================================================================
header("💡 ГЛАВНЫЕ ВЫВОДЫ")
first, last = monthly["median_views"].iloc[:3].mean(), monthly["median_views"].iloc[-4:-1].mean()
top_topic = topic_stats.drop("прочее", errors="ignore")["медиана"].idxmax()
best_fmt = fmt["ER"].idxmax()
insights = [
    f"Охват {'вырос' if last > first else 'снизился'} в {max(last, first) / min(last, first):.1f} раза: "
    f"с ~{first:.0f} до ~{last:.0f} просмотров на пост.",
    f"Самая «смотрибельная» тема — {top_topic} (медиана {topic_stats.loc[top_topic, 'медиана']:.0f}).",
    f"Лучший формат по вовлечённости — «{best_fmt}» (ER {fmt.loc[best_fmt, 'ER']:.2f}%).",
    f"Главная реакция аудитории — {total_react.index[0]} ({total_react.iloc[0] / total_react.sum():.0%} всех реакций).",
    f"Пик просмотров — {DAYS[slots.index[0][0]]} в {slots.index[0][1]}:00.",
    f"Самый вирусный пост ({viral.iloc[0]['date']:%d.%m.%Y}) набрал в {viral.iloc[0]['virality']:.0f} раз больше обычного.",
]
for s in insights:
    print("  •", s)

# =====================================================================
#  15. Экспорт: Excel + интерактивный HTML-отчёт
# =====================================================================
with pd.ExcelWriter(f"{OUT_DIR}/channel_report.xlsx") as xw:
    pd.Series(kpi).to_frame("значение").to_excel(xw, sheet_name="Паспорт")
    monthly.to_excel(xw, sheet_name="По месяцам")
    slots.to_excel(xw, sheet_name="Слоты")
    topic_stats.to_excel(xw, sheet_name="Темы")
    fmt.to_excel(xw, sheet_name="Форматы")
    total_react.to_frame("кол-во").to_excel(xw, sheet_name="Реакции")
    wv.sort_values("медиана просмотров", ascending=False).to_excel(xw, sheet_name="Слова", index=False)
    (posts.sort_values("views", ascending=False)
     [["date", "views", "reactions", "comments", "forwards", "er", "тема", "format", "text"]]
     .to_excel(xw, sheet_name="Все посты", index=False))

with open(f"{OUT_DIR}/channel_report.html", "w", encoding="utf-8") as f:
    f.write(f"<html><head><meta charset='utf-8'><title>{channel}</title></head>"
            f"<body style='font-family:sans-serif;max-width:1200px;margin:auto'>"
            f"<h1>Аналитика канала «{channel}»</h1><ul>"
            + "".join(f"<li><b>{k}:</b> {v}</li>" for k, v in kpi.items())
            + "</ul><h2>Выводы</h2><ul>" + "".join(f"<li>{s}</li>" for s in insights) + "</ul>")
    for i, fg in enumerate(figures):
        f.write(fg.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False))
    f.write("</body></html>")

print(f"\n✅ Готово! Файлы в {OUT_DIR}: channel_report.xlsx, channel_report.html, wordcloud.png")
print("   Попробуйте: find_posts('самбо')  или  predict_views('текст поста', weekday=4, hour=17, photos=3)")
