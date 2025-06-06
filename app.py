import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from PIL import Image

# Загрузка моделей
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'k-Nearest Neighbors': 'kNN_model_1.pkl',
        'Gradient Boosting': 'gb_model_2.pkl',
        'CatBoost': 'catboost_model_3.cbm',
        'Bagging': 'bagging_model_4.pkl',
        'Stacking': 'stacking_model_5.pkl'
    }
    
    for name, file in model_files.items():
        try:
            if name == 'CatBoost':
                models[name] = CatBoostClassifier().load_model(file)
            else:
                with open(file, 'rb') as f:
                    models[name] = pickle.load(f)
        except Exception as e:
            st.error(f"Ошибка загрузки {name}: {str(e)}")
            st.warning(f"Убедитесь, что файл {file} существует в папке с приложением")
    
    return models

models = load_models()

# Загрузка данных
@st.cache_data
def load_data():
    return pd.read_csv('smoke_detector_task_cleaned2.csv')

data = load_data()

# Настройка страниц
st.sidebar.title("Навигация")
page = st.sidebar.radio("Выберите страницу:",
                        ["О разработчике", "О данных", "Визуализации", "Предсказание"])

if page == "О разработчике":
    st.title("Информация о разработчике")
    col1, col2 = st.columns(2)

    with col1:
        st.header("ФИО: Нестерова Анна Алексеевна")
        st.subheader("Номер учебной группы: ФИТ-232")
        st.write("Тема РГР: Анализ данных датчиков дыма и прогнозирование пожарной тревоги")

    with col2:
        try:
            image = Image.open('developer_photo.jpg')
            st.image(image, caption='Фото разработчика', width=200)
        except:
            st.warning("Фото разработчика не найдено")

elif page == "О данных":
    st.title("Информация о наборе данных")

    st.header("Описание предметной области")
    st.write("""
    Данный датасет содержит информацию с датчиков, отслеживающих параметры окружающей среды
    для выявления потенциальных пожаров. Включает показатели температуры, влажности,
    концентрации газов и других параметров.
    """)

    st.header("Описание признаков")
    st.write("""
    - Temperature[C]: Температура в градусах Цельсия
    - Humidity[%]: Влажность в процентах
    - TVOC[ppb]: Концентрация летучих органических соединений
    - eCO2[ppm]: Эквивалентная концентрация CO2
    - Raw H2: Концентрация водорода
    - Raw Ethanol: Концентрация этанола
    - Pressure[hPa]: Атмосферное давление
    - PM1.0, PM2.5: Концентрация частиц
    - NC0.5, NC1.0, NC2.5: Количество частиц
    - CNT: Общее количество частиц
    - Fire Alarm: Пожарная тревога (целевая переменная)
    """)

    st.header("Предобработка данных")
    st.write("""
    - Удаление дубликатов
    - Нормализация числовых признаков
    - Балансировка классов (если необходимо)
    """)

elif page == "Визуализации":
    st.title("Визуализации зависимостей в данных")

    st.header("1. Распределение случаев пожарной тревоги")
    fig1, ax1 = plt.subplots()
    data['Fire Alarm'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax1)
    ax1.set_ylabel("")
    st.pyplot(fig1)

    st.header("2. Зависимость тревоги от температуры")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x='Fire Alarm', y='Temperature[C]', data=data, ax=ax2)
    st.pyplot(fig2)

    st.header("3. Корреляция между параметрами")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".1f", cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    st.header("4. Распределение концентрации CO2")
    fig4, ax4 = plt.subplots()
    sns.histplot(data['eCO2[ppm]'], bins=30, kde=True, ax=ax4)
    ax4.set_xlabel("Концентрация CO2 [ppm]")
    st.pyplot(fig4)

elif page == "Предсказание":
    st.title("Прогнозирование пожарной тревоги")

    model_choice = st.selectbox("Выберите модель для предсказания:",
                              list(models.keys()))

    st.header("Введите параметры датчиков")

    col1, col2 = st.columns(2)

    with col1:
        temperature = st.number_input("Температура [°C]", value=25.0, min_value=-50.0, max_value=100.0)
        humidity = st.number_input("Влажность [%]", value=50.0, min_value=0.0, max_value=100.0)
        tvoc = st.number_input("TVOC [ppb]", value=500.0, min_value=0.0)
        eco2 = st.number_input("eCO2 [ppm]", value=400.0, min_value=0.0)
        h2 = st.number_input("Raw H2", value=13000.0, min_value=0.0)
        ethanol = st.number_input("Raw Ethanol", value=20000.0, min_value=0.0)
        pressure = st.number_input("Давление [hPa]", value=1013.0, min_value=800.0, max_value=1200.0)

    with col2:
        pm10 = st.number_input("PM1.0", value=10.0, min_value=0.0, max_value=500.0)
        pm25 = st.number_input("PM2.5", value=10.0)
        nc05 = st.number_input("NC0.5", value=1000.0, min_value=0.0)
        nc10 = st.number_input("NC1.0", value=800.0, min_value=0.0)
        nc25 = st.number_input("NC2.5", value=500.0, min_value=0.0)
        cnt = st.number_input("CNT", value=3000.0, min_value=0.0)

    input_data = pd.DataFrame([[temperature, humidity, tvoc, eco2, h2, ethanol, pressure, 
                              pm10, pm25, nc05, nc10, nc25, cnt]],
                            columns=['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 
                                    'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]',
                                    'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'CNT'])

    if st.button("Проверить вероятность пожара"):
        model = models[model_choice]
        
        if model_choice == 'CatBoost':
            prediction_proba  = model.predict_proba(input_data)[0][1]
            prediction_class = model.predict(input_data)[0]
        else:
            prediction_proba  = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else model.predict(input_data)[0]
            prediction_class = model.predict(input_data)[0] if not hasattr(model, "predict_proba") else (1 if prediction_proba > 0.5 else 0)

        probability = prediction_proba  * 100
        
        if probability > 50:
            st.error(f"🚨 ВЕРОЯТНОСТЬ ПОЖАРА: {probability:.1f}%")
            st.warning("Рекомендуется провести проверку!")
        else:
            st.success(f"✅ Вероятность пожара: {probability:.1f}%")
            st.info("Ситуация в норме")

        # Покажем аналогичные случаи из данных
        st.subheader("Аналогичные случаи в данных")
        similar_cases = data[
            (data['Temperature[C]'].between(temperature-5, temperature+5)) &
            (data['Humidity[%]'].between(humidity-10, humidity+10)) &
            (data['eCO2[ppm]'].between(eco2-100, eco2+100)) &
            (data['Fire Alarm'] == prediction_class)
        ]
        
        if not similar_cases.empty:
            st.dataframe(similar_cases[['Temperature[C]', 'Humidity[%]', 'eCO2[ppm]', 'Fire Alarm']].head(10))
        else:
            st.warning("В данных нет достаточно похожих случаев")