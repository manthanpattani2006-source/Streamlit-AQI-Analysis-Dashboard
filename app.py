import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="AQI Analysis and Prediction",
    page_icon="📈",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "INDIA_AQI_COMPLETE_20251126.csv"
MODEL_PATH = BASE_DIR / "aqi_random_forest_pipeline.pkl"


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(
            "Dataset not found. "
            "Please place INDIA_AQI_COMPLETE_20251126.csv "
            "in the application directory."
        )
        st.stop()

    # df = pd.read_csv(DATA_PATH)

    # df["Datetime"] = pd.to_datetime(
    #     df["Datetime"],
    #     errors="coerce"
    # )

    # df = df.dropna(
    #     subset=["Datetime"]
    # )

    # df["Date"] = df["Datetime"].dt.date

    # df = df.sort_values(
    #     ["Datetime", "City"]
    # ).reset_index(drop=True)


    # pollution_features = [
    #     "PM2_5_ugm3",
    #     "PM10_ugm3",
    #     "PM_Ratio",
    #     "CO_ugm3",
    #     "NO2_ugm3",
    #     "SO2_ugm3",
    #     "O3_ugm3",
    #     "Dust_ugm3",
    #     "NH3_ugm3",
    #     "AOD"
    # ]

    # weather_features = [
    #     "Temp_2m_C",
    #     "Humidity_Percent",
    #     "Dew_Point_C",
    #     "Wind_Speed_10m_kmh",
    #     "Wind_Gusts_kmh",
    #     "Wind_Stagnation",
    #     "Precipitation_mm",
    #     "Is_Raining",
    #     "Heavy_Rain",
    #     "Pressure_MSL_hPa",
    #     "Solar_Radiation_Wm2",
    #     "UV_Index",
    #     "Cloud_Cover_Percent",
    #     "Is_Daytime",
    #     "Temp_Inversion",
    #     "Inversion_Strength_C"
    # ]

    # location_features = [
    #     "City",
    #     "Latitude",
    #     "Longitude"
    # ]

    # time_features = [
    #     "Year",
    #     "Month",
    #     "Hour",
    #     "Day_of_Week",
    #     "Is_Weekend",
    #     "Season",
    #     "Time_of_Day",
    #     "Festival_Period",
    #     "Crop_Burning_Season",
    #     "Date"
    # ]

    # features = list(
    #     pollution_features
    #     + weather_features
    #     + location_features
    #     + time_features
    # )
    # df=df[features]

    # group_cols = [
    #     "City",
    #     "Date",
    #     "Time_of_Day"
    # ]

    # initial_numeric_cols = df.select_dtypes(
    #     include=["int64", "float64"]
    # ).columns.tolist()

    # initial_categorical_cols = df.select_dtypes(
    #     include=["object", "category"]
    # ).columns.tolist()


    # numeric_features_for_agg = [
    #     col
    #     for col in initial_numeric_cols
    #     if col not in group_cols
    # ]

    # categorical_features_for_agg = [
    #     col
    #     for col in initial_categorical_cols
    #     if col not in group_cols
    # ]

    # numeric_result = (
    #     df.groupby(group_cols)[numeric_features_for_agg]
    #     .mean()
    # )

    # categorical_result = (
    #     df.groupby(group_cols)[categorical_features_for_agg]
    #     .agg(
    #         lambda x:
    #         x.mode().iloc[0]
    #         if not x.mode().empty
    #         else np.nan
    #     )
    # )


    # df = pd.concat(
    #     [
    #         numeric_result,
    #         categorical_result
    #     ],
    #     axis=1
    # ).reset_index()

    # df = df.dropna(axis=1, how="all") 
    # df = df.drop_duplicates()

    df=pd.read_csv("aqi-data.csv")
 
    return df

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            "Trained model not found. "
            "Please place aqi_random_forest_pipeline.pkl "
            "in the application directory."
        )
        st.stop()

    return joblib.load(
        MODEL_PATH
    )


df = load_data()
model_pipeline = load_model()

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    elif aqi <= 200:
        return "Unhealthy"

    elif aqi <= 300:
        return "Very Unhealthy"

    else:
        return "Hazardous"



st.title("Indian Air Quality Index (AQI)")


analysis_tab, prediction_tab = st.tabs(
    [
        "Analysis",
        "Prediction"
    ]
)



# ANALYSIS TAB

with analysis_tab:
    st.header("Historical AQI Analysis")

    col1, col2 = st.columns(2)

    cities = sorted(
        df["City"].dropna().unique()
    )

    with col1:

        selected_city = st.selectbox(
            "Select City",
            cities
        )

    city_df = df[
        df["City"] == selected_city
    ].copy()

    years = sorted(
        city_df["Year"].dropna().unique()
    )

    with col2:

        selected_year = st.selectbox(
            "Select Year",
            years
        )

    filtered_df = city_df[
        city_df["Year"] == selected_year
    ].copy()


    if filtered_df.empty:

        st.warning(
            "No data available for this selection."
        )

    else:
        st.subheader("AQI Summary")

        avg_aqi = filtered_df["US_AQI"].mean()
        max_aqi = filtered_df["US_AQI"].max()
        min_aqi = filtered_df["US_AQI"].min()

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Average AQI",
                f"{avg_aqi:.1f}"
            )

        with c2:

            st.metric(
                "Maximum AQI",
                f"{max_aqi:.1f}"
            )

        with c3:

            st.metric(
                "Minimum AQI",
                f"{min_aqi:.1f}"
            )



        st.subheader(
            f"📈AQI Trend — {selected_city}, {selected_year}"
        )

        daily_aqi = (
            filtered_df
            .groupby("Date")["US_AQI"]
            .mean()
            .reset_index()
        )

        daily_aqi["Date"] = pd.to_datetime(
            daily_aqi["Date"]
        )

        fig_trend = px.line(
            daily_aqi,
            x="Date",
            y="US_AQI",
            markers=True
        )

        fig_trend.update_layout(
            xaxis_title="Date",
            yaxis_title="Average US AQI",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig_trend,
            use_container_width=True
        )



        st.subheader(
            "🌦️AQI Trend by Season"
        )

        seasonal = (
            filtered_df
            .groupby(
                ["Season", "Month"]
            )["US_AQI"]
            .mean()
            .reset_index()
        )

        seasonal = seasonal.sort_values(
            "Month"
        )

        fig_season = px.line(
            seasonal,
            x="Month",
            y="US_AQI",
            color="Season",
            markers=True
        )

        fig_season.update_layout(
            xaxis_title="Month",
            yaxis_title="Average US AQI"
        )

        st.plotly_chart(
            fig_season,
            use_container_width=True
        )



        st.subheader(
            "AQI Category Distribution"
        )

        filtered_df["AQI_Category_Display"] = (
            filtered_df["US_AQI"]
            .apply(get_aqi_category)
        )

        category_counts = (
            filtered_df["AQI_Category_Display"]
            .value_counts()
            .reset_index()
        )

        category_counts.columns = [
            "AQI_Category",
            "Count"
        ]

        fig_donut = px.pie(
            category_counts,
            names="AQI_Category",
            values="Count",
            hole=0.55
        )

        fig_donut.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        st.plotly_chart(
            fig_donut,
            use_container_width=True
        )



# PREDICTION TAB

with prediction_tab:

    st.header("AQI Prediction")

    st.info(
        "Enter the parameters"
    )


    st.subheader("Location")

    col1, col2, col3 = st.columns(3)

    with col1:

        prediction_city = st.selectbox(
            "City",
            cities,
            key="prediction_city"
        )

    city_info = (
        df[
            df["City"] == prediction_city
        ]
        .iloc[0]
    )

    with col2:

        latitude = st.number_input(
            "Latitude",
            value=float(city_info["Latitude"]),
            format="%.6f"
        )

    with col3:

        longitude = st.number_input(
            "Longitude",
            value=float(city_info["Longitude"]),
            format="%.6f"
        )


    st.subheader("Pollution Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:

        pm25 = st.number_input(
            "PM2.5 (µg/m³)",
            min_value=0.0,
            value=50.0
        )

        pm10 = st.number_input(
            "PM10 (µg/m³)",
            min_value=0.0,
            value=80.0
        )

        pm_ratio = st.number_input(
            "PM Ratio",
            min_value=0.0,
            value=0.60
        )

    with col2:

        co = st.number_input(
            "CO (µg/m³)",
            min_value=0.0,
            value=500.0
        )

        no2 = st.number_input(
            "NO2 (µg/m³)",
            min_value=0.0,
            value=30.0
        )

        so2 = st.number_input(
            "SO2 (µg/m³)",
            min_value=0.0,
            value=10.0
        )

    with col3:

        o3 = st.number_input(
            "O3 (µg/m³)",
            min_value=0.0,
            value=40.0
        )

        dust = st.number_input(
            "Dust (µg/m³)",
            min_value=0.0,
            value=20.0
        )

        aod = st.number_input(
            "AOD",
            min_value=0.0,
            value=0.5
        )



    st.subheader("Weather Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:

        dew_point = st.number_input(
            "Dew Point (°C)",
            value=15.0
        )

        is_raining = st.selectbox(
            "Is Raining?",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )

    with col2:

        solar_radiation = st.number_input(
            "Solar Radiation (W/m²)",
            min_value=0.0,
            value=200.0
        )

        is_daytime = st.selectbox(
            "Is Daytime?",
            [0, 1],
            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )



    st.subheader("Time Parameters")

    col1, col2, col3 = st.columns(3)

    with col1:

        month = st.selectbox(
            "Month",
            list(range(1, 13))
        )

    with col2:

        hour = st.slider(
            "Hour",
            min_value=0,
            max_value=23,
            value=12
        )

    with col3:

        time_of_day = st.selectbox(
            "Time of Day",
            [
                "Morning",
                "Afternoon",
                "Evening",
                "Night"
            ]
        )



    st.divider()

    predict_button = st.button(
        "Predict AQI",
        type="primary",
        use_container_width=True
    )


    if predict_button:

        input_data = pd.DataFrame(
            [{

                "PM2_5_ugm3": pm25,
                "PM10_ugm3": pm10,
                "PM_Ratio": pm_ratio,
                "CO_ugm3": co,
                "NO2_ugm3": no2,
                "SO2_ugm3": so2,
                "O3_ugm3": o3,
                "Dust_ugm3": dust,
                "AOD": aod,

                "Dew_Point_C": dew_point,
                "Is_Raining": is_raining,
                "Solar_Radiation_Wm2": solar_radiation,
                "Is_Daytime": is_daytime,

                "Latitude": latitude,
                "Longitude": longitude,

                "Month": month,
                "Hour": hour,


                "Time_of_Day": time_of_day,


                "Temp_2m_C": 25.0,
                "Humidity_Percent": 60.0,
                "Wind_Speed_10m_kmh": 10.0,
                "Wind_Gusts_kmh": 15.0,
                "Wind_Stagnation": 0,
                "Precipitation_mm": 0.0,
                "Heavy_Rain": 0,
                "Pressure_MSL_hPa": 1013.0,
                "UV_Index": 5.0,
                "Cloud_Cover_Percent": 40.0,
                "Temp_Inversion": 0,
                "Inversion_Strength_C": 0.0,

                "City": prediction_city,

                "Year": 2025,
                "Day_of_Week": 0,
                "Is_Weekend": 0,
                "Season": "Winter",
                "Festival_Period": 0,
                "Crop_Burning_Season": 0,
                "Date": ""
            }]
        )


        try:

            prediction = model_pipeline.predict(
                input_data
            )[0]

            prediction = max(
                0,
                prediction
            )

            category = get_aqi_category(
                prediction
            )


            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            st.success(
                "AQI predicted!"
            )

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Predicted US AQI",
                    f"{prediction:.2f}"
                )

            with c2:

                st.metric(
                    "AQI Category",
                    category
                )


            if prediction <= 50:

                st.success(
                    "🟢 Air Quality Looks Good"
                )

            elif prediction <= 100:

                st.info(
                    "🟡 Air Quality is Moderate"
                )

            elif prediction <= 150:

                st.warning(
                    "🟠 Air Quality may not Appropriate for Sensitive Groups"
                )

            elif prediction <= 200:

                st.warning(
                    "🔴 Air Quality is Unhealthy (Avoid Unnecessary Planning)"
                )

            elif prediction <= 300:

                st.error(
                    "🟣 Air Quality is Very Unhealthy (May need Improvement)"
                )

            else:

                st.error(
                    "⚫ Air Quality is Hazardous (Use Appropriate Precautions)"
                )


        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )