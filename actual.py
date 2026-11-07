


import joblib
import pandas as pd
import streamlit as st

from pathlib import Path

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Crop Prediction System",
    
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "crop_prediction.pkl"
IMAGE_DIR = BASE_DIR / "images"

MODEL_FEATURES = [
    "season",
    "soil_type",
    "irrigation_source",
    "previous_crop",
    "pH",
    "rainfall_mm",
    "avg_temp_c",
    "humidity_pct"
]

CROP_LABELS = [
    "Bengalgram",
    "Chilli",
    "Cotton",
    "Groundnut",
    "Jowar",
    "Maize",
    "Paddy",
    "Redgram",
    "Soyabean"
]

CROP_IMAGES = {
    "Bengalgram": "bengalgram.jpg",
    "Chilli": "chilli.jpg",
    "Cotton": "cotton.jpg",
    "Groundnut": "groundnut.jpg",
    "Jowar": "jowar.jpg",
    "Maize": "maize.jpg",
    "Paddy": "paddy.jpg",
    "Redgram": "redgram.jpg",
    "Soyabean": "soyabean.jpg"
}


# =========================================================
# 2. CUSTOM THEME
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #071713;
        color: #F0FDF4;
    }

    [data-testid="stSidebar"] {
        background-color: #102B24;
    }

    h1, h2, h3 {
        color: #86EFAC !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3. HEADER
# =========================================================

st.title("CROP PREDICTION SYSTEM")

st.caption(
    "Smart Crop Recommendations Using Machine Learning"
)

st.divider()


# =========================================================
# 4. LOAD MODEL
# =========================================================

@st.cache_resource
def load_model_package():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    package = joblib.load(MODEL_PATH)

    if isinstance(package, dict):

        if "model" not in package:
            raise KeyError(
                "The saved model dictionary does not contain "
                "'model'. Available keys: "
                + ", ".join(package.keys())
            )

        model = package["model"]

    else:
        model = package
        package = {"model": model}

    return package, model


# =========================================================
# 5. CROP NAME CONVERSION
# =========================================================

def get_crop_name(prediction, package, model):

    if isinstance(prediction, str):
        if not prediction.strip().lstrip("-").isdigit():
            return prediction.strip()

    target_encoder = package.get("target_encoder")

    if target_encoder is not None:

        return str(
            target_encoder.inverse_transform(
                [int(prediction)]
            )[0]
        )

    classes = getattr(model, "classes_", None)

    if classes is not None:

        for class_value, crop_name in zip(
            classes, CROP_LABELS
        ):

            if prediction == class_value:
                return crop_name

    try:
        index = int(prediction)

        if 0 <= index < len(CROP_LABELS):
            return CROP_LABELS[index]

    except (TypeError, ValueError, OverflowError):
        pass

    raise ValueError(
        f"Unable to identify crop for prediction: {prediction}"
    )


# =========================================================
# 6. SIDEBAR INPUT FORM
# =========================================================

with st.sidebar:

    st.header(" Crop Prediction")

    st.write(
        "Enter your field, soil and weather information."
    )

    with st.form("crop_prediction_form"):

        st.subheader(" Field Information")

        season = st.selectbox(
            "Season",
            ["Kharif", "Rabbi", "Summer"]
        )

        soil_type = st.selectbox(
            "Soil Type",
            [
                "BlackCotton",
                "Alluvial",
                "Redsandyloam",
                "RedClay",
                "Sandy",
                "Loamy"
            ]
        )

        irrigation_source = st.selectbox(
            "Irrigation Source",
            [
                "BoreWell",
                "Canal",
                "RainFed",
                "Open Well",
                "Tank"
            ]
        )

        previous_crop = st.selectbox(
            "Previous Crop",
            [
                "Paddy",
                "Cotton",
                "Maize",
                "Chilli",
                "Groundnut",
                "Soyabean",
                "Jowar",
                "RedGram",
                "BengalGram",
                "Fallow"
            ]
        )

        st.divider()

        st.subheader("Soil Information")

        ph = st.number_input(
            "Soil pH",
            min_value=0.0,
            max_value=14.0,
            value=7.0,
            step=0.1
        )

        st.divider()

        st.subheader(" Weather Information")

        rainfall_mm = st.number_input(
            "Rainfall (mm)",
            min_value=0.0,
            value=120.0,
            step=1.0
        )

        avg_temp_c = st.number_input(
            "Average Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=28.5,
            step=0.5
        )

        humidity_pct = st.number_input(
            "Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=70.0,
            step=1.0
        )

        predict_clicked = st.form_submit_button(
            "🌿 Predict Crop",
            use_container_width=True
        )


# =========================================================
# 7. MAIN DASHBOARD
# =========================================================

left, right = st.columns([1, 1.2], gap="large")


# =========================================================
# 8. FIELD SUMMARY
# =========================================================

with left:

    st.subheader(" Field Summary")

    with st.container(border=True):

        st.markdown("###  Field Information")

        st.write(f"**Season:** {season}")

        st.write(f"**Soil Type:** {soil_type}")

        st.write(f"**Irrigation:** {irrigation_source}")

        st.write(f"**Previous Crop:** {previous_crop}")

        st.divider()

        st.markdown("### Soil and Weather Conditions")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Soil pH", f"{ph:.1f}")

            st.metric("Rainfall", f"{rainfall_mm:.1f} mm")

        with col2:
            st.metric("Temperature", f"{avg_temp_c:.1f} °C")

            st.metric("Humidity", f"{humidity_pct:.1f}%")


    st.subheader(" Model Information")

    st.info(
        """
        **Total Features: 8**

        - Categorical features: 4
        - Numerical features: 4

        The machine learning model uses these inputs
        to recommend a suitable crop.
        """
    )


# =========================================================
# 9. PREDICTION RESULT
# =========================================================

with right:

    st.subheader(" Prediction Result")

    if predict_clicked:

        try:

            package, model = load_model_package()

            # Create input DataFrame
            input_data = pd.DataFrame(
                [[
                    season,
                    soil_type,
                    irrigation_source,
                    previous_crop,
                    ph,
                    rainfall_mm,
                    avg_temp_c,
                    humidity_pct
                ]],
                columns=MODEL_FEATURES
            )

            # Check saved model feature names
            saved_features = package.get("model_features")

            if saved_features is not None:

                missing = [
                    col for col in saved_features
                    if col not in input_data.columns
                ]

                if missing:
                    raise ValueError(
                        f"Missing model features: {missing}. "
                        "Your model may require more than 8 columns."
                    )

                input_data = input_data[saved_features]

            # Model prediction
            prediction = model.predict(input_data)[0]

            crop_name = get_crop_name(
                prediction,
                package,
                model
            )

            # Display recommended crop
            with st.container(border=True):

                st.success("Crop Prediction Successful!")

                st.markdown(
                    f"#  {crop_name}"
                )

                st.write("**Recommended Crop**")

                image_filename = CROP_IMAGES.get(crop_name)

                if image_filename:

                    image_path = IMAGE_DIR / image_filename

                    if image_path.exists():

                        st.image(
                            str(image_path),
                            caption=f"{crop_name} Crop",
                            use_container_width=True
                        )

                    else:

                        st.info(
                            f"Image not found: {image_path}"
                        )

            # Confidence scores
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                classes = model.classes_

                st.subheader(" Top 3 Crop Confidence Scores")

                top_predictions = sorted(
                    zip(classes, probabilities),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]

                for class_value, probability in top_predictions:

                    label = get_crop_name(
                        class_value,
                        package,
                        model
                    )

                    st.progress(
                        float(probability),
                        text=f"{label}: {probability:.1%}"
                    )

            st.warning(
                "This is a machine-learning-based recommendation. "
                "Actual crop suitability also depends on local "
                "agricultural conditions."
            )

        except Exception as error:

            st.error("Unable to predict the crop.")

            st.exception(error)

    else:

        with st.container(border=True):

            st.markdown(
                """
                # 
                ### Your prediction will appear here
                """
            )

            st.info(
                "Complete the sidebar inputs and click "
                "Predict Crop."
            )


# =========================================================
# 10. FOOTER
# =========================================================

st.divider()

st.caption(
    " Crop Prediction System | Machine Learning Project"
)

# -------------------------------------------------------------------------------------------------------------------



# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# from pathlib import Path

# # ==================================================
# # 1. PAGE CONFIGURATION
# # ==================================================

# st.set_page_config(
#     page_title="Crop Prediction System",
#     page_icon="🌱",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ==================================================
# # 2. FILE PATH
# # ==================================================

# BASE_DIR = Path(__file__).resolve().parent
# MODEL_PATH = BASE_DIR / "crop_prediction.pkl"

# # ==================================================
# # 3. MODEL FEATURES
# # ==================================================

# MODEL_FEATURES = [
#     "season",
#     "soil_type",
#     "irrigation_source",
#     "previous_crop",
#     "pH",
#     "rainfall_mm",
#     "avg_temp_c",
#     "humidity_pct"
# ]

# CROP_LABELS = [
#     "Bengalgram",
#     "Chilli",
#     "Cotton",
#     "Groundnut",
#     "Jowar",
#     "Maize",
#     "Paddy",
#     "Redgram",
#     "Soyabean"
# ]

# # ==================================================
# # 4. LOAD MODEL PACKAGE
# # ==================================================

# @st.cache_resource
# def load_model_package():

#     if not MODEL_PATH.exists():
#         raise FileNotFoundError(
#             f"Model file not found: {MODEL_PATH}"
#         )

#     package = joblib.load(MODEL_PATH)

#     if isinstance(package, dict):

#         if "model" not in package:
#             raise KeyError(
#                 f"Missing 'model' key. Available keys: {list(package.keys())}"
#             )

#         model = package["model"]

#     else:

#         model = package
#         package = {"model": model}

#     return package, model


# # ==================================================
# # 5. CROP NAME DECODING
# # ==================================================

# def get_crop_name(prediction, package, model):

#     # A. Prediction is already a crop name
#     if isinstance(prediction, (str, np.str_)):

#         prediction = str(prediction).strip()

#         if prediction in CROP_LABELS:
#             return prediction

#         if not prediction.lstrip("-").isdigit():
#             return prediction

#     # B. Find saved target encoder
#     encoder_keys = [
#         "target_encoder",
#         "label_encoder",
#         "le",
#         "crop_encoder"
#     ]

#     for key in encoder_keys:

#         encoder = package.get(key)

#         if encoder is not None and hasattr(
#             encoder, "inverse_transform"
#         ):

#             return str(
#                 encoder.inverse_transform(
#                     [prediction]
#                 )[0]
#             )

#     # C. Use explicitly saved crop labels
#     saved_labels = package.get(
#         "crop_labels",
#         package.get("class_names")
#     )

#     if saved_labels is not None:

#         class_value = int(prediction)

#         # Use saved class order only when it represents
#         # the original encoded class order.
#         if 0 <= class_value < len(saved_labels):
#             return str(saved_labels[class_value])

#     # D. Use model classes
#     classes = getattr(model, "classes_", None)

#     if classes is not None:

#         for class_value in classes:

#             if prediction == class_value:

#                 # If model classes are crop names
#                 if isinstance(class_value, (str, np.str_)):
#                     return str(class_value)

#                 # If classes are encoded as 0 to 8,
#                 # use the known label order only when
#                 # it matches the training LabelEncoder.
#                 if (
#                     isinstance(class_value, (int, np.integer))
#                     and np.array_equal(
#                         classes,
#                         np.arange(len(CROP_LABELS))
#                     )
#                 ):
#                     return CROP_LABELS[int(class_value)]

#     raise ValueError(
#         f"Cannot identify crop for prediction: {prediction!r}. "
#         "The saved model package needs its original target encoder "
#         "or the correct class label mapping."
#     )


# # ==================================================
# # 6. PREDICTION CONFIDENCE
# # ==================================================

# def get_prediction_probabilities(input_data, model, package):

#     if not hasattr(model, "predict_proba"):
#         return None, None

#     probabilities = model.predict_proba(input_data)[0]

#     classes = getattr(model, "classes_", None)

#     if classes is None:
#         return None, probabilities

#     crop_names = []

#     for class_value in classes:

#         crop_names.append(
#             get_crop_name(class_value, package, model)
#         )

#     return crop_names, probabilities


# # ==================================================
# # 7. HEADER
# # ==================================================

# st.title("🌱 Crop Prediction System")

# st.write(
#     "An intelligent agricultural recommendation system "
#     "that predicts suitable crops based on soil and "
#     "environmental conditions."
# )

# st.divider()


# # ==================================================
# # 8. SIDEBAR INPUT FORM
# # ==================================================

# st.sidebar.title("🌾 Crop Details")

# st.sidebar.write(
#     "Enter your agricultural field information."
# )

# with st.sidebar:

#     with st.form("crop_prediction_form"):

#         st.subheader("Field Information")

#         season = st.selectbox(
#             "Season",
#             ["Kharif", "Rabbi", "Summer"]
#         )

#         soil_type = st.selectbox(
#             "Soil Type",
#             [
#                 "BlackCotton",
#                 "Alluvial",
#                 "Redsandyloam",
#                 "RedClay",
#                 "Sandy",
#                 "Loamy"
#             ]
#         )

#         irrigation_source = st.selectbox(
#             "Irrigation Source",
#             [
#                 "BoreWell",
#                 "Canal",
#                 "RainFed",
#                 "Open Well",
#                 "Tank"
#             ]
#         )

#         previous_crop = st.selectbox(
#             "Previous Crop",
#             [
#                 "Paddy",
#                 "Cotton",
#                 "Maize",
#                 "Chilli",
#                 "Groundnut",
#                 "Soyabean",
#                 "Jowar",
#                 "RedGram",
#                 "BengalGram",
#                 "Fallow"
#             ]
#         )

#         st.divider()

#         st.subheader("Environmental Conditions")

#         ph = st.number_input(
#             "Soil pH",
#             min_value=0.0,
#             max_value=14.0,
#             value=7.0,
#             step=0.1
#         )

#         rainfall_mm = st.number_input(
#             "Rainfall (mm)",
#             min_value=0.0,
#             value=120.0,
#             step=1.0
#         )

#         avg_temp_c = st.number_input(
#             "Average Temperature (°C)",
#             min_value=-10.0,
#             max_value=60.0,
#             value=28.5,
#             step=0.5
#         )

#         humidity_pct = st.number_input(
#             "Humidity (%)",
#             min_value=0.0,
#             max_value=100.0,
#             value=70.0,
#             step=1.0
#         )

#         predict_clicked = st.form_submit_button(
#             "🌿 Predict Crop",
#             use_container_width=True
#         )


# # ==================================================
# # 9. THREE-COLUMN DASHBOARD
# # ==================================================

# left, center, right = st.columns(
#     [1, 1.3, 1],
#     gap="large"
# )


# # ==================================================
# # 10. LEFT COLUMN - FIELD SUMMARY
# # ==================================================

# with left:

#     st.subheader("📋 Field Summary")

#     st.write("Selected agricultural details")

#     st.metric("Soil pH", f"{ph:.1f}")

#     st.metric("Rainfall", f"{rainfall_mm:.1f} mm")

#     st.metric("Temperature", f"{avg_temp_c:.1f} °C")

#     st.metric("Humidity", f"{humidity_pct:.1f}%")

#     st.divider()

#     st.write("**Season:**", season)

#     st.write("**Soil Type:**", soil_type)

#     st.write("**Irrigation:**", irrigation_source)

#     st.write("**Previous Crop:**", previous_crop)


# # ==================================================
# # 11. CENTER COLUMN - PREDICTION RESULT
# # ==================================================

# with center:

#     st.subheader("🌱 Crop Prediction")

#     if predict_clicked:

#         try:

#             package, model = load_model_package()

#             input_data = pd.DataFrame(
#                 [[
#                     season,
#                     soil_type,
#                     irrigation_source,
#                     previous_crop,
#                     ph,
#                     rainfall_mm,
#                     avg_temp_c,
#                     humidity_pct
#                 ]],
#                 columns=MODEL_FEATURES
#             )

#             # Check whether model expects these features
#             expected_features = package.get(
#                 "model_features"
#             )

#             if expected_features is None:

#                 expected_features = getattr(
#                     model,
#                     "feature_names_in_",
#                     None
#                 )

#             if expected_features is not None:

#                 expected_features = list(
#                     expected_features
#                 )

#                 if expected_features != MODEL_FEATURES:

#                     raise ValueError(
#                         "Model feature mismatch. "
#                         "Your model must be trained using exactly "
#                         "these 8 input features in the same order. "
#                         f"Model expects: {expected_features}"
#                     )

#             # Predict
#             prediction = model.predict(input_data)[0]

#             # Decode predicted crop
#             crop_name = get_crop_name(
#                 prediction,
#                 package,
#                 model
#             )

#             st.success("Prediction completed successfully!")

#             st.divider()

#             st.write("### Recommended Crop")

#             st.title(f"🌾 {crop_name}")

#             st.write(
#                 "This crop was recommended based on "
#                 "the selected agricultural conditions."
#             )

#             st.session_state["last_prediction"] = crop_name

#             # Prediction confidence
#             crop_names, probabilities = (
#                 get_prediction_probabilities(
#                     input_data,
#                     model,
#                     package
#                 )
#             )

#             if probabilities is not None:

#                 st.divider()

#                 st.write("### Prediction Confidence")

#                 if crop_names is not None:

#                     sorted_indices = np.argsort(
#                         probabilities
#                     )[::-1]

#                     for index in sorted_indices:

#                         st.write(
#                             f"**{crop_names[index]}:** "
#                             f"{probabilities[index] * 100:.2f}%"
#                         )

#                         st.progress(
#                             float(probabilities[index])
#                         )

#                 else:

#                     st.info(
#                         "Confidence values are available, "
#                         "but crop labels could not be mapped."
#                     )

#         except Exception as error:

#             st.error(f"Prediction Error: {error}")

#     else:

#         st.info(
#             "Enter your field details in the sidebar "
#             "and click Predict Crop."
#         )

#         st.write("### Your prediction will appear here.")

#         st.write(
#             "The recommended crop and prediction "
#             "confidence will be displayed in this section."
#         )


# # ==================================================
# # 12. RIGHT COLUMN - MODEL INFORMATION
# # ==================================================

# with right:

#     st.subheader("🤖 Model Information")

#     st.write("**Application:** Crop Prediction")

#     st.write("**Model Type:** ML Classifier")

#     st.write("**Input Features:** 8")

#     st.write("**Crop Categories:** 9")

#     st.write("**Prediction Type:** Multiclass Classification")

#     st.divider()

#     st.write("### Input Features")

#     for feature in MODEL_FEATURES:

#         st.write(f"• {feature}")

#     st.divider()

#     st.write("### Prediction Status")

#     if "last_prediction" in st.session_state:

#         st.success(
#             f"Last Prediction: "
#             f"{st.session_state['last_prediction']}"
#         )

#     else:

#         st.info("No prediction made yet.")


# # ==================================================
# # 13. FOOTER
# # ==================================================

# st.divider()

# st.caption(
#     "Crop Prediction System | Machine Learning Based "
#     "Agricultural Recommendation"
# )
