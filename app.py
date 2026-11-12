
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.ndimage import gaussian_filter

from src.inference.inference_engine import ForgeVisionInference


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ForgeVision | Industrial Defect Detection",
    page_icon="🔍",
    layout="wide"
)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_inference_engine():
    return ForgeVisionInference()


# --------------------------------------------------
# VISUALIZATION
# --------------------------------------------------

def generate_heatmap(anomaly_map):

    # Smooth the anomaly map for clearer visualization
    smoothed_map = gaussian_filter(
        anomaly_map,
        sigma=2.0
    )

    # Percentile-based normalization
    lower = np.percentile(smoothed_map, 1)
    upper = np.percentile(smoothed_map, 99.5)

    normalized_map = np.clip(
        (smoothed_map - lower) / (upper - lower + 1e-8),
        0,
        1
    )

    colormap = plt.get_cmap("inferno")

    heatmap = colormap(normalized_map)[:, :, :3]

    heatmap = (heatmap * 255).astype(np.uint8)

    return Image.fromarray(heatmap)


def generate_overlay(original_image, heatmap):

    original = original_image.resize((256, 256))

    original_array = np.asarray(original).astype(np.float32)
    heatmap_array = np.asarray(heatmap).astype(np.float32)

    overlay = (
        0.55 * original_array
        + 0.45 * heatmap_array
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    return Image.fromarray(overlay)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🔍 ForgeVision")

st.subheader("AI-Powered Industrial Defect Detection")

st.write(
    "An unsupervised visual anomaly detection system "
    "using Convolutional Autoencoder, PaDiM, and PatchCore."
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("ForgeVision")

st.sidebar.markdown("### Detection Models")

st.sidebar.write("1. Convolutional Autoencoder")
st.sidebar.write("2. PaDiM")
st.sidebar.write("3. PatchCore")

st.sidebar.divider()

st.sidebar.markdown("### Dataset")

st.sidebar.write("MVTec AD — Bottle Category")

st.sidebar.markdown("### Calibration")

st.sidebar.write("42 normal images")
st.sidebar.write("95th-percentile thresholds")

st.sidebar.divider()

st.sidebar.caption(
    "Research prototype. Predictions are based on "
    "calibration thresholds from normal training images."
)


# --------------------------------------------------
# LOAD ENGINE
# --------------------------------------------------

try:

    with st.spinner("Loading detection models..."):
        engine = load_inference_engine()

    st.success("All detection models are ready.")

except Exception as error:

    st.error(f"Model loading failed: {error}")
    st.stop()


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

st.markdown("### Upload an Image")

uploaded_file = st.file_uploader(
    "Upload a bottle image for defect inspection",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    original_image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:

        st.markdown("#### Uploaded Image")

        st.image(
            original_image,
            use_container_width=True
        )

    with col2:

        st.markdown("#### Image Information")

        st.write(f"Width: {original_image.width} pixels")
        st.write(f"Height: {original_image.height} pixels")
        st.write(f"Format: {uploaded_file.type}")

    st.divider()

    if st.button(
        "Analyze Image",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner("Analyzing image with all three models..."):

                prediction_output = engine.predict(original_image)

                results = prediction_output["results"]

            st.success("Image analysis completed.")

            # ------------------------------------------
            # MODEL PREDICTION SUMMARY
            # ------------------------------------------

            st.markdown("## Detection Results")

            model_names = {
                "autoencoder": "Convolutional Autoencoder",
                "padim": "PaDiM",
                "patchcore": "PatchCore"
            }

            columns = st.columns(3)

            for column, model_key in zip(
                columns,
                model_names.keys()
            ):

                result = results[model_key]

                with column:

                    st.markdown(
                        f"#### {model_names[model_key]}"
                    )

                    if result["prediction"] == "Normal":

                        st.success("NORMAL")

                    else:

                        st.error("DEFECTIVE")

                    st.metric(
                        "Anomaly Score",
                        f"{result['score']:.6f}"
                    )

                    st.metric(
                        "Threshold",
                        f"{result['threshold']:.6f}"
                    )

                    st.caption(
                        f"Inference: {result['inference_ms']:.2f} ms"
                    )

            st.divider()

            # ------------------------------------------
            # DETAILED ANOMALY VISUALIZATION
            # ------------------------------------------

            st.markdown("## Anomaly Localization")

            tabs = st.tabs([
                "Autoencoder",
                "PaDiM",
                "PatchCore"
            ])

            for tab, model_key in zip(
                tabs,
                model_names.keys()
            ):

                result = results[model_key]

                heatmap = generate_heatmap(
                    result["anomaly_map"]
                )

                overlay = generate_overlay(
                    original_image,
                    heatmap
                )

                with tab:

                    st.markdown(
                        f"### {model_names[model_key]}"
                    )

                    if result["prediction"] == "Normal":

                        st.success(
                            "Prediction: Normal"
                        )

                    else:

                        st.error(
                            "Prediction: Defective"
                        )

                    image_col1, image_col2, image_col3 = st.columns(3)

                    with image_col1:

                        st.markdown("**Original Image**")

                        st.image(
                            original_image,
                            use_container_width=True
                        )

                    with image_col2:

                        st.markdown("**Anomaly Heatmap**")

                        st.image(
                            heatmap,
                            use_container_width=True
                        )

                    with image_col3:

                        st.markdown("**Defect Overlay**")

                        st.image(
                            overlay,
                            use_container_width=True
                        )

                    st.caption(
                        "Heatmap colors represent relative anomaly intensity. "
                        "Visualization smoothing does not change the prediction."
                    )

            st.divider()

            st.caption(
                "ForgeVision | Unsupervised Industrial Anomaly Detection"
            )

        except Exception as error:

            st.error(f"Image analysis failed: {error}")

else:

    st.info(
        "Upload a bottle image to begin defect detection."
    )
