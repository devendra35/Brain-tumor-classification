
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



# PAGE CONFIGURATION


st.set_page_config(
    page_title="Brain MRI  Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)



# CONFIGURATION


CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

CLASSIFICATION_SIZE = (300, 300)
SEGMENTATION_SIZE = (128, 128)

CLASSIFIER_PATH = "models/best_efficientnetb3_final.keras"
SEGMENTER_PATH = "models/best_light_unet_128_final.keras"



# CUSTOM CSS


st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 20px;
    }

    .small-text {
        font-size: 14px;
        opacity: 0.7;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# LOAD MODELS


@st.cache_resource
def load_models():

    classifier = tf.keras.models.load_model(
        CLASSIFIER_PATH
    )

    segmenter = tf.keras.models.load_model(
        SEGMENTER_PATH,
        compile=False
    )

    return classifier, segmenter



# MODEL LOADING


try:

    classifier, segmenter = load_models()

except Exception as e:

    st.error(" Unable to load the AI models.")

    st.code(
        str(e),
        language="text"
    )

    st.info(
        "Make sure the following files exist:\n\n"
        "models/best_efficientnetb3_final.keras\n\n"
        "models/best_light_unet_128_final.keras"
    )

    st.stop()



# IMAGE PREPROCESSING


def prepare_classification_image(image):

    image_array = np.array(
        image.convert("RGB")
    )

    image_array = tf.image.resize(
        image_array,
        CLASSIFICATION_SIZE
    )

    image_array = tf.cast(
        image_array,
        tf.float32
    )

    image_array = tf.expand_dims(
        image_array,
        axis=0
    )

    return image_array


def prepare_segmentation_image(image):

    image_array = np.array(
        image.convert("RGB")
    )

    image_array = tf.image.resize(
        image_array,
        SEGMENTATION_SIZE
    )

    image_array = tf.cast(
        image_array,
        tf.float32
    )

    image_array = image_array / 255.0

    image_array = tf.expand_dims(
        image_array,
        axis=0
    )

    return image_array



# CREATE TUMOR OVERLAY


def create_overlay(
    original_array,
    mask
):

    overlay = original_array.copy()

    tumor_pixels = mask == 1

    if np.any(tumor_pixels):

        red = np.array(
            [255, 0, 0],
            dtype=np.float32
        )

        overlay[tumor_pixels] = (
            0.60 * overlay[tumor_pixels]
            + 0.40 * red
        ).astype(np.uint8)

    return overlay



# ANALYZE MRI


def analyze_mri(
    image,
    segmentation_threshold=0.5
):

    original = image.convert("RGB")

    original_array = np.array(
        original
    )

   
    # CLASSIFICATION
    
    classification_input = (
        prepare_classification_image(
            original
        )
    )

    probabilities = classifier.predict(
        classification_input,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = (
        CLASS_NAMES[predicted_index]
    )

    confidence = float(
        probabilities[predicted_index]
    )

  
    # NO TUMOR
    

    if predicted_class == "no_tumor":

        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "mask": None,
            "overlay": None,
            "raw_mask": None,
            "tumor_pixels": 0,
            "tumor_percentage": 0.0,
            "max_probability": 0.0
        }

   
    # SEGMENTATION
    

    segmentation_input = (
        prepare_segmentation_image(
            original
        )
    )

    raw_mask = segmenter.predict(
        segmentation_input,
        verbose=0
    )[0, :, :, 0]

    raw_mask = np.asarray(
        raw_mask,
        dtype=np.float32
    )

    max_probability = float(
        np.max(raw_mask)
    )

  
    # THRESHOLD
    

    binary_mask = (
        raw_mask >= segmentation_threshold
    ).astype(np.uint8)

    
    # RESIZE MASK
    

    height, width = (
        original_array.shape[:2]
    )

    resized_mask = tf.image.resize(
        binary_mask[..., np.newaxis],
        (height, width),
        method="nearest"
    ).numpy()[:, :, 0]

    resized_mask = (
        resized_mask > 0.5
    ).astype(np.uint8)

    
    # OVERLAY
  

    overlay = create_overlay(
        original_array,
        resized_mask
    )

  
    # STATISTICS
    

    tumor_pixels = int(
        np.sum(resized_mask)
    )

    total_pixels = (
        height * width
    )

    tumor_percentage = (
        tumor_pixels / total_pixels
    ) * 100

    return {
        "class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "mask": resized_mask,
        "overlay": overlay,
        "raw_mask": raw_mask,
        "tumor_pixels": tumor_pixels,
        "tumor_percentage": tumor_percentage,
        "max_probability": max_probability
    }



# HEADER


st.markdown(
    '<div class="main-title">🧠 Brain MRI  Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Trained Model-assisted brain tumor classification and tumor localization
    using EfficientNetB3 + U-Net
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()



# SIDEBAR

with st.sidebar:

    st.header("⚙️ Model Information")

    st.markdown(
        """
        ### Classification

        **EfficientNetB3**

        Input: **300 × 300**

        ### Segmentation

        **Lightweight U-Net**

        Input: **128 × 128**

        ### Classes

        🟣 Glioma  
        🔵 Meningioma  
        ⚪ No Tumor  
        🟢 Pituitary
        """
    )

    st.divider()

    st.header(" Model Performance")

    st.metric(
        "Classification Accuracy",
        "88.33%"
    )

    st.metric(
        "Segmentation Dice",
        "70.72%"
    )

    st.metric(
        "Segmentation IoU",
        "59.91%"
    )

    st.divider()

    st.info(
        "This application is intended for research and "
        "educational purposes only. It is not a medical "
        "diagnostic system."
    )



# IMAGE UPLOAD


st.subheader("📤 Upload Brain MRI")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help="Upload a brain MRI image for AI-assisted analysis."
)



# MAIN APPLICATION


if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception:

        st.error(
            "❌ Unable to read the uploaded image."
        )

        st.stop()

   
    # IMAGE INFORMATION
   

    st.subheader("🖼️ Uploaded MRI")

    col1, col2 = st.columns(
        [1.4, 1]
    )

    with col1:

        st.image(
            image,
            caption="Original MRI",
            use_container_width=True
        )

    with col2:

        st.markdown(
            """
            <div class="result-box">
            """,
            unsafe_allow_html=True
        )

        st.write("### Image Information")

        st.write(
            f"**Width:** {image.width} px"
        )

        st.write(
            f"**Height:** {image.height} px"
        )

        st.write(
            f"**Mode:** {image.mode}"
        )

        st.write(
            f"**Format:** {uploaded_file.type}"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    st.divider()

  
    # SEGMENTATION THRESHOLD
   

    st.subheader(
        " Segmentation Settings"
    )

    threshold = st.slider(
        "Tumor mask threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "Higher values produce a stricter tumor mask. "
            "Lower values produce a more sensitive mask."
        )
    )

    st.divider()


    # ANALYZE BUTTON
   

    if st.button(
        "🔍 Analyze MRI",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "🧠 Model is analyzing the MRI..."
        ):

            result = analyze_mri(
                image,
                segmentation_threshold=threshold
            )

   
        # CLASSIFICATION RESULT
       

        st.subheader(
            "📋 Classification Result"
        )

        result_col1, result_col2 = st.columns(2)

        display_class = (
            result["class"]
            .replace("_", " ")
            .title()
        )

        with result_col1:

            st.metric(
                "Predicted Class",
                display_class
            )

        with result_col2:

            st.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

        
        # PROBABILITY CHART
       

        st.subheader(
            "📊 Class Probabilities"
        )

        probability_values = (
            result["probabilities"] * 100
        )

        fig, ax = plt.subplots(
            figsize=(9, 4)
        )

        labels = [
            name.replace(
                "_",
                " "
            ).title()
            for name in CLASS_NAMES
        ]

        ax.bar(
            labels,
            probability_values
        )

        ax.set_ylabel(
            "Probability (%)"
        )

        ax.set_ylim(
            0,
            100
        )

        ax.set_title(
            "Classification Probability"
        )

        for index, value in enumerate(
            probability_values
        ):

            ax.text(
                index,
                value + 1,
                f"{value:.1f}%",
                ha="center"
            )

        plt.xticks(
            rotation=15
        )

        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)

       
        # NO TUMOR RESULT
       

        if result["class"] == "no_tumor":

            st.success(
                " The classifier predicted: No Tumor"
            )

            st.info(
                "Because the classifier predicted the "
                "No Tumor class, the U-Net segmentation "
                "stage was not performed."
            )

       
        # TUMOR RESULT
        

        else:

            st.warning(
                f"⚠️ Tumor class predicted: "
                f"{display_class}"
            )

            st.subheader(
                " Tumor Localization"
            )

           
            # SEGMENTATION STATUS
            

            status_col1, status_col2, status_col3 = (
                st.columns(3)
            )

            with status_col1:

                st.metric(
                    "Mask Max Probability",
                    f"{result['max_probability']:.4f}"
                )

            with status_col2:

                st.metric(
                    "Tumor Pixels",
                    f"{result['tumor_pixels']:,}"
                )

            with status_col3:

                st.metric(
                    "Mask Area",
                    f"{result['tumor_percentage']:.2f}%"
                )

            # MASK WARNING
            

            if result["tumor_pixels"] == 0:

                st.error(
                    " The U-Net produced an empty mask "
                    "at the selected threshold."
                )

                st.info(
                    "Try lowering the segmentation threshold "
                    "using the slider above."
                )

            else:

                st.success(
                    " Tumor region detected by U-Net."
                )

          
            # VISUALIZATION
           

            image_col1, image_col2, image_col3 = (
                st.columns(3)
            )

            with image_col1:

                st.image(
                    image,
                    caption="Original MRI",
                    use_container_width=True
                )

            with image_col2:

                st.image(
                    result["mask"] * 255,
                    caption="Predicted Tumor Mask",
                    use_container_width=True,
                    clamp=True
                )

            with image_col3:

                st.image(
                    result["overlay"],
                    caption="Tumor Localization",
                    use_container_width=True
                )

           
            # RAW MASK VISUALIZATION
           

            with st.expander(
                "🔬 View Raw U-Net Probability Map"
            ):

                st.image(
                    result["raw_mask"],
                    caption=(
                        "Raw segmentation probability "
                        "(brighter = higher probability)"
                    ),
                    use_container_width=True,
                    clamp=True
                )

       
        # FINAL DISCLAIMER
     

        st.divider()

        st.warning(
            "⚠️ Research/Educational Use Only — "
            "The model-assisted result is not a medical diagnosis "
            "and should not replace assessment by a qualified "
            "medical professional."
        )



# FOOTER


st.divider()

st.caption(
    "🧠 Brain MRI  Analyzer | "
    "Developed by Devendra. | "
)