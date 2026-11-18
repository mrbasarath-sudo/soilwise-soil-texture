"""Local web app for experimental soil texture composition predictions."""

from pathlib import Path
from io import BytesIO

import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT_DIR / "soil_model_output" / "soil_texture_model.pth"
LOGO_PATH = ROOT_DIR / "assets" / "skuast_logo.png"
TARGET_NAMES = ("Sand", "Silt", "Clay")
IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ]
)


def classify_texture(sand: float, silt: float, clay: float) -> str:
    """Classify a sand/silt/clay composition using USDA texture-triangle rules."""
    total = sand + silt + clay
    if total <= 0:
        raise ValueError("The sand, silt, and clay percentages must have a positive sum.")
    sand, silt, clay = (value * 100.0 / total for value in (sand, silt, clay))

    if silt + 1.5 * clay < 15:
        return "Sand"
    if silt + 2 * clay < 30:
        return "Loamy sand"
    if (
        clay < 7
        or (clay < 20 and sand > 52)
    ) and silt + 2 * clay >= 30:
        return "Sandy loam"
    if clay >= 7 and clay < 27 and silt >= 28 and silt < 50 and sand <= 52:
        return "Loam"
    if silt >= 80 and clay < 12:
        return "Silt"
    if silt >= 50 and clay < 27:
        return "Silt loam"
    if clay >= 20 and clay < 35 and sand > 45:
        return "Sandy clay loam"
    if clay >= 27 and clay < 40 and silt >= 40:
        return "Silty clay loam"
    if clay >= 27 and clay < 40 and sand <= 45:
        return "Clay loam"
    if clay >= 35 and sand > 45:
        return "Sandy clay"
    if clay >= 40 and silt >= 40:
        return "Silty clay"
    if clay >= 40 and sand <= 45:
        return "Clay"
    raise ValueError(
        "The estimated percentages fall on an unclassified texture-triangle boundary."
    )


class SoilRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = models.mobilenet_v3_small(weights=None)
        input_features = self.network.classifier[-1].in_features
        self.network.classifier[-1] = nn.Linear(input_features, 3)

    def forward(self, images):
        return torch.softmax(self.network(images), dim=1) * 100.0


@st.cache_resource(max_entries=2, show_spinner="Loading your soil model...")
def load_model(model_path: str, modified_ns: int):
    del modified_ns
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            "This file is not a soil model checkpoint created by train_soil_model.py."
        )
    if checkpoint.get("architecture") != "mobilenet_v3_small":
        raise ValueError("This checkpoint uses an unsupported model architecture.")

    model = SoilRegressor()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, device


def main():
    st.set_page_config(
        page_title="Soilwise | Soil texture report",
        page_icon=":material/eco:",
        layout="centered",
    )

    st.html(
        """
        <style>
        .stApp {
            background:
                radial-gradient(1200px 600px at 85% -10%, #1f7a55 0%, transparent 60%),
                radial-gradient(900px 500px at -10% 10%, #13573d 0%, transparent 55%),
                linear-gradient(180deg, #0b3b2a 0%, #062c1f 100%);
            background-attachment: fixed;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMainBlockContainer"] {
            max-width: 960px;
            padding-top: 1.25rem;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            color: #14281f;
            border: 1px solid #d6e5db;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
        }
        [data-testid="stVerticalBlockBorderWrapper"] h1,
        [data-testid="stVerticalBlockBorderWrapper"] h2,
        [data-testid="stVerticalBlockBorderWrapper"] h3 {
            color: #11523a;
        }
        [data-testid="stVerticalBlockBorderWrapper"] p,
        [data-testid="stVerticalBlockBorderWrapper"] label,
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stWidgetLabel"],
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMetricLabel"],
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMetricValue"] {
            color: #14281f;
        }
        [data-testid="stSidebar"] {
            background: #0b3b2a;
        }
        @media (max-width: 760px) {
            [data-testid="stMainBlockContainer"] {
                padding: 1rem 1rem 2rem;
            }
        }
        </style>
        """
    )

    with st.container():
        brand_column, title_column = st.columns([0.13, 1], vertical_alignment="center")
        with brand_column:
            if LOGO_PATH.is_file():
                st.image(LOGO_PATH, width=72)
        with title_column:
            st.caption("SKUAST-KASHMIR  ·  SOIL COMPOSITION")
            st.badge("EXPERIMENTAL", icon=":material/science:", color="yellow")
            st.title("Soilwise", anchor=False)
            st.write(
                "Estimate soil texture from a photograph, built on Kashmir soil samples."
            )

    model_path = DEFAULT_MODEL
    st.warning(
        " In the held-out test, the model's estimates were "
        "off by about **21.4 percentage points for sand**, **7.7 for silt**, "
        "and **22.2 for clay** on average. Use these results for experiments "
        "only—not for agricultural or purchasing decisions."
    )
    st.space("small")
    upload_column, preview_column = st.columns(
        [1, 1],
        vertical_alignment="top",
    )

    with upload_column:
        with st.container(border=True):
            st.subheader("1. Soil photo")
            st.write("Choose a clear, well-lit photograph of your soil.")
            uploaded_image = st.file_uploader(
                "Upload a soil image",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                max_upload_size=20,
                help="Use a clear, well-lit photo of the soil sample.",
            )
            st.caption("JPG · PNG · WEBP · BMP  /  Maximum 20 MB")
            st.caption("Your photo is sent to the server for prediction and is not stored.")

    image = None
    image_error = None
    if uploaded_image is not None:
        try:
            image = ImageOps.exif_transpose(
                Image.open(BytesIO(uploaded_image.getvalue()))
            ).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            image_error = f"Could not read this image: {exc}"

    with preview_column:
        with st.container(border=True):
            st.subheader("2. Prediction")
            if image is not None:
                st.image(image, caption=uploaded_image.name, width="stretch")
            elif image_error:
                st.error(image_error)
            else:
                st.caption("Choose a photo and press Predict.")

    analyze = st.button(
        "Predict",
        type="primary",
        icon=":material/analytics:",
        disabled=image is None,
    )
    if analyze and image is not None:
        if not model_path.is_file():
            st.error(
                "Model checkpoint not found. Copy the soil_model_output folder "
                "beside streamlit_app.py, or set its path under Analysis settings."
            )
        else:
            try:
                model, device = load_model(
                    str(model_path.resolve()),
                    model_path.stat().st_mtime_ns,
                )
                input_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)
                with torch.inference_mode():
                    prediction = model(input_tensor)[0].cpu().tolist()

                texture = classify_texture(*prediction)
                st.space("small")
                with st.container(border=True):
                    st.subheader("USDA texture class")
                    st.metric("Estimated class", texture)
                    st.markdown("**Estimated composition**")
                    st.html(
                        f"""
                        <div style="display:flex;height:26px;border-radius:10px;
                                    overflow:hidden;margin:10px 0;background:#e6f2ea">
                          <span title="Sand {prediction[0]:.1f}%"
                                style="width:{prediction[0]:.3f}%;background:#d9a441"></span>
                          <span title="Silt {prediction[1]:.1f}%"
                                style="width:{prediction[1]:.3f}%;background:#a98467"></span>
                          <span title="Clay {prediction[2]:.1f}%"
                                style="width:{prediction[2]:.3f}%;background:#8c3b1e"></span>
                        </div>
                        <div style="display:flex;gap:14px;flex-wrap:wrap;
                                    color:#5a6f63;font:0.9rem system-ui,sans-serif">
                          <span><b style="color:#d9a441">●</b> Sand {prediction[0]:.1f}%</span>
                          <span><b style="color:#a98467">●</b> Silt {prediction[1]:.1f}%</span>
                          <span><b style="color:#8c3b1e">●</b> Clay {prediction[2]:.1f}%</span>
                        </div>
                        """
                    )
                    st.caption(
                        "Texture name is derived from predicted percentages using "
                        "USDA texture-triangle categories."
                    )
            except (OSError, RuntimeError, ValueError, KeyError) as exc:
                st.error(f"Could not run the model: {exc}")

    with st.container(border=True):
        st.subheader("Before using the estimate")
        st.markdown(
            "- This is an experimental image model and laboratory soil analysis.\n"
            "- Lighting, moisture, camera distance, and sample preparation may change results.\n"
            "- Use a lab test when accurate soil texture is important."
        )

    st.caption("SOILWISE  ·  SKUAST-KASHMIR")


main()
