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
        page_title="Soilwise | Soil texture estimate",
        page_icon=":material/eco:",
        layout="wide",
    )

    st.title("Soilwise", anchor=False)
    st.caption("A first look at the texture composition of your soil")

    with st.container(border=True):
        st.subheader("From a soil photo to a composition estimate")
        st.write(
            "Upload a clear image of a soil sample to estimate its sand, silt, "
            "and clay percentages."
        )
        st.caption(
            "Research prototype · Estimates are experimental and are not a "
            "replacement for laboratory soil analysis."
        )

    with st.sidebar:
        st.header("Model setup")
        model_path_text = st.text_input(
            "Model checkpoint",
            value=str(DEFAULT_MODEL),
            help=(
                "Choose soil_texture_model.pth from your soil_model_output folder "
                "if it is not at the default location."
            ),
        )
        st.caption("The model runs locally on this computer.")

    model_path = Path(model_path_text).expanduser()
    upload_column, preview_column, result_column = st.columns(
        [1.05, 0.95, 1.15],
        vertical_alignment="top",
    )

    with upload_column:
        with st.container(border=True):
            st.subheader("1 · Choose a sample")
            uploaded_image = st.file_uploader(
                "Upload a soil image",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
                max_upload_size=20,
                help="Use a clear, well-lit photo of the soil sample.",
            )
            st.caption("JPG, PNG, WEBP, or BMP · Up to 20 MB")

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
            st.subheader("2 · Preview")
            if image is not None:
                st.image(image, caption=uploaded_image.name, width="stretch")
            elif image_error:
                st.error(image_error)
            else:
                st.caption("Your selected image will appear here.")

    with result_column:
        with st.container(border=True):
            st.subheader("3 · Estimate")
            if image is None:
                st.caption("Choose an image to start.")
            elif st.button(
                "Analyze soil image",
                type="primary",
                icon=":material/analytics:",
                width="stretch",
            ):
                if not model_path.is_file():
                    st.error(
                        "Model checkpoint not found. Copy the soil_model_output "
                        "folder beside streamlit_app.py, or set the checkpoint "
                        "path in Model setup."
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
                        st.success("Analysis complete")
                        st.metric("Estimated soil texture", texture)
                        value_columns = st.columns(3)
                        for column, name, value in zip(
                            value_columns, TARGET_NAMES, prediction
                        ):
                            column.metric(name, f"{value:.1f}%")

                        st.markdown("**Estimated composition**")
                        for name, value in zip(TARGET_NAMES, prediction):
                            st.progress(
                                min(max(int(round(value)), 0), 100),
                                text=f"{name} · {value:.1f}%",
                            )

                        st.caption(
                            "The percentages sum to 100%. This prototype predicts "
                            "composition only; results can vary with lighting, "
                            "moisture, and sample preparation."
                        )
                    except (OSError, RuntimeError, ValueError, KeyError) as exc:
                        st.error(f"Could not run the model: {exc}")

    st.space("small")
    with st.expander("About this estimate", icon=":material/info:"):
        st.write(
            "The percentages are predicted by a model fine-tuned from "
            "MobileNetV3-Small. The soil texture name is derived from those "
            "predicted percentages using USDA texture-triangle categories. Test "
            "errors were substantial, especially for sand and clay, so both "
            "outputs are early research estimates. For decisions that depend on "
            "accurate soil texture, use a laboratory measurement."
        )


main()
