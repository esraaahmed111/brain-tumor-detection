# Import libraries
import os
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms, models
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Detector",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuration
CLASSIFIER_PATH = "best_brain_model.pth"
UNET_PATH       = "outputs/best_unet.pth"
IMG_SIZE        = 224
SEG_SIZE        = 256
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN            = [0.485, 0.456, 0.406]
STD             = [0.229, 0.224, 0.225]

# Transforms
clf_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
seg_transform = transforms.Compose([
    transforms.Resize((SEG_SIZE, SEG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


# Load Classifier
@st.cache_resource(show_spinner="Loading classifier...")
def load_classifier():
    if not os.path.exists(CLASSIFIER_PATH):
        return None, None
    model = models.resnet50(weights=None)
    try:
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
        class_names = ["no", "yes"]
    except Exception:
        model.fc = nn.Linear(model.fc.in_features, 4)
        model.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
        class_names = ["class_0", "class_1", "class_2", "class_3"]
    return model.to(DEVICE).eval(), class_names


# Load U-Net
@st.cache_resource(show_spinner="Loading U-Net...")
def load_unet():
    if not os.path.exists(UNET_PATH):
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("unet", "models/unet.py")
    umod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(umod)
    model = umod.UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()


# Inference
@torch.no_grad()
def classify_image(model, class_names, pil_img):
    tensor = clf_transform(pil_img).unsqueeze(0).to(DEVICE)
    probs  = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
    idx    = int(np.argmax(probs))
    return class_names[idx], probs, class_names

@torch.no_grad()
def segment_image(model, pil_img):
    tensor = seg_transform(pil_img).unsqueeze(0).to(DEVICE)
    mask   = torch.sigmoid(model(tensor)).cpu().squeeze().numpy()
    return mask


# Overlay mask on image
def overlay_mask(pil_img, mask, alpha=0.4):
    img_arr = np.array(pil_img.resize((SEG_SIZE, SEG_SIZE)))
    heat    = cm.hot(mask)[:, :, :3]
    overlay = (1 - alpha) * img_arr / 255 + alpha * heat
    return Image.fromarray((np.clip(overlay, 0, 1) * 255).astype(np.uint8))


# Confidence bar
def confidence_bar(label, prob):
    color = "#ef4444" if label.lower() in ("yes", "tumor", "malignant") else "#22c55e"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:6px 0">
      <span style="width:100px;font-weight:600">{label}</span>
      <div style="flex:1;background:#e5e7eb;border-radius:8px;height:20px">
        <div style="width:{prob*100:.1f}%;background:{color};height:100%;border-radius:8px;"></div>
      </div>
      <span style="width:52px;text-align:right;font-weight:600">{prob*100:.1f}%</span>
    </div>
    """, unsafe_allow_html=True)


def main():
    # Sidebar
    st.sidebar.title("Brain Tumor Detector")
    st.sidebar.markdown("Upload an MRI scan to detect and segment brain tumors.")
    run_seg   = st.sidebar.checkbox("Run segmentation (U-Net)", value=True)
    threshold = st.sidebar.slider("Segmentation threshold", 0.1, 0.9, 0.5, 0.05)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Models:**")

    clf_model, class_names = load_classifier()
    unet_model             = load_unet() if run_seg else None

    st.sidebar.markdown(f"- Classifier : {'Loaded' if clf_model else 'Not found'}")
    st.sidebar.markdown(f"- U-Net      : {'Loaded' if unet_model else ('Not found' if run_seg else 'Disabled')}")

    # Main
    st.title("Brain Tumor Detection and Segmentation")
    st.markdown("Upload a brain MRI image. The classifier detects tumor presence and U-Net highlights the tumor region.")

    uploaded = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded:
        pil_img    = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Input MRI")
            st.image(pil_img, use_column_width=True)

        # Classification
        if clf_model:
            label, probs, cnames = classify_image(clf_model, class_names, pil_img)
            with col2:
                st.subheader("Classification Result")
                is_tumor    = label.lower() in ("yes", "tumor", "malignant")
                badge_color = "#ef4444" if is_tumor else "#22c55e"
                st.markdown(f"""
                <div style="background:{badge_color};color:white;padding:16px 24px;
                            border-radius:12px;font-size:24px;font-weight:700;
                            text-align:center;margin-bottom:16px">
                  {'Tumor Detected' if is_tumor else 'No Tumor Detected'}
                </div>
                <div style="text-align:center;font-size:18px;margin-bottom:20px">
                  Predicted class: <b>{label}</b>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**Confidence scores:**")
                for name, prob in sorted(zip(cnames, probs), key=lambda x: -x[1]):
                    confidence_bar(name, prob)
        else:
            with col2:
                st.warning("Classifier not found. Run classifier.py first.")

        # Segmentation
        if run_seg and unet_model:
            st.subheader("Tumor Segmentation (U-Net)")
            mask        = segment_image(unet_model, pil_img)
            binary_mask = (mask > threshold).astype(np.uint8)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Probability Heatmap**")
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(mask, cmap='hot', vmin=0, vmax=1)
                plt.colorbar(ax.images[0], ax=ax)
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
            with c2:
                st.markdown("**Binary Mask**")
                st.image(binary_mask * 255, clamp=True, use_column_width=True)
            with c3:
                st.markdown("**Overlay**")
                st.image(overlay_mask(pil_img, mask), use_column_width=True)

            tumor_pct = binary_mask.sum() / binary_mask.size * 100
            st.metric("Estimated Tumor Coverage", f"{tumor_pct:.2f}%")

        elif run_seg and not unet_model:
            st.info("U-Net not found. Run segmentation.py first.")

    else:
        st.info("Upload a brain MRI image to get started.")

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:gray'>For research and educational purposes only. "
        "Not a medical diagnostic tool.</p>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
