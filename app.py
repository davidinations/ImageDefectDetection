import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="wide",
)

MODEL_DIR = "models"
TEST_IMAGE_DIR = "test_images"
DEFAULT_CONF = 0.8
DEFAULT_IOU = 0.9

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

CONF_DEFAULT_LABEL = "Set default model (auto)"
CUSTOM_LABEL = "Upload custom model (.pt)"


@st.cache_resource
def load_model(model_source):
    """Load a YOLO model. model_source is either a path or uploaded-file bytes."""
    if model_source is None:
        return None
    return YOLO(model_source)


def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def resolve_possible_models():
    """Return a list of .pt model files present in the models/ directory."""
    ensure_model_dir()
    return sorted(
        f for f in os.listdir(MODEL_DIR) if f.lower().endswith(".pt")
    )


def load_model_from_bytes(uploaded_file):
    """Save an uploaded .pt file to a temp path and load it."""
    suffix = os.path.splitext(uploaded_file.name)[1] or ".pt"
    with tempfile.NamedTemporaryFile(
        suffix=suffix, delete=False, dir=tempfile.gettempdir()
    ) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    model = YOLO(tmp_path)
    return model, uploaded_file.name


def find_test_image(model_name):
    """
    Given a model filename (e.g. 'sample1.pt'), return the path of the matching
    test image in test_images/ (e.g. 'sample1.png') if one exists.
    """
    stem = os.path.splitext(model_name)[0]
    for ext in IMAGE_EXTENSIONS:
        path = os.path.join(TEST_IMAGE_DIR, stem + ext)
        if os.path.exists(path):
            return path, stem + ext
    return None, None


def find_all_test_images(model_name):
    """
    Given a model filename (e.g. 'sample1.pt'), return the paths of all matching
    original test images in test_images/ (e.g. both 'sample1.png' and
    'sample1.jpg' if present), as a list of (path, filename).
    """
    stem = os.path.splitext(model_name)[0]
    matches = []
    for ext in IMAGE_EXTENSIONS:
        path = os.path.join(TEST_IMAGE_DIR, stem + ext)
        if os.path.exists(path):
            matches.append((path, stem + ext))
    return matches


def clear_image_source():
    """Remove the stored image source so no image is processed."""
    st.session_state.pop("image_mode", None)
    st.session_state.pop("image_path", None)
    st.session_state.pop("image_filename", None)
    st.session_state.pop("image_bytes", None)


def set_sample_source(sample_path, sample_filename):
    """Point the app at a sample test image for detection."""
    st.session_state["image_mode"] = "sample"
    st.session_state["image_path"] = sample_path
    st.session_state["image_filename"] = sample_filename
    st.session_state.pop("image_bytes", None)
    st.session_state.pop("uploaded_name", None)
    # Flag that we should scroll to the "Asli" subheader on the next render.
    st.session_state["scroll_to_asli"] = True


def scroll_to_anchor(anchor_id: str):
    """Scroll the page to the element with the given id (smooth)."""
    import streamlit.components.v1 as components

    components.html(
        f"""
        <script>
            var el = window.parent.document.getElementById("{anchor_id}");
            if (el) {{
                el.scrollIntoView({{behavior: "smooth", block: "start"}});
            }}
        </script>
        """,
        height=0,
    )


def get_active_image():
    """
    Returns (kind, image_source, filename):
      - kind "bytes": uploaded through st.file_uploader
      - kind "path":  sample image selected via a button
      - kind None:    nothing to process
    """
    mode = st.session_state.get("image_mode")
    if mode == "sample":
        path = st.session_state.get("image_path")
        if path and os.path.exists(path):
            return "path", path, st.session_state.get("image_filename", os.path.basename(path))
    if mode == "upload":
        data = st.session_state.get("image_bytes")
        name = st.session_state.get("uploaded_name")
        if data is not None:
            return "bytes", data, name
    return None, None, None


def main():
    st.title("Deteksi Objek dengan YOLO")
    st.markdown(
        "Silakan pilih model dan unggah gambar (JPG, PNG, BMP). "
        "Hasil deteksi akan ditampilkan di sebelah kanan."
    )

    with st.sidebar:
        st.header("Model Selection")

        possible_models = resolve_possible_models()

        model_choice = st.radio(
            "Model pengaturan",
            options=[CONF_DEFAULT_LABEL, CUSTOM_LABEL],
            help=(
                f"Pilih '{CONF_DEFAULT_LABEL}' untuk memakai model default, "
                f"atau '{CUSTOM_LABEL}' untuk mengunggah model Anda sendiri (.pt)."
            ),
        )

        model = None
        model_name = None
        is_default_model = model_choice == CONF_DEFAULT_LABEL

        if model_choice == CONF_DEFAULT_LABEL:
            if possible_models:
                model_name = st.selectbox(
                    "Pilih model default",
                    options=possible_models,
                    index=0,
                )
                model_path = os.path.join(MODEL_DIR, model_name)
                model = load_model(model_path)
            else:
                st.warning(
                    f"Tidak ada model di folder `{MODEL_DIR}/`. "
                    "Letakkan file `.pt` di sana, atau unggah model manual."
                )
        else:
            uploaded_model = st.file_uploader(
                "Unggah model (.pt)",
                type=["pt"],
                accept_multiple_files=False,
            )
            if uploaded_model is not None:
                with st.spinner("Memuat model..."):
                    try:
                        model, model_name = load_model_from_bytes(
                            uploaded_model)
                        st.success(f"Model diunggah: {uploaded_model.name}")
                    except Exception as e:
                        st.error(f"Gagal memuat model: {e}")
                        model = None

        st.divider()
        st.header("Pengaturan Deteksi")

        conf_threshold = st.slider(
            "Confidence threshold",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_CONF,
            step=0.05,
            help="Semakin tinggi, semakin yakin deteksinya.",
        )

        iou_threshold = st.slider(
            "IoU threshold",
            min_value=0.0,
            max_value=1.0,
            value=DEFAULT_IOU,
            step=0.05,
            help="Semakin rendah, semakin sedikit tumpang tindih bbox.",
        )

        if is_default_model:
            st.divider()
            st.header("Sample Images")
            st.caption(
                f"Tombol dibawah akan memuat gambar contoh yang sesuai dengan model yang dipilih."
                "(mis. `sample1.pt` → `sample1.png` / `.jpg`). "
                "Tombol aktif hanya untuk model yang sudah diset default dan jika gambar contoh ada di folder `test_images/`"
            )

            sample1_path, sample1_file = find_test_image("sample1.pt")
            sample2_path, sample2_file = find_test_image("sample2.pt")

            active_sample_key = os.path.splitext(
                model_name)[0] if model_name else None

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                btn1 = st.button(
                    "Sample 1",
                    key="btn_sample1",
                    disabled=(active_sample_key !=
                              "sample1" or sample1_path is None),
                    help=(
                        "Aktif hanya saat `sample1.pt` dipilih."
                        if sample1_path
                        else "Tidak ada gambar sample1 di folder test_images."
                    ),
                )
            with btn_col2:
                btn2 = st.button(
                    "Sample 2",
                    key="btn_sample2",
                    disabled=(active_sample_key !=
                              "sample2" or sample2_path is None),
                    help=(
                        "Aktif hanya saat `sample2.pt` dipilih."
                        if sample2_path
                        else "Tidak ada gambar sample2 di folder test_images."
                    ),
                )

            if btn1:
                set_sample_source(sample1_path, sample1_file)
            if btn2:
                set_sample_source(sample2_path, sample2_file)

            # if active_sample_key == "sample1":
            #     if sample1_path is not None:
            #         st.info(
            #             f"Klik **Sample 1** untuk memakai gambar contoh "
            #             f"`{sample1_file}` pada model `sample1.pt`. "
            #             "Jika kamu punya gambar sendiri, unggah lewat "
            #             "`Unggah gambar` di bawah — itu akan menggantikan contoh."
            #         )
            #     else:
            #         st.warning(
            #             "Model `sample1.pt` dipilih, tapi belum ada gambar "
            #             f"contoh di folder `{TEST_IMAGE_DIR}/`. Letakkan file "
            #             "`sample1.png`/`.jpg` di sana (atau gunakan gambar "
            #             "sendiri lewat `Unggah gambar`)."
            #         )
            # elif active_sample_key == "sample2":
            #     if sample2_path is not None:
            #         st.info(
            #             f"Klik **Sample 2** untuk memakai gambar contoh "
            #             f"`{sample2_file}` pada model `sample2.pt`. "
            #             "Jika kamu punya gambar sendiri, unggah lewat "
            #             "`Unggah gambar` di bawah — itu akan menggantikan contoh."
            #         )
            #     else:
            #         st.warning(
            #             "Model `sample2.pt` dipilih, tapi belum ada gambar "
            #             f"contoh di folder `{TEST_IMAGE_DIR}/`. Letakkan file "
            #             "`sample2.png`/`.jpg` di sana (atau gunakan gambar "
            #             "sendiri lewat `Unggah gambar`)."
            #         )

    if model is None:
        st.info(
            "⚠️ Belum ada model yang dimuat. Letakkan file `.pt` di folder "
            f"`{MODEL_DIR}/` atau unggah model manual di sidebar."
        )
        clear_image_source()
        return

    # If the selected model changed (different default model, or switched
    # between default/custom), clear any previously loaded image so the old
    # detection is wiped out.
    identity = f"{model_choice}:{model_name}"
    if st.session_state.get("model_identity") != identity:
        clear_image_source()
        st.session_state["model_identity"] = identity

    st.sidebar.caption(f"Model aktif: **{model_name}**")

    # ---- Uploaded image ----
    uploaded_image = st.file_uploader(
        "Unggah gambar (JPG, PNG, BMP)",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=False,
    )

    if uploaded_image is not None:
        # A newly uploaded image overrides any sample selection.
        if st.session_state.get("uploaded_name") != uploaded_image.name:
            # Only scroll when a different image is uploaded (not on every
            # rerun while the same file stays selected).
            st.session_state["scroll_to_asli"] = True
        st.session_state["image_mode"] = "upload"
        st.session_state["image_bytes"] = uploaded_image.getvalue()
        st.session_state["uploaded_name"] = uploaded_image.name
        st.session_state.pop("image_path", None)

    # ---- Show original sample images bound to the selected model ----
    if is_default_model and model_name:
        matching_samples = find_all_test_images(model_name)
        if matching_samples:
            st.markdown(
                "**Gambar asli sample (dari `test_images/`). Jika tidak ada gambar klik tombol sample yang terbuka di sidebar bawah**")
            sample_cols = st.columns(len(matching_samples))
            for col, (s_path, s_file) in zip(sample_cols, matching_samples):
                with col:
                    st.image(
                        Image.open(s_path).convert("RGB").resize((256, 256)),
                        width=256,
                    )
                    st.caption(s_file)

    image_kind, image_source, image_filename = get_active_image()

    st.markdown(
        '<a id="asli-anchor" style="display:block; height:0; scroll-margin-top: 30px"></a>',
        unsafe_allow_html=True,
    )
    input_col, output_col = st.columns(2)

    with input_col:
        st.subheader("Gambar Asli")
        input_placeholder = st.empty()

    with output_col:
        st.subheader("Hasil Deteksi")
        output_placeholder = st.empty()

    tables_placeholder = st.empty()

    if image_kind is None:
        input_placeholder.markdown("Tunggu gambar diunggah...")
        output_placeholder.markdown("Hasil akan tampil di sini.")
        return

    if image_kind == "path":
        image = Image.open(image_source).convert("RGB")
        st.caption(f"Gambar sample: **{image_filename}**")
    else:
        image = Image.open(io_bytes(image_source)).convert("RGB")

    input_placeholder.image(image, width="stretch")

    with st.spinner("Memproses deteksi..."):
        try:
            results = model.predict(
                source=np.array(image),
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
        except Exception as e:
            msg = str(e)
            output_placeholder.error(
                "Model tidak dapat memproses gambar ini.\n\n"
                f"Detail: {msg}"
            )
            st.warning(
                "Model yang dipilih tidak mendukung input gambar (biasanya "
                "bukan model deteksi YOLO). Silakan gunakan model deteksi "
                "(.pt) yang kompatibel atau unggah model yang tepat."
            )
            return

    result = results[0]

    EXCLUDED_CLASSES = {"sample"}

    # ---- Draw annotated image manually, skipping excluded classes ----
    img_bgr = np.array(image)[:, :, ::-1].copy()  # RGB -> BGR
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        if cls_name.lower() in EXCLUDED_CLASSES:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # label = f"{cls_name} {conf:.2f}"
        label = f"{cls_name}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        ty = y1 - th - 4 if y1 - th - 4 > 0 else y1 + 4
        cv2.rectangle(
            img_bgr,
            (x1, ty),
            (x1 + tw + 4, ty + th + 4),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            img_bgr,
            label,
            (x1 + 2, ty + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    annotated_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    output_placeholder.image(annotated_rgb, width="stretch")

    # ---- Results table (excluding e.g. "sample") ----
    rows = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        cls_name = result.names[cls_id]
        if cls_name.lower() in EXCLUDED_CLASSES:
            continue
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [round(v) for v in box.xyxy[0].tolist()]
        rows.append(
            {
                "Kelas": cls_name,
                "Confidence": round(conf, 3),
                "Bounding Box": f"[{x1}, {y1}, {x2}, {y2}]",
                "Lebar (px)": x2 - x1,
                "Tinggi (px)": y2 - y1,
            }
        )

    st.success(f"Jumlah objek terdeteksi: {len(rows)}")

    if rows:
        df = pd.DataFrame(rows)
        counts = df["Kelas"].value_counts().rename_axis(
            "Kelas").reset_index(name="Jumlah")

        table_col, count_col = tables_placeholder.columns(2)
        with table_col:
            st.subheader("Tabel Hasil")
            st.dataframe(df, width="stretch", hide_index=True)
        with count_col:
            st.subheader("Jumlah per Kelas")
            st.dataframe(counts, width="stretch", hide_index=True)
    else:
        tables_placeholder.info(
            "Tidak ada objek yang terdeteksi pada gambar ini.")

    # Auto-scroll to the "Asli" subheader once an image is shown (e.g. after a
    # Sample button click or a new image upload), then clear the flag so we do
    # not keep scrolling on unrelated reruns (e.g. slider changes).
    if st.session_state.get("scroll_to_asli"):
        scroll_to_anchor("asli-anchor")
        st.session_state["scroll_to_asli"] = False


def io_bytes(data: bytes):
    import io
    return io.BytesIO(data)


if __name__ == "__main__":
    main()
