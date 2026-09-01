"""
Streamlit Demo for Sino-Nom Text Classification
Distilled BERT 6 tầng (Jihuai/bert-ancient-chinese + 1,786 chữ Nôm) — 8 classes
Model: models_student_6l — test acc 91.8%, 148 MB fp16, ~300 MB RAM, CPU-ready
Nếu không có model local, app tự tải models_student_6l từ Kaggle (kagglehub).

Thay thế pipeline BERT-LSTM/Decision-Templates cũ (7 lớp, macro-F1 0.838)
bằng baseline chuẩn hiện tại (8 lớp, macro-F1 0.919):
  - fine-tune toàn mạng thay vì BERT đóng băng + LSTM
  - vocab mở rộng 1.786 chữ Nôm (CJK Ext-B), [UNK] 8.6% -> 1.2%
  - thêm lớp Philosophy (Triết học)
  - max_len 512 (cũ: 128)
  - văn bản dài: cửa sổ trượt 280 ký tự + trung bình xác suất
    (đọc toàn văn thay vì cắt cụt) — trả về MỘT nhãn tự tin nhất
"""
import base64
import io
import json
import os
import unicodedata

import numpy as np
import requests
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# Set page config
st.set_page_config(
    page_title="Sino-Nom Text Classification",
    page_icon="📜",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
.stTextArea > div > div > textarea {
    font-family: 'Courier New', monospace !important;
    font-size: 14px !important;
    color: #000000 !important;
    background-color: #f8f9fa !important;
    border: 1px solid #dee2e6 !important;
    border-radius: 8px !important;
    padding: 12px !important;
    line-height: 1.6 !important;
}
.stTextArea { margin: 8px 0px !important; }
.stMarkdown p { margin-bottom: 8px !important; }
.result-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 12px;
    margin: 16px 0px;
    color: white;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
.result-title {
    font-size: 1.8em;
    font-weight: bold;
    margin: 0;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}
.result-subtitle {
    color: rgba(255,255,255,0.9);
    margin: 8px 0 0 0;
    font-size: 1em;
}
.confidence-container {
    background: #f8f9fa;
    padding: 16px;
    border-radius: 8px;
    margin: 8px 0px;
}
.streamlit-info {
    background: #e1f5fe;
    padding: 16px;
    border-radius: 8px;
    border-left: 5px solid #0288d1;
    margin: 16px 0;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------- Constants
# Thu muc mo hinh: uu tien secrets/env, roi cac duong dan local quen thuoc.
# Chi dung student 6 tang (148MB fp16, ~300MB RAM, ~2x nhanh tren CPU).
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR_CANDIDATES = [
    os.environ.get("MODEL_DIR", ""),
    "models_student_6l",
    os.path.join(_APP_DIR, "models_student_6l"),
    os.path.join(_APP_DIR, "..", "yhct_classification_model2", "models_student_6l"),
]

# Kaggle dataset chua thu muc models_student_6l (model_fp16.pt, tokenizer/, meta.json).
# Sua handle nay theo dataset cua ban, hoac ghi de bang secrets/env KAGGLE_DATASET.
_DEFAULT_KAGGLE_DATASET = "phuchoangnguyen/sinonomtext-bert-fp16/pyTorch/default"

CATEGORY_ICONS = {
    "Admin": "🏛️",
    "Medical": "🏥",
    "History": "📚",
    "Literature": "📖",
    "Buddhism": "🪷",
    "Catholics": "⛪",
    "Philosophy": "☯️",
    "Others": "📋",
}
CATEGORY_VI = {
    "Admin": "Hành chính",
    "Medical": "Y học",
    "History": "Lịch sử",
    "Literature": "Văn học",
    "Buddhism": "Phật giáo",
    "Catholics": "Công giáo",
    "Philosophy": "Triết học",
    "Others": "Khác",
}

# Cua so truot cho van ban dai (khop phan phoi huan luyen ~280 ky tu)
WINDOW = 280
OVERLAP = 50


def _is_model_dir(d):
    """Thu muc hop le: co checkpoint (fp16 hoac fp32) + meta.json."""
    return bool(d) and os.path.exists(os.path.join(d, "meta.json")) and (
        os.path.exists(os.path.join(d, "model_fp16.pt"))
        or os.path.exists(os.path.join(d, "model.pt"))
    )


def _get_secret(key, default=""):
    """Doc tu st.secrets, fallback ve bien moi truong."""
    try:
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(key, default)


def download_model_from_kaggle():
    """Tai models_student_6l tu Kaggle bang kagglehub.

    - Handle dataset lay tu secrets/env KAGGLE_DATASET (mac dinh _DEFAULT_KAGGLE_DATASET).
    - Dataset private can them KAGGLE_USERNAME + KAGGLE_KEY trong secrets/env.
    Tra ve duong dan thu muc chua model, hoac None neu tai that bai.
    """
    dataset = _get_secret("KAGGLE_DATASET", _DEFAULT_KAGGLE_DATASET)
    if not dataset or dataset.startswith("<"):
        st.error(
            "❌ Chưa cấu hình Kaggle dataset. Khai báo `KAGGLE_DATASET` "
            "(dạng `username/dataset-slug`) trong secrets hoặc biến môi trường."
        )
        return None

    # kagglehub doc credentials tu env — day secrets vao env neu co
    for key in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
        val = _get_secret(key)
        if val:
            os.environ[key] = val

    try:
        import kagglehub
    except ImportError:
        st.error("❌ Thiếu thư viện `kagglehub`. Cài bằng: `pip install kagglehub`")
        return None

    try:
        with st.spinner(f"⬇️ Đang tải mô hình từ Kaggle (`{dataset}`)..."):
            root = kagglehub.dataset_download(dataset)
    except Exception as e:
        st.error(f"❌ Lỗi khi tải mô hình từ Kaggle (`{dataset}`): {e}")
        return None

    # Tim thu muc model: goc dataset hoac thu muc con (vd. models_student_6l/)
    if _is_model_dir(root):
        return root
    for cur, _dirs, _files in os.walk(root):
        if _is_model_dir(cur):
            return cur

    st.error(
        f"❌ Dataset Kaggle `{dataset}` không chứa model hợp lệ "
        "(cần `model_fp16.pt`/`model.pt`, `tokenizer/`, `meta.json`)."
    )
    return None


def resolve_model_dir():
    """Tim thu muc mo hinh student local; khong co thi tai tu Kaggle."""
    for d in [_get_secret("MODEL_DIR")] + _MODEL_DIR_CANDIDATES:
        if _is_model_dir(d):
            return os.path.abspath(d)
    kaggle_dir = download_model_from_kaggle()
    return os.path.abspath(kaggle_dir) if kaggle_dir else None


# ----------------------------------------------- Han-Nom character filtering
def is_han_nom_char(char):
    """Kiểm tra xem ký tự có phải là Hán-Nôm không"""
    return any([
        '一' <= char <= '鿿',   # CJK Unified Ideographs
        '㐀' <= char <= '䶿',   # CJK Extension A
        '\U00020000' <= char <= '\U0002a6df',  # CJK Extension B (chữ Nôm)
        '\U0002a700' <= char <= '\U0002b73f',  # CJK Extension C
        '\U0002b740' <= char <= '\U0002b81f',  # CJK Extension D
        '\U0002b820' <= char <= '\U0002ceaf',  # CJK Extension E
        '\U0002ceb0' <= char <= '\U0002ebef',  # CJK Extension F
    ])


def preprocess_han_nom_text(text):
    """Tiền xử lý: chuẩn hoá NFC rồi lọc chỉ giữ ký tự Hán-Nôm."""
    text = unicodedata.normalize("NFC", text)
    filtered = ''.join(c for c in text if is_han_nom_char(c))
    return filtered.strip()


# ------------------------------------------------------- Model (student 6L)
class HanNomClassifier(nn.Module):
    """BERT fine-tuned + masked mean-pooling + Linear(768 -> 8).

    Giong het kien truc huan luyen trong han_nom_classification_v2.ipynb.
    """

    def __init__(self, base_model, emb_rows, num_classes, num_layers=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model)
        if emb_rows != self.bert.config.vocab_size:
            self.bert.resize_token_embeddings(emb_rows)   # +1.786 chu Nom
        if num_layers and num_layers < len(self.bert.encoder.layer):
            # student chung cat: giu dung so tang truoc khi nap trong so
            self.bert.encoder.layer = nn.ModuleList(self.bert.encoder.layer[:num_layers])
            self.bert.config.num_hidden_layers = num_layers
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        h = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)   # mean-pool bo padding
        return self.fc(self.dropout(pooled))


@st.cache_resource
def load_models():
    """Load tokenizer + mo hinh student 6 tang (local, hoac tai tu Kaggle)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model_dir = resolve_model_dir()
    if model_dir is None:
        st.error(
            "❌ Không tìm thấy mô hình `models_student_6l` "
            "(cần `model_fp16.pt`/`model.pt`, `tokenizer/`, `meta.json`).\n\n"
            "Đặt thư mục cạnh app, khai báo `MODEL_DIR`, hoặc cấu hình "
            "`KAGGLE_DATASET` trong secrets/biến môi trường để tải từ Kaggle."
        )
        return None

    try:
        meta = json.load(open(os.path.join(model_dir, "meta.json"), encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        model = HanNomClassifier(meta["base_model"], meta["emb_rows"],
                                 len(meta["classes"]), meta.get("num_layers"))
        fp16_path = os.path.join(model_dir, "model_fp16.pt")
        ckpt = fp16_path if os.path.exists(fp16_path) else os.path.join(model_dir, "model.pt")
        state = torch.load(ckpt, map_location="cpu")
        # checkpoint fp16 -> ep ve fp32 de chay CPU (CPU khong ho tro fp16 tot)
        state = {k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}
        model.load_state_dict(state)
        model.to(device).eval()
        meta["_model_dir"] = model_dir
        meta["_checkpoint"] = os.path.basename(ckpt)
        return tokenizer, model, meta, device
    except Exception as e:
        st.error(f"❌ Lỗi khi tải mô hình từ {model_dir}: {e}")
        return None


# ------------------------------------------------------------ Inference
@torch.no_grad()
def _predict_probs(texts, tokenizer, model, meta, device, batch_size=16):
    """Xac suat 8 lop cho danh sach doan van (pad dong theo batch)."""
    out = []
    for k in range(0, len(texts), batch_size):
        chunk = texts[k:k + batch_size]
        enc = tokenizer(chunk, padding=True, truncation=True,
                        max_length=meta["max_len"], return_tensors="pt")
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device)).float()
        out.append(torch.softmax(logits, -1).cpu().numpy())
    return np.concatenate(out, axis=0)


def classify_text(text, tokenizer, model, meta, device):
    """Phan loai mot van ban BAT KY do dai — tra ve MOT nhan tu tin nhat.

    - Van ban ngan/vua (<= WINDOW*1.5 ky tu): phan loai truc tiep.
    - Van ban dai: cat cua so WINDOW ky tu (chong lan OVERLAP), phan loai
      tung cua so roi TRUNG BINH xac suat — doc duoc toan van thay vi
      chi 512 token dau nhu cach cat cut cu.
    """
    text = text.strip()
    if len(text) <= int(WINDOW * 1.5):
        spans = [text]
    else:
        step = WINDOW - OVERLAP
        spans = [text[i:i + WINDOW] for i in range(0, len(text) - OVERLAP, step)]
        spans = [s for s in spans if len(s) >= 30] or [text[:WINDOW]]

    probs = _predict_probs(spans, tokenizer, model, meta, device)
    p = probs.mean(axis=0)
    order = np.argsort(p)[::-1]
    classes = meta["classes"]
    return {
        "label": classes[int(order[0])],
        "confidence": float(p[order[0]]),
        "n_windows": len(spans),
        "all_probs": {classes[int(i)]: float(p[i]) for i in order},
    }


# ------------------------------------------------------------ OCR API
def get_ocr_api_url():
    """Get OCR API URL from secrets or use default"""
    try:
        return st.secrets.get("OCR_API_URL",
                              "https://kimhannom.clc.hcmus.edu.vn/meta-ocr-normal/nom-ocr")
    except Exception:
        return "https://kimhannom.clc.hcmus.edu.vn/meta-ocr-normal/nom-ocr"


def call_ocr_api(base64_image):
    """Call OCR API with error handling"""
    try:
        headers = {'User-Agent': 'StreamlitApp', 'Content-Type': 'application/json'}
        if isinstance(base64_image, bytes):
            base64_str = base64.b64encode(base64_image).decode('utf-8')
        else:
            base64_str = base64_image
        payload = {"base64Data": base64_str, "lang_type": 2, "reading_direction": 1}
        return requests.post(get_ocr_api_url(), json=payload, headers=headers,
                             verify=False, timeout=120)
    except requests.exceptions.Timeout:
        st.error("⏱️ OCR API timeout. Vui lòng thử lại.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"🌐 Lỗi kết nối OCR API: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Lỗi khi gọi OCR API: {e}")
        return None


def run_ocr_on_image(image_bytes):
    """Perform OCR using external API with error handling"""
    try:
        if not isinstance(image_bytes, bytes) or len(image_bytes) == 0:
            return '', None
        Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý ảnh: {str(e)}")
        return '', None

    api_result = call_ocr_api(image_bytes)
    if not api_result:
        return '', None

    raw_text = ""
    ocr_text_list = []
    if api_result.status_code == 200:
        try:
            ocr_response_json = api_result.json()
            ocr_text_list = ocr_response_json.get("ocrResult", [])
            raw_text = "\n".join(ocr_text_list) if ocr_text_list else ""
        except Exception as e:
            st.error(f"❌ Lỗi khi parse JSON response: {e}")
            return '', None
    else:
        st.error(f"❌ OCR API trả về lỗi: {api_result.status_code}")
        return '', None

    if raw_text:
        st.markdown("**Văn bản đã nhận diện:**")
        ocr_display = "\n".join(ocr_text_list) if ocr_text_list else raw_text
        st.text_area("Văn bản gốc", value=ocr_display, height=300,
                     disabled=False, label_visibility="hidden")
        processed_text = preprocess_han_nom_text(raw_text)
        if processed_text:
            return processed_text, api_result
        st.info("💡 Sử dụng văn bản gốc do không có ký tự Hán-Nôm.")
        return raw_text, api_result

    st.warning("⚠️ Không phát hiện văn bản trong ảnh.")
    return raw_text, api_result


# ------------------------------------------------------------ UI helpers
def render_result(result):
    """Hien thi ket qua phan loai: MOT nhan + bang do tin cay top-3."""
    label = result["label"]
    main_col, conf_col = st.columns([1, 1])

    with main_col:
        st.markdown(f"""
        <div class="result-container">
            <h2 class="result-title">
                {CATEGORY_ICONS.get(label, '📄')} {label}
            </h2>
            <p class="result-subtitle">
                {CATEGORY_VI.get(label, label)} — độ tin cậy {result['confidence']:.1%}
            </p>
        </div>
        """, unsafe_allow_html=True)
        if result["n_windows"] > 1:
            st.caption(f"📏 Văn bản dài — đã đọc bằng {result['n_windows']} cửa sổ "
                       f"{WINDOW} ký tự và trung bình xác suất.")

    with conf_col:
        st.markdown('<div class="confidence-container">', unsafe_allow_html=True)
        st.markdown("**Độ tin cậy (top 3):**")
        for i, (name, conf) in enumerate(list(result["all_probs"].items())[:3]):
            icon = CATEGORY_ICONS.get(name, '📄')
            if i == 0:
                st.success(f"{icon} {name} ({CATEGORY_VI.get(name, name)}): {conf:.1%}")
            else:
                st.info(f"{icon} {name} ({CATEGORY_VI.get(name, name)}): {conf:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)


def main():
    st.title("📜 Sino-Nom Text Classification")
    st.markdown("### Phân loại văn bản Hán-Nôm — fine-tuned BERT, 8 lĩnh vực")

    st.markdown("""
    <div class="streamlit-info">
        <strong>Mô hình student 6 tầng</strong> — chưng cất (knowledge distillation)
        từ baseline fine-tuned <code>Jihuai/bert-ancient-chinese</code> + 1.786 chữ Nôm.<br>
        Nhẹ cho CPU: <strong>148 MB</strong> (fp16) · ~300 MB RAM · ~24 ms/đoạn.
        Độ chính xác test: <strong>91,8%</strong> (mô hình đầy đủ: 92,4%).
        Văn bản dài được đọc <em>toàn văn</em> bằng cửa sổ trượt.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    model_data = None
    with st.spinner("🔄 Đang tải mô hình... (có thể mất vài phút lần đầu)"):
        model_data = load_models()

    if not model_data:
        st.stop()

    tokenizer, model, meta, device = model_data
    class_names = meta["classes"]
    n_layers = meta.get("num_layers", 12)
    st.success(f"✅ Đã tải mô hình ({n_layers} tầng, {meta.get('_checkpoint','model.pt')}) — Device: {device}")

    st.markdown("**Các loại văn bản (Categories):**")
    cols = st.columns(len(class_names))
    for col, name in zip(cols, class_names):
        with col:
            st.info(f"{CATEGORY_ICONS.get(name, '📄')} {name}")

    st.markdown("---")

    st.markdown("### 📝 Nhập văn bản hoặc upload hình ảnh để phân loại")
    tab1, tab2 = st.tabs(["📤 Upload hình ảnh", "✏️ Nhập văn bản"])

    with tab1:
        st.markdown("**Chọn hình ảnh chứa văn bản Hán-Nôm để tự động nhận diện và phân loại:**")
        uploaded_file = st.file_uploader(
            "Chọn file ảnh",
            type=["jpg", "jpeg", "png"],
            help="Hỗ trợ các định dạng: JPG, JPEG, PNG. Kích thước tối đa: 200MB"
        )

        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, 'getvalue') else uploaded_file.read()
            img_col, result_col = st.columns([1, 2])

            with img_col:
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image, caption="📷 Ảnh đã upload", use_column_width=True)
                except Exception as e:
                    st.error(f"❌ Lỗi hiển thị ảnh: {str(e)}")
                    st.stop()

            with result_col:
                with st.spinner("🔄 Đang nhận diện và phân loại..."):
                    text_from_image, _ = run_ocr_on_image(image_bytes)

                if text_from_image and text_from_image.strip():
                    with st.spinner("🤖 Đang phân loại nội dung..."):
                        result = classify_text(text_from_image, tokenizer, model, meta, device)
                    st.markdown("### 📊 Kết quả phân loại tự động")
                    render_result(result)
                else:
                    st.warning("⚠️ Không nhận diện được nội dung từ ảnh.")

    with tab2:
        st.markdown("**Nhập văn bản Hán-Nôm để phân loại:**")
        text_input = st.text_area(
            "Văn bản Hán-Nôm:",
            height=300,
            placeholder="Nhập văn bản Hán-Nôm vào đây...\n(Ví dụ: 運衰死沙場宣立萬春國)",
            help="Văn bản có thể bằng chữ Hán, chữ Nôm, hoặc hỗn hợp — ngắn hay dài đều được, "
                 "văn bản dài sẽ được đọc toàn văn bằng cửa sổ trượt."
        )

        if st.button("🔍 Phân loại văn bản", type="primary", key="classify_text"):
            cleaned = preprocess_han_nom_text(text_input) or text_input.strip()
            if not cleaned:
                st.warning("⚠️ Vui lòng nhập văn bản để phân loại!")
            else:
                with st.spinner("🔄 Đang phân loại..."):
                    result = classify_text(cleaned, tokenizer, model, meta, device)
                st.markdown("---")
                st.markdown("### 📊 Kết quả phân loại")
                render_result(result)

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: gray;">
            <p>📜 <strong>Sino-Nom Text Classification</strong> | Distilled BERT 6-layer (bert-ancient-chinese + chữ Nôm)</p>
            <p>8 Classes: Admin, Medical, History, Literature, Buddhism, Catholics, Philosophy, Others</p>
            <p><em>Student 6-layer: 91.8% acc · 148 MB fp16 · CPU-ready</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
