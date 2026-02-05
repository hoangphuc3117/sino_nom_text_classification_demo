"""
Streamlit Demo for Sino-Nom Text Classification
Using BERT-LSTM Model with 6 Classes
Optimized for Streamlit Cloud deployment
"""
import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import json
import os
import re
# Thêm các import cho OCR API
import requests
import base64
from PIL import Image
import io
import kagglehub
# Thêm import cho Jiayan NLP với error handling
# Disabled due to kenlm compatibility issues with Python 3.13
try:
    # from jiayan import load_lm, CRFSentencizer, CharHMMTokenizer
    JIAYAN_AVAILABLE = False  # Force disable
except ImportError:
    JIAYAN_AVAILABLE = False
    CRFSentencizer = None
    CharHMMTokenizer = None
    load_lm = None

# Set page config
st.set_page_config(
    page_title="Sino-Nom Text Classification",
    page_icon="📜",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
/* Custom styling for text areas */
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

/* Compact margins for text areas */
.stTextArea {
    margin: 8px 0px !important;
}

/* Label styling */
.stMarkdown p {
    margin-bottom: 8px !important;
}

/* Result container styling */
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

/* Confidence scores styling */
.confidence-container {
    background: #f8f9fa;
    padding: 16px;
    border-radius: 8px;
    margin: 8px 0px;
}

/* Alert styling for Streamlit Cloud */
.streamlit-info {
    background: #e1f5fe;
    padding: 16px;
    border-radius: 8px;
    border-left: 5px solid #0288d1;
    margin: 16px 0;
}
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_DIR = "models_lstm_6class"
BERT_MODEL_NAME = "Jihuai/bert-ancient-chinese"
MAX_LEN = 128

# Hán-Nôm character processing functions
def is_han_nom_char(char):
    """Kiểm tra xem ký tự có phải là Hán-Nôm không"""
    # Kiểm tra các range Unicode cho chữ Hán
    return any([
        '\u4e00' <= char <= '\u9fff',  # CJK Unified Ideographs
        '\u3400' <= char <= '\u4dbf',  # CJK Extension A
        '\u20000' <= char <= '\u2a6df', # CJK Extension B
        '\u2a700' <= char <= '\u2b73f', # CJK Extension C
        '\u2b740' <= char <= '\u2b81f', # CJK Extension D
        '\u2b820' <= char <= '\u2ceaf', # CJK Extension E
        '\u2ceb0' <= char <= '\u2ebef', # CJK Extension F
    ])

def filter_han_nom_text(text):
    """Lọc chỉ giữ lại ký tự Hán-Nôm"""
    return ''.join([char for char in text if is_han_nom_char(char)])

@st.cache_resource
def load_jiayan_models():
    """Load Jiayan models - disabled due to kenlm compatibility issues"""
    st.info("💡 Jiayan processing disabled due to Python 3.13 compatibility. Using basic text processing.")
    return None, None

def preprocess_han_nom_text(text):
    """Tiền xử lý văn bản Hán-Nôm: tách câu và lọc ký tự"""
    # Lọc chỉ giữ ký tự Hán-Nôm
    filtered_text = filter_han_nom_text(text)
    
    if not filtered_text.strip():
        return []
    
    # Tải Jiayan models nếu có
    sentencizer, tokenizer = load_jiayan_models()
    
    sentences = []
    if sentencizer:
        try:
            # Sử dụng Jiayan để tách câu
            sentences = sentencizer.sentencize(filtered_text)
        except Exception as e:
            st.warning(f"Lỗi khi sử dụng Jiayan sentencizer: {e}")
            sentences = []
    
    # Fallback: dùng regex đơn giản
    if not sentences:
        # Tách theo dấu câu truyền thống
        sentences = re.split(r'[,。，；：""（）《》]', filtered_text)
        sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

# Model Definition
class BertLSTMClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=3, dropout=0.5, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        return out

@st.cache_resource
def load_models():
    """Load all models and templates with error handling for Streamlit Cloud"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load class info
        class_info_path = os.path.join(MODEL_DIR, "class_info.json")
        if not os.path.exists(class_info_path):
            st.error(f"❌ Không tìm thấy file {class_info_path}")
            return None
            
        with open(class_info_path, "r") as f:
            class_info = json.load(f)
        
        class_names = class_info["class_names"]
        num_classes = class_info["num_classes"]
        
        # Load BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        bert_model = BertModel.from_pretrained(BERT_MODEL_NAME, use_safetensors=True).to(device)
        bert_model.eval()
        
        # Load LSTM classifier
        lstm_model = BertLSTMClassifier(num_classes=num_classes).to(device)
        model_path = os.path.join(MODEL_DIR, "best_model.pt")
        if not os.path.exists(model_path):
            st.error(f"❌ Không tìm thấy file mô hình {model_path}")
            return None
            
        lstm_model.load_state_dict(torch.load(model_path, map_location=device))
        lstm_model.eval()
        
        # Load templates
        templates = {}
        for i, name in enumerate(class_names):
            template_file = os.path.join(MODEL_DIR, f"template_{name.lower()}.npy")
            if os.path.exists(template_file):
                templates[i] = np.load(template_file)
            else:
                st.warning(f"⚠️ Không tìm thấy template file {template_file}")
                # Tạo template mặc định
                templates[i] = np.random.random((128, 768))
        
        return tokenizer, bert_model, lstm_model, templates, class_names, num_classes, device
        
    except Exception as e:
        st.error(f"❌ Lỗi khi tải models: {e}")
        return None

def extract_bert_features(text, tokenizer, bert_model, device, max_len=128):
    """Extract BERT features from text"""
    encoded = tokenizer(
        text, 
        padding='max_length', 
        truncation=True, 
        max_length=max_len, 
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        features = outputs.last_hidden_state.cpu().numpy()
    
    return features

def make_prob_table(logits, num_classes=6):
    """Convert logits to probability table using softmax"""
    probs = torch.softmax(torch.FloatTensor(logits), dim=-1).numpy()
    return probs

def predict_with_templates(prob_table, templates, num_classes=6):
    """Classify using nearest template (Euclidean distance)"""
    distances = np.zeros((prob_table.shape[0], num_classes))
    
    for class_id in range(num_classes):
        if class_id in templates:
            dist = np.sqrt(np.sum((prob_table - templates[class_id]) ** 2, axis=(1, 2)))
            distances[:, class_id] = dist
        else:
            distances[:, class_id] = np.inf  # Nếu không có template
    
    return np.argmin(distances, axis=1), distances

def classify_text(text, tokenizer, bert_model, lstm_model, templates, class_names, num_classes, device):
    """Classify a single text"""
    # Extract BERT features
    features = extract_bert_features(text, tokenizer, bert_model, device)
    
    # Get LSTM logits
    with torch.no_grad():
        logits = lstm_model(torch.FloatTensor(features).to(device)).cpu().numpy()
    
    # Get probability table
    prob_table = make_prob_table(logits, num_classes)
    
    # Predict with templates
    pred_idx, distances = predict_with_templates(prob_table, templates, num_classes)
    
    # Calculate confidence (inverse of distance, normalized)
    all_dists = distances[0]
    
    # Convert distances to similarity scores (inverse)
    similarities = 1 / (1 + all_dists)
    confidence = similarities / similarities.sum()
    
    return class_names[pred_idx[0]], confidence, pred_idx[0]

# OCR API Configuration - sử dụng external API hoặc secrets
def get_ocr_api_url():
    """Get OCR API URL from secrets or use default"""
    try:
        # Thử lấy từ Streamlit secrets
        return st.secrets.get("OCR_API_URL", "https://kimhannom.clc.hcmus.edu.vn/meta-ocr-normal/nom-ocr")
    except:
        # Fallback URL
        return "https://kimhannom.clc.hcmus.edu.vn/meta-ocr-normal/nom-ocr"

def call_ocr_api(base64_image):
    """Call OCR API with error handling"""
    try:
        headers = {
            'User-Agent': 'StreamlitApp',
            'Content-Type': 'application/json'
        }
        
        if isinstance(base64_image, bytes):
            base64_str = base64.b64encode(base64_image).decode('utf-8')
        else:
            base64_str = base64_image

        payload = {
            "base64Data": base64_str, 
            "lang_type": 2, 
            "reading_direction": 1
        }
        
        ocr_url = get_ocr_api_url()
        response = requests.post(ocr_url, json=payload, headers=headers, verify=False, timeout=120)
        return response
        
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
        # Đảm bảo image_bytes là bytes hợp lệ
        if not isinstance(image_bytes, bytes) or len(image_bytes) == 0:
            return '', None
            
        # Validate image format
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý ảnh: {str(e)}")
        return '', None
    
    # Gọi OCR API
    api_result = call_ocr_api(image_bytes)
    if not api_result:
        return '', None
    
    raw_text = ""
    
    # Xử lý kết quả từ API
    if api_result.status_code == 200:
        try:
            ocr_response_json = api_result.json()
            ocr_text_list = ocr_response_json.get("ocrResult", [])
            raw_text = "\\n".join(ocr_text_list) if ocr_text_list else ""
        except Exception as e:
            st.error(f"❌ Lỗi khi parse JSON response: {e}")
            return '', None
    else:
        st.error(f"❌ OCR API trả về lỗi: {api_result.status_code}")
        return '', None
    
    if raw_text:
        # Hiển thị văn bản đã nhận diện
        st.markdown("**Văn bản đã nhận diện:**")
        st.text_area("Văn bản gốc", value=raw_text, height=120, disabled=True, label_visibility="hidden")

        # Tiền xử lý văn bản Hán-Nôm
        if raw_text.strip():
            processed_sentences = preprocess_han_nom_text(raw_text)
            
            if processed_sentences:
                # Hiển thị các câu sau khi xử lý
                st.markdown("**Văn bản sau khi xử lý:**")
                processed_text_display = '\\n'.join(processed_sentences)
                st.text_area("Văn bản đã xử lý", value=processed_text_display, height=100, disabled=True, label_visibility="hidden")
                
                # Ghép lại thành văn bản hoàn chỉnh để phân loại
                processed_text = ' '.join(processed_sentences)
                return processed_text, api_result
            else:
                st.info("💡 Sử dụng văn bản gốc do không tách được câu Hán-Nôm.")
                return raw_text, api_result
    else:
        st.warning("⚠️ Không phát hiện văn bản trong ảnh.")
    
    return raw_text, api_result

def main():
    st.title("📜 Sino-Nom Text Classification")
    st.markdown("### Phân loại văn bản Hán-Nôm sử dụng mô hình BERT-LSTM")
    
    # Streamlit Cloud info
    st.markdown("""
    <div class="streamlit-info">
        <strong>🚀 Deployed on Streamlit Cloud</strong><br>
        Ứng dụng này sử dụng BERT-LSTM để phân loại văn bản Hán-Nôm thành 6 loại: Y học, Lịch sử, Văn học, Phật giáo, Công giáo, và Khác.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Load models with progress indicator
    model_data = None
    with st.spinner("🔄 Đang tải mô hình... (có thể mất vài phút lần đầu)"):
        model_data = load_models()
    
    if not model_data:
        st.error("❌ Không thể tải mô hình. Vui lòng kiểm tra lại.")
        st.stop()
        
    tokenizer, bert_model, lstm_model, templates, class_names, num_classes, device = model_data
    st.success(f"✅ Đã tải mô hình thành công! (Device: {device})")
    
    # Display class labels
    st.markdown("**Các loại văn bản (Categories):**")
    cols = st.columns(6)
    category_icons = {
        "Medical": "🏥",
        "History": "📚", 
        "Literature": "📖",
        "Buddhism": "🪷",
        "Catholics": "⛪",
        "Others": "📋"
    }
    
    for i, (col, name) in enumerate(zip(cols, class_names)):
        with col:
            st.info(f"{category_icons.get(name, '📄')} {name}")
    
    st.markdown("---")
    
    # Upload image or text input
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
            # Đọc bytes từ file uploader
            image_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, 'getvalue') else uploaded_file.read()
            
            # Layout responsive
            img_col, result_col = st.columns([1, 2])
            
            with img_col:
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image, caption="📷 Ảnh đã upload", use_column_width=True)
                except Exception as e:
                    st.error(f"❌ Lỗi hiển thị ảnh: {str(e)}")
                    st.stop()
            
            with result_col:
                # Thực hiện OCR và phân loại
                with st.spinner("🔄 Đang nhận diện và phân loại..."):
                    text_from_image, ocr_result = run_ocr_on_image(image_bytes)
                
                if text_from_image and text_from_image.strip():
                    # Phân loại văn bản
                    with st.spinner("🤖 Đang phân loại nội dung..."):
                        pred_label, confidence, pred_idx = classify_text(
                            text_from_image, tokenizer, bert_model, lstm_model, 
                            templates, class_names, num_classes, device
                        )
                    
                    # Hiển thị kết quả
                    st.markdown("### 📊 Kết quả phân loại tự động")
                    
                    main_result, confidence_scores = st.columns([1, 1])
                    
                    with main_result:
                        st.markdown(f"""
                        <div class="result-container">
                            <h2 class="result-title">
                                {category_icons.get(pred_label, '📄')} {pred_label}
                            </h2>
                            <p class="result-subtitle">
                                Phân loại: {pred_label}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with confidence_scores:
                        st.markdown('<div class="confidence-container">', unsafe_allow_html=True)
                        st.markdown("**Độ tin cậy:**")
                        sorted_indices = np.argsort(confidence)[::-1]
                        for idx in sorted_indices[:3]:
                            name = class_names[idx]
                            conf = confidence[idx]
                            icon = category_icons.get(name, '📄')
                            if idx == pred_idx:
                                st.success(f"{icon} {name}: {conf:.1%}")
                            else:
                                st.info(f"{icon} {name}: {conf:.1%}")
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Không nhận diện được nội dung từ ảnh.")

    with tab2:
        st.markdown("**Nhập văn bản Hán-Nôm để phân loại:**")
        text_input = st.text_area(
            "Văn bản Hán-Nôm:",
            height=200,
            placeholder="Nhập văn bản Hán-Nôm vào đây...\\n(Ví dụ: 運衰死沙場宣立萬春國)",
            help="Nhập văn bản cần phân loại. Văn bản có thể bằng chữ Hán, chữ Nôm, hoặc hỗn hợp."
        )
        
        if st.button("🔍 Phân loại văn bản", type="primary", key="classify_text"):
            if not text_input.strip():
                st.warning("⚠️ Vui lòng nhập văn bản để phân loại!")
            else:
                with st.spinner("🔄 Đang phân loại..."):
                    pred_label, confidence, pred_idx = classify_text(
                        text_input, tokenizer, bert_model, lstm_model, 
                        templates, class_names, num_classes, device
                    )
                
                st.markdown("---")
                st.markdown("### 📊 Kết quả phân loại")
                
                result_col1, result_col2 = st.columns([1, 1])
                with result_col1:
                    st.markdown(f"""
                    <div class="result-container">
                        <h2 class="result-title">
                            {category_icons.get(pred_label, '📄')} {pred_label}
                        </h2>
                        <p class="result-subtitle">
                            Phân loại: {pred_label}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with result_col2:
                    st.markdown('<div class="confidence-container">', unsafe_allow_html=True)
                    st.markdown("**Độ tin cậy:**")
                    sorted_indices = np.argsort(confidence)[::-1]
                    for idx in sorted_indices[:3]:
                        name = class_names[idx]
                        conf = confidence[idx]
                        icon = category_icons.get(name, '📄')
                        if idx == pred_idx:
                            st.success(f"{icon} {name}: {conf:.1%}")
                        else:
                            st.info(f"{icon} {name}: {conf:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: gray;">
            <p>📜 <strong>Sino-Nom Text Classification</strong> | BERT-LSTM Model</p>
            <p>6 Classes: Medical, History, Literature, Buddhism, Catholics, Others</p>
            <p><em>🚀 Powered by Streamlit Cloud</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()