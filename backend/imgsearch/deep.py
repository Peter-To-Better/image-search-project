"""深度學習特徵。

  resnet50 : ImageNet 預訓練 ResNet-50，取 Global Average Pooling 後的 2048 維向量。
             使用 ONNX Runtime 執行 (不需要安裝 PyTorch)，模型從 ONNX Model Zoo (GitHub) 下載。
  dinov2   : Meta 的自監督 ViT (facebook/dinov2-small, 384 維)。需要 torch + transformers。
  clip     : OpenAI CLIP (openai/clip-vit-base-patch32, 512 維)，同時支援「以文字搜圖」。
             需要 torch + transformers。

所有向量都做 L2 正規化 → 內積 = 餘弦相似度。
"""
import urllib.request

import numpy as np
from PIL import Image

from .config import MODEL_DIR

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

RESNET_URL = ("https://media.githubusercontent.com/media/onnx/models/main/validated/"
              "vision/classification/resnet/model/resnet50-v2-7.onnx")
RESNET_FEAT_OUTPUT = "resnetv24_flatten0_reshape0"   # GAP 之後、分類層之前


def _l2(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def preprocess_imagenet(img: Image.Image, size=224) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.Resampling.BILINEAR)
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    img = img.crop((l, t, l + size, t + size))
    x = (np.asarray(img, np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)


class ResNet50Onnx:
    name, dim = "resnet50", 2048

    def __init__(self):
        import onnxruntime as ort
        path = MODEL_DIR / "resnet50_feat.onnx"
        if not path.exists():
            self._download_and_patch(path)
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    @staticmethod
    def _download_and_patch(path):
        import onnx
        raw = MODEL_DIR / "resnet50-v2-7.onnx"
        if not raw.exists():
            print(f"下載 ResNet-50 ONNX 模型 (~100MB) → {raw}")
            urllib.request.urlretrieve(RESNET_URL, raw)
        m = onnx.load(str(raw))
        # 把中間層 (2048 維特徵) 設成模型輸出
        del m.graph.output[:]
        m.graph.output.append(onnx.helper.make_tensor_value_info(
            RESNET_FEAT_OUTPUT, onnx.TensorProto.FLOAT, None))
        onnx.save(m, str(path))

    def embed_images(self, imgs):
        x = np.stack([preprocess_imagenet(i) for i in imgs]).astype(np.float32)
        out = self.sess.run(None, {self.inp: x})[0].reshape(len(imgs), -1)
        return _l2(out)


def _as_tensor(out):
    """不同版本的 transformers 回傳型態不同 (tensor 或 ModelOutput)，統一取出向量。"""
    if hasattr(out, "shape"):
        return out
    for k in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, k, None)
        if v is not None:
            return v
    return out[0]


class _HF:
    """Hugging Face transformers 模型的共同部分。"""
    hf_id = ""

    def __init__(self):
        try:
            import torch  # noqa
            import transformers  # noqa
        except ImportError as e:
            raise RuntimeError("需要先安裝：pip install torch transformers") from e
        import torch
        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self):
        raise NotImplementedError


class DINOv2(_HF):
    name, dim, hf_id = "dinov2", 384, "facebook/dinov2-small"

    def _load(self):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained(self.hf_id, cache_dir=MODEL_DIR)
        self.model = AutoModel.from_pretrained(self.hf_id, cache_dir=MODEL_DIR).to(self.device).eval()

    def embed_images(self, imgs):
        t = self.proc(images=[i.convert("RGB") for i in imgs], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            h = self.model(**t).last_hidden_state
        # CLS token 與 patch token 平均串接是常見做法，這裡用 CLS 以保持 384 維
        return _l2(h[:, 0].float().cpu().numpy())


class CLIP(_HF):
    name, dim, hf_id = "clip", 512, "openai/clip-vit-base-patch32"

    def _load(self):
        from transformers import CLIPModel, CLIPProcessor
        self.proc = CLIPProcessor.from_pretrained(self.hf_id, cache_dir=MODEL_DIR)
        self.model = CLIPModel.from_pretrained(self.hf_id, cache_dir=MODEL_DIR).to(self.device).eval()

    def embed_images(self, imgs):
        t = self.proc(images=[i.convert("RGB") for i in imgs], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            f = _as_tensor(self.model.get_image_features(**t))
        return _l2(f.float().cpu().numpy())

    def embed_text(self, texts):
        t = self.proc(text=texts, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            f = _as_tensor(self.model.get_text_features(**t))
        return _l2(f.float().cpu().numpy())


DEEP_MODELS = {"resnet50": ResNet50Onnx, "dinov2": DINOv2, "clip": CLIP}
