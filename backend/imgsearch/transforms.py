"""產生「近似副本」查詢圖用的影像變形，用來測試各方法的穩健性。"""
import io
import random

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


def rotate(img, rng: random.Random, deg=None):
    deg = deg if deg is not None else rng.choice([-15, -10, 10, 15])
    return img.rotate(deg, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(255, 255, 255))


def crop(img, rng: random.Random, ratio=0.75):
    w, h = img.size
    cw, ch = int(w * ratio), int(h * ratio)
    l, t = rng.randint(0, w - cw), rng.randint(0, h - ch)
    return img.crop((l, t, l + cw, t + ch))


def brightness(img, rng: random.Random, factor=None):
    return ImageEnhance.Brightness(img).enhance(factor or rng.choice([0.6, 1.5]))


def jpeg(img, rng: random.Random, quality=10):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def blur_scale(img, rng: random.Random):
    w, h = img.size
    small = img.resize((max(1, w // 3), max(1, h // 3)))
    return small.filter(ImageFilter.GaussianBlur(1))


def watermark(img, rng: random.Random):
    img = img.copy()
    d = ImageDraw.Draw(img)
    w, h = img.size
    d.rectangle([0, int(h * 0.8), w, h], fill=(0, 0, 0))
    d.text((8, int(h * 0.83)), "COPYRIGHT 2026 SAMPLE", fill=(255, 255, 255))
    return img


def flip(img, rng: random.Random):
    return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


TRANSFORMS = {
    "rotate": rotate,
    "crop": crop,
    "brightness": brightness,
    "jpeg": jpeg,
    "blur_scale": blur_scale,
    "watermark": watermark,
    "flip": flip,
}

TRANSFORM_LABELS = {
    "rotate": "旋轉 ±10~15°", "crop": "裁切 75%", "brightness": "亮度 ×0.6 / ×1.5",
    "jpeg": "JPEG 壓縮 (品質 10)", "blur_scale": "縮小 1/3 + 模糊", "watermark": "加浮水印字條",
    "flip": "左右翻轉",
}
