import os
import re
import shutil
import mimetypes
from collections import deque
from datetime import datetime
from difflib import SequenceMatcher
from io import BytesIO
from typing import Dict, List, Tuple
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import easyocr
import numpy as np
import pandas as pd
import streamlit as st

try:
    import enchant
except Exception:
    enchant = None

st.set_page_config(page_title="视频字幕提取工作站", layout="wide")
st.title("视频字幕自动提取工作站")
st.caption("OpenCV + EasyOCR | 输出: 视频名称 / 视频原字幕")

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
}
MAX_BATCH_VIDEOS = 200
UPLOAD_ROOT_DIR = os.path.join(os.getcwd(), "_uploaded_videos")
DOWNLOAD_ROOT_DIR = os.path.join(os.getcwd(), "_downloaded_videos")
BATCH_OUTPUT_ROOT_DIR = os.path.join(os.getcwd(), "batch_outputs")
URL_PREFIXES = ("http://", "https://")


@st.cache_resource(show_spinner=False)
def get_ocr_reader(use_gpu: bool) -> easyocr.Reader:
    return easyocr.Reader(["en"], gpu=use_gpu)


@st.cache_resource(show_spinner=False)
def get_en_dict():
    if enchant is None:
        return None
    try:
        return enchant.Dict("en_US")
    except Exception:
        return None


def ensure_video_path(path_text: str) -> str:
    path_text = path_text.strip().strip('"')
    return os.path.abspath(path_text) if path_text else ""


def normalize_input_line(text: str) -> str:
    return text.strip().strip('"').strip("'")


def is_http_url(text: str) -> bool:
    return normalize_input_line(text).lower().startswith(URL_PREFIXES)


def is_video_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS


def unique_paths(paths: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for path in paths:
        abspath = os.path.abspath(path)
        key = os.path.normcase(abspath)
        if key in seen:
            continue
        seen.add(key)
        out.append(abspath)
    return out


def list_videos_in_directory(dir_path: str) -> List[str]:
    found: List[str] = []
    for root, _, files in os.walk(dir_path):
        for name in files:
            full_path = os.path.join(root, name)
            if is_video_file(full_path):
                found.append(os.path.abspath(full_path))
    return sorted(found)


def parse_path_lines(path_text: str) -> List[str]:
    out: List[str] = []
    for raw in path_text.splitlines():
        normalized = ensure_video_path(raw)
        if normalized:
            out.append(normalized)
    return out


def parse_url_lines(url_text: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in url_text.splitlines():
        normalized = normalize_input_line(raw)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def make_timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_filename(name: str) -> str:
    clean = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", name).strip(" .")
    return clean or "video"


def ensure_directory(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_uploaded_videos(uploaded_files, upload_dir: str) -> List[str]:
    ensure_directory(upload_dir)
    saved_paths: List[str] = []
    for idx, uploaded in enumerate(uploaded_files, start=1):
        ext = os.path.splitext(str(uploaded.name))[1].lower()
        if ext not in VIDEO_EXTENSIONS:
            continue
        stem = sanitize_filename(os.path.splitext(str(uploaded.name))[0])
        filename = f"{idx:03d}_{stem}{ext}"
        target_path = os.path.join(upload_dir, filename)
        with open(target_path, "wb") as f:
            f.write(uploaded.getbuffer())
        saved_paths.append(target_path)
    return saved_paths


def guess_download_extension(url: str, headers) -> str:
    url_ext = os.path.splitext(urlparse(url).path)[1].lower()
    if url_ext in VIDEO_EXTENSIONS:
        return url_ext

    content_type = str(headers.get("Content-Type", "")).split(";", 1)[0].strip().lower()
    guessed = mimetypes.guess_extension(content_type) if content_type else ""
    if guessed in VIDEO_EXTENSIONS:
        return guessed

    return ".mp4"


def build_download_filename(url: str, headers, order: int) -> str:
    parsed = urlparse(url)
    raw_name = os.path.basename(unquote(parsed.path)).strip()
    stem = ""
    ext = ""
    if raw_name:
        stem, ext = os.path.splitext(raw_name)
        ext = ext.lower()

    if ext not in VIDEO_EXTENSIONS:
        ext = guess_download_extension(url, headers)
    stem = sanitize_filename(stem) if stem else f"video_{order:03d}"
    return f"{order:03d}_{stem}{ext}"


def download_video_from_url(url: str, download_dir: str, order: int) -> str:
    ensure_directory(download_dir)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=180) as response:
        target_path = os.path.join(download_dir, build_download_filename(url, response.headers, order))
        with open(target_path, "wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    if not os.path.exists(target_path) or os.path.getsize(target_path) <= 0:
        raise RuntimeError("下载结果为空文件")
    return os.path.abspath(target_path)


def collect_batch_video_paths(path_text: str, uploaded_files) -> Tuple[List[str], List[str]]:
    resolved_paths: List[str] = []
    notes: List[str] = []

    for entry in parse_path_lines(path_text):
        if not os.path.exists(entry):
            notes.append(f"路径不存在，已跳过：{entry}")
            continue
        if os.path.isfile(entry):
            if is_video_file(entry):
                resolved_paths.append(entry)
            else:
                notes.append(f"不是支持的视频文件，已跳过：{entry}")
            continue

        dir_videos = list_videos_in_directory(entry)
        if dir_videos:
            resolved_paths.extend(dir_videos)
        else:
            notes.append(f"文件夹内未找到视频文件，已跳过：{entry}")

    if uploaded_files:
        upload_dir = os.path.join(UPLOAD_ROOT_DIR, make_timestamp_slug())
        resolved_paths.extend(save_uploaded_videos(uploaded_files, upload_dir))

    resolved_paths = unique_paths(resolved_paths)
    if len(resolved_paths) > MAX_BATCH_VIDEOS:
        raise ValueError(f"本次批处理最多支持 {MAX_BATCH_VIDEOS} 个视频，请分批处理。")

    return resolved_paths, notes


def collect_batch_video_urls(url_text: str) -> Tuple[List[str], List[str]]:
    urls = parse_url_lines(url_text)
    if len(urls) > MAX_BATCH_VIDEOS:
        raise ValueError(f"本次批处理最多支持 {MAX_BATCH_VIDEOS} 个视频，请分批处理。")

    download_dir = os.path.join(DOWNLOAD_ROOT_DIR, make_timestamp_slug())
    resolved_paths: List[str] = []
    notes: List[str] = []

    for order, url in enumerate(urls, start=1):
        if not is_http_url(url):
            notes.append(f"不是有效的视频链接，已跳过：{url}")
            continue
        try:
            resolved_paths.append(download_video_from_url(url, download_dir, order))
        except Exception as exc:
            notes.append(f"链接下载失败，已跳过：{url} | {exc}")

    resolved_paths = unique_paths(resolved_paths)
    if not resolved_paths and os.path.isdir(download_dir):
        shutil.rmtree(download_dir, ignore_errors=True)
    return resolved_paths, notes


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_output_text(text: str) -> str:
    text = text.replace("_", " ")
    text = clean_text(text)
    text = re.sub(r"\s+([,.:;!?])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]}])", r"\1", text)
    return text


def valid_text(text: str, min_core_chars: int) -> bool:
    core = re.sub(r"[^A-Za-z0-9]+", "", text)
    return len(core) >= min_core_chars


def preprocess_gray_roi(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 35, 35)
    gray = cv2.equalizeHist(gray)
    return gray


def preprocess_white_roi(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    bw = cv2.resize(bw, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
    return bw


def preprocess_white_roi_alt(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        29,
        9,
    )
    kernel = np.ones((2, 2), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    bw = cv2.resize(bw, None, fx=1.20, fy=1.20, interpolation=cv2.INTER_CUBIC)
    return bw


def group_to_lines(
    detections: List[Tuple[List[List[float]], str, float]],
    roi_h: int,
    min_core_chars: int,
    conf_threshold: float,
    box_conf_floor: float,
) -> List[Tuple[str, float]]:
    if not detections:
        return []

    boxes: List[Dict[str, float]] = []
    for bbox, raw_text, conf in detections:
        text = clean_text(raw_text)
        if not text:
            continue
        if float(conf) < box_conf_floor:
            continue
        if not valid_text(text, min_core_chars):
            continue

        xs = [float(p[0]) for p in bbox]
        ys = [float(p[1]) for p in bbox]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        h_box = max(1.0, y2 - y1)
        if h_box < max(8.0, roi_h * 0.040):
            continue

        boxes.append(
            {
                "text": text,
                "x1": x1,
                "x2": x2,
                "y": (y1 + y2) / 2.0,
                "conf": float(conf),
            }
        )

    if not boxes:
        return []

    boxes.sort(key=lambda b: (b["y"], b["x1"]))
    line_tol = max(10.0, roi_h * 0.038)

    lines: List[Dict[str, object]] = []
    for b in boxes:
        hit = None
        for line in lines:
            if abs(b["y"] - line["y_mean"]) <= line_tol:
                hit = line
                break
        if hit is None:
            lines.append({"y_mean": b["y"], "items": [b]})
        else:
            items = hit["items"]
            items.append(b)
            hit["y_mean"] = sum(i["y"] for i in items) / len(items)

    lines.sort(key=lambda line: line["y_mean"])
    out: List[Tuple[str, float]] = []
    for line in lines:
        items = sorted(line["items"], key=lambda i: i["x1"])
        merged = normalize_output_text(" ".join(item["text"] for item in items))
        if not valid_text(merged, min_core_chars):
            continue

        avg_conf = float(sum(item["conf"] for item in items) / len(items))
        if avg_conf < conf_threshold:
            continue
        out.append((merged, avg_conf))

    return out


def extract_lines_from_roi(
    reader: easyocr.Reader,
    roi_bgr: np.ndarray,
    mode: str,
    min_core_chars: int,
    conf_threshold: float,
) -> List[Tuple[str, float]]:
    roi_h = roi_bgr.shape[0]

    if mode == "gray":
        img = preprocess_gray_roi(roi_bgr)
        detections = reader.readtext(
            img,
            detail=1,
            paragraph=False,
            decoder="greedy",
            text_threshold=0.60,
            low_text=0.20,
            link_threshold=0.30,
            width_ths=0.80,
            height_ths=0.80,
        )
        box_floor = max(0.24, min(conf_threshold - 0.18, 0.72))
        return group_to_lines(detections, roi_h, min_core_chars, conf_threshold, box_floor)

    img1 = preprocess_white_roi(roi_bgr)

    # 白字独立阈值：在全局0.75时适度放宽，减少漏首字母
    white_conf_threshold = max(0.52, min(0.72, conf_threshold - 0.10))
    white_min_core_chars = max(3, min_core_chars)

    det1 = reader.readtext(
        img1,
        detail=1,
        paragraph=False,
        decoder="greedy",
        text_threshold=0.45,
        low_text=0.10,
        link_threshold=0.18,
        width_ths=0.95,
        height_ths=0.95,
    )
    img2 = preprocess_white_roi_alt(roi_bgr)
    det2 = reader.readtext(
        img2,
        detail=1,
        paragraph=False,
        decoder="greedy",
        text_threshold=0.40,
        low_text=0.08,
        link_threshold=0.16,
        width_ths=0.98,
        height_ths=0.98,
    )

    merged = list(det1) + list(det2)
    box_floor = max(0.18, min(white_conf_threshold - 0.20, 0.66))
    return group_to_lines(merged, roi_h, white_min_core_chars, white_conf_threshold, box_floor)


def get_target_rois(frame: np.ndarray) -> List[Tuple[np.ndarray, str]]:
    h, w = frame.shape[:2]
    y_gray = int(h * 0.78)
    y_upper = int(h * 0.60)
    y_upper_bottom = int(h * 0.82)
    x_mid = int(w * 0.5)
    overlap = int(w * 0.08)

    rois = [
        (frame[y_gray:h, 0:w], "gray"),
        (frame[y_upper:y_upper_bottom, 0:min(w, x_mid + overlap)], "white"),
        (frame[y_upper:y_upper_bottom, max(0, x_mid - overlap):w], "white"),
    ]
    return [(roi, mode) for roi, mode in rois if roi.size > 0]


def text_similarity(a: str, b: str) -> float:
    a_norm = re.sub(r"[^a-z0-9]+", " ", a.lower()).strip()
    b_norm = re.sub(r"[^a-z0-9]+", " ", b.lower()).strip()
    if not a_norm or not b_norm:
        return 0.0
    return SequenceMatcher(None, a_norm, b_norm).ratio()


def mergeable(a: str, b: str) -> bool:
    a_key = re.sub(r"\s+", " ", a).strip().lower()
    b_key = re.sub(r"\s+", " ", b).strip().lower()
    if a_key == b_key:
        return True

    a_core = re.sub(r"[^a-z0-9]+", "", a_key)
    b_core = re.sub(r"[^a-z0-9]+", "", b_key)
    if min(len(a_core), len(b_core)) < 8:
        return False

    sim = text_similarity(a, b)
    if sim >= 0.86:
        return True

    short, long_ = (a_core, b_core) if len(a_core) <= len(b_core) else (b_core, a_core)
    return short in long_ and sim >= 0.72


def choose_best_suggestion(token: str, suggestions: List[str]) -> str:
    if not suggestions:
        return token
    t_low = token.lower()
    best = token
    best_score = 0.0
    for cand in suggestions[:8]:
        cand_low = cand.lower()
        if not cand_low.isalpha():
            continue
        if abs(len(cand_low) - len(t_low)) > 2:
            continue
        sim = SequenceMatcher(None, t_low, cand_low).ratio()
        if sim > best_score:
            best_score = sim
            best = cand
    return best if best_score >= 0.72 else token


def one_edit_valid_words(token: str, en_dict) -> List[str]:
    letters = "abcdefghijklmnopqrstuvwxyz"
    out = set()
    n = len(token)

    # replace
    for i in range(n):
        for ch in letters:
            if ch == token[i]:
                continue
            cand = token[:i] + ch + token[i + 1:]
            if en_dict.check(cand):
                out.add(cand)

    # delete
    for i in range(n):
        cand = token[:i] + token[i + 1:]
        if len(cand) >= 3 and en_dict.check(cand):
            out.add(cand)

    # transpose
    for i in range(n - 1):
        if token[i] == token[i + 1]:
            continue
        cand = token[:i] + token[i + 1] + token[i] + token[i + 2:]
        if en_dict.check(cand):
            out.add(cand)

    return list(out)


def edit_distance_limited(a: str, b: str, max_dist: int = 2) -> int:
    # 轻量编辑距离：只用于短词纠错筛选
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        row_min = dp[0]
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # delete
                dp[j - 1] + 1,  # insert
                prev + cost,    # replace
            )
            prev = cur
            row_min = min(row_min, dp[j])
        if row_min > max_dist:
            return max_dist + 1
    return dp[lb]


def apply_dict_correction(text: str, conf: float, enable: bool) -> str:
    if not enable:
        return text
    en_dict = get_en_dict()
    if en_dict is None:
        return text

    def repl(match: re.Match) -> str:
        token = match.group(0)
        if len(token) < 4:
            return token
        low = token.lower()
        if en_dict.check(low):
            return token

        # 通用规则：若去掉末尾1个字母后是合法词，优先修正（如 Rakel -> Rake）
        if len(low) >= 5 and en_dict.check(low[:-1]):
            fixed = low[:-1]
            if token.isupper():
                return fixed.upper()
            if token[0].isupper():
                return fixed.capitalize()
            return fixed

        suggestions = en_dict.suggest(low)
        # 额外候选：单步编辑可达的合法词，覆盖中间错字（如 dyty -> duty）
        suggestions.extend(one_edit_valid_words(low, en_dict))
        # 去重并保持顺序
        seen = set()
        uniq_suggestions = []
        for s in suggestions:
            sl = str(s).lower()
            if sl in seen:
                continue
            seen.add(sl)
            uniq_suggestions.append(str(s))
        suggestions = uniq_suggestions
        # 优先选择编辑距离最小的建议，覆盖中间字符错误（如 Dyty -> Duty）
        best_by_dist = None
        best_dist = 3
        for cand in suggestions[:20]:
            c = cand.lower()
            if not c.isalpha():
                continue
            d = edit_distance_limited(low, c, max_dist=2)
            if d < best_dist:
                best_dist = d
                best_by_dist = cand
                if d == 1:
                    break
        if best_by_dist is not None and best_dist <= 1:
            fixed = best_by_dist
            if token.isupper():
                return fixed.upper()
            if token[0].isupper():
                return fixed.capitalize()
            return fixed

        fixed = choose_best_suggestion(low, suggestions)
        if fixed == low:
            return token
        if token.isupper():
            return fixed.upper()
        if token[0].isupper():
            return fixed.capitalize()
        return fixed

    return re.sub(r"\b[A-Za-z]{4,}\b", repl, text)


def text_quality(text: str, conf: float) -> float:
    core_len = len(re.sub(r"[^A-Za-z0-9]+", "", text))
    word_cnt = len([w for w in text.split(" ") if w])
    return conf * 2.0 + core_len * 0.03 + word_cnt * 0.02


def pick_cluster_best_text(cluster: Dict[str, object]) -> str:
    variants: Dict[str, Dict[str, float]] = cluster["variants"]
    best_text = str(cluster["best_text"])
    best_score = -1.0
    for txt, info in variants.items():
        count = float(info["count"])
        conf_avg = float(info["conf_sum"]) / max(1.0, count)
        core_len = float(info["core_len"])
        vote_score = count * 2.0 + conf_avg * 1.2 + core_len * 0.02
        if vote_score > best_score:
            best_score = vote_score
            best_text = txt
    return best_text


def dedupe_in_frame(lines: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    deduped: List[Tuple[str, float]] = []
    seen = set()
    for txt, cf in lines:
        key = re.sub(r"\s+", " ", txt).strip().lower()
        if key in seen:
            continue
        deduped.append((txt, cf))
        seen.add(key)
    return deduped


def white_consensus_lines(job: Dict[str, object], current_frame_idx: int) -> List[Tuple[str, float]]:
    history: List[Dict[str, object]] = job["white_history"]
    if not history:
        return []

    window = int(job["white_vote_window"])
    min_hits = int(job["white_vote_min_hits"])
    recent = history[-window:]

    candidates: List[Tuple[int, str, float]] = []
    for entry in recent:
        fidx = int(entry["frame"])
        for txt, cf in entry["lines"]:
            candidates.append((fidx, str(txt), float(cf)))

    if not candidates:
        return []

    clusters: List[Dict[str, object]] = []
    for fidx, txt, cf in candidates:
        hit = None
        for c in clusters:
            if mergeable(txt, str(c["rep"])):
                hit = c
                break
        if hit is None:
            clusters.append(
                {
                    "rep": txt,
                    "best_text": txt,
                    "best_score": text_quality(txt, cf),
                    "count": 1,
                    "conf_sum": cf,
                    "last_frame": fidx,
                }
            )
        else:
            hit["count"] = int(hit["count"]) + 1
            hit["conf_sum"] = float(hit["conf_sum"]) + cf
            hit["last_frame"] = max(int(hit["last_frame"]), fidx)
            score = text_quality(txt, cf)
            if score > float(hit["best_score"]):
                hit["best_score"] = score
                hit["best_text"] = txt
            old_core = len(re.sub(r"[^A-Za-z0-9]+", "", str(hit["rep"])))
            new_core = len(re.sub(r"[^A-Za-z0-9]+", "", txt))
            if new_core >= old_core:
                hit["rep"] = txt

    out: List[Tuple[str, float]] = []
    for c in clusters:
        # 只输出当前帧仍在出现且命中次数满足阈值的白字结果
        if int(c["last_frame"]) != current_frame_idx:
            continue
        if int(c["count"]) < min_hits:
            continue
        conf_avg = float(c["conf_sum"]) / max(1, int(c["count"]))
        out.append((str(c["best_text"]), conf_avg))

    return dedupe_in_frame(out)


def init_job_state() -> None:
    st.session_state.job = {
        "active": False,
        "paused": False,
        "finished": False,
        "video_path": "",
        "video_name": "",
        "use_gpu": True,
        "sample_every_n_frames": 5,
        "min_core_chars": 2,
        "conf_threshold": 0.5,
        "dedupe_global": True,
        "min_stable_hits": 2,
        "white_vote_window": 7,
        "white_vote_min_hits": 2,
        "use_dict_correction": True,
        "frame_idx": 0,
        "total_frames": 0,
        "rows": [],
        "recent_texts": [],
        "seen_texts": [],
        "clusters": [],
        "white_history": [],
        "started_at": None,
    }


def init_batch_state() -> None:
    st.session_state.batch = {
        "enabled": False,
        "queue": [],
        "config": {},
        "current_index": 0,
        "completed_count": 0,
        "results": [],
        "all_rows": [],
        "output_dir": "",
        "zip_bytes": b"",
        "notes": [],
        "started_at": None,
        "uploaded_dir": "",
        "finished": False,
    }


def ensure_job_state() -> None:
    if "job" not in st.session_state:
        init_job_state()


def ensure_batch_state() -> None:
    if "batch" not in st.session_state:
        init_batch_state()


def start_job(
    video_path: str,
    use_gpu: bool,
    sample_every_n_frames: int,
    min_core_chars: int,
    conf_threshold: float,
    dedupe_global: bool,
    min_stable_hits: int,
    white_vote_window: int,
    white_vote_min_hits: int,
    use_dict_correction: bool,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    st.session_state.job = {
        "active": True,
        "paused": False,
        "finished": False,
        "video_path": video_path,
        "video_name": os.path.basename(video_path),
        "use_gpu": use_gpu,
        "sample_every_n_frames": sample_every_n_frames,
        "min_core_chars": min_core_chars,
        "conf_threshold": conf_threshold,
        "dedupe_global": dedupe_global,
        "min_stable_hits": min_stable_hits,
        "white_vote_window": white_vote_window,
        "white_vote_min_hits": white_vote_min_hits,
        "use_dict_correction": use_dict_correction,
        "frame_idx": 0,
        "total_frames": total_frames,
        "rows": [],
        "recent_texts": [],
        "seen_texts": [],
        "clusters": [],
        "white_history": [],
        "started_at": datetime.now().isoformat(),
    }


def make_job_config(
    use_gpu: bool,
    sample_every_n_frames: int,
    min_core_chars: int,
    conf_threshold: float,
    dedupe_global: bool,
    min_stable_hits: int,
    white_vote_window: int,
    white_vote_min_hits: int,
    use_dict_correction: bool,
) -> Dict[str, object]:
    return {
        "use_gpu": bool(use_gpu),
        "sample_every_n_frames": int(sample_every_n_frames),
        "min_core_chars": int(min_core_chars),
        "conf_threshold": float(conf_threshold),
        "dedupe_global": bool(dedupe_global),
        "min_stable_hits": int(min_stable_hits),
        "white_vote_window": int(white_vote_window),
        "white_vote_min_hits": int(white_vote_min_hits),
        "use_dict_correction": bool(use_dict_correction),
    }


def get_managed_temp_dir(path: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    managed_roots = [os.path.abspath(UPLOAD_ROOT_DIR), os.path.abspath(DOWNLOAD_ROOT_DIR)]
    current_dir = os.path.normcase(parent)
    for root in managed_roots:
        if current_dir.startswith(os.path.normcase(root)):
            return parent
    return ""


def start_batch_run(video_paths: List[str], config: Dict[str, object], notes: List[str]) -> None:
    output_dir = ensure_directory(os.path.join(BATCH_OUTPUT_ROOT_DIR, make_timestamp_slug()))
    uploaded_dir = ""
    for path in video_paths:
        temp_dir = get_managed_temp_dir(path)
        if temp_dir:
            uploaded_dir = temp_dir
            break

    st.session_state.batch = {
        "enabled": True,
        "queue": list(video_paths),
        "config": dict(config),
        "current_index": 0,
        "completed_count": 0,
        "results": [],
        "all_rows": [],
        "output_dir": output_dir,
        "zip_bytes": b"",
        "notes": list(notes),
        "started_at": datetime.now().isoformat(),
        "uploaded_dir": uploaded_dir,
        "finished": False,
    }
    start_job(video_path=video_paths[0], **config)


def build_video_dataframe(rows: List[Dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["视频名称", "视频原字幕"])


def export_video_result_files(
    order: int,
    video_name: str,
    rows: List[Dict[str, str]],
    output_dir: str,
) -> Dict[str, object]:
    df = build_video_dataframe(rows)
    stem = sanitize_filename(os.path.splitext(video_name)[0])
    base_name = f"{order:03d}_{stem}"
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    excel_path = os.path.join(output_dir, f"{base_name}.xlsx")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="results")

    return {
        "视频名称": video_name,
        "字幕条数": int(len(rows)),
        "CSV": csv_path,
        "Excel": excel_path,
    }


def export_batch_summary_files(batch: Dict[str, object]) -> None:
    output_dir = str(batch["output_dir"])
    all_df = build_video_dataframe(batch["all_rows"])
    all_df.to_csv(os.path.join(output_dir, "all_subtitle_results.csv"), index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(os.path.join(output_dir, "all_subtitle_results.xlsx"), engine="openpyxl") as writer:
        all_df.to_excel(writer, index=False, sheet_name="results")

    summary_df = pd.DataFrame(batch["results"], columns=["视频名称", "字幕条数", "CSV", "Excel"])
    summary_df.to_csv(os.path.join(output_dir, "batch_summary.csv"), index=False, encoding="utf-8-sig")


def build_zip_bytes(dir_path: str) -> bytes:
    bio = BytesIO()
    with ZipFile(bio, "w", compression=ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(dir_path):
            for name in sorted(files):
                full_path = os.path.join(root, name)
                arcname = os.path.relpath(full_path, dir_path)
                zf.write(full_path, arcname)
    return bio.getvalue()


def finalize_finished_batch_video() -> None:
    batch = st.session_state.batch
    job = st.session_state.job
    if not bool(batch["enabled"]) or not bool(job["finished"]):
        return
    if int(batch["completed_count"]) > int(batch["current_index"]):
        return

    result = export_video_result_files(
        order=int(batch["current_index"]) + 1,
        video_name=str(job["video_name"]),
        rows=list(job["rows"]),
        output_dir=str(batch["output_dir"]),
    )
    batch["results"].append(result)
    batch["all_rows"].extend(list(job["rows"]))
    batch["completed_count"] = int(batch["current_index"]) + 1

    if int(batch["completed_count"]) < len(batch["queue"]):
        batch["current_index"] = int(batch["completed_count"])
        next_path = str(batch["queue"][int(batch["current_index"])])
        start_job(video_path=next_path, **dict(batch["config"]))
        return

    export_batch_summary_files(batch)
    batch["zip_bytes"] = build_zip_bytes(str(batch["output_dir"]))
    batch["finished"] = True


def finalize_old_clusters(job: Dict[str, object], current_frame_idx: int, force: bool = False) -> None:
    clusters: List[Dict[str, object]] = job["clusters"]
    rows: List[Dict[str, str]] = job["rows"]
    recent_texts = deque(job["recent_texts"], maxlen=12)
    seen_texts = set(job["seen_texts"])

    finalize_gap = max(int(job["sample_every_n_frames"]) * int(job["white_vote_window"]), 12)

    for cluster in clusters:
        if cluster["emitted"]:
            continue

        inactive = current_frame_idx - int(cluster["last_frame"])
        if not force and inactive < finalize_gap:
            continue

        if int(cluster["hits"]) < int(job["min_stable_hits"]):
            if float(cluster["best_score"]) < 1.95:
                cluster["emitted"] = True
                continue

        out_text = pick_cluster_best_text(cluster)
        if out_text in recent_texts:
            cluster["emitted"] = True
            continue
        if bool(job["dedupe_global"]) and out_text in seen_texts:
            cluster["emitted"] = True
            continue

        rows.append({"视频名称": str(job["video_name"]), "视频原字幕": out_text})
        recent_texts.append(out_text)
        seen_texts.add(out_text)
        cluster["emitted"] = True

    job["recent_texts"] = list(recent_texts)
    job["seen_texts"] = list(seen_texts)


def process_batch(batch_samples: int = 35) -> None:
    job = st.session_state.job
    if not job["active"] or job["paused"] or job["finished"]:
        return

    video_path = str(job["video_path"])
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频不存在: {video_path}")

    reader = get_ocr_reader(bool(job["use_gpu"]))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件")

    frame_idx = int(job["frame_idx"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    sampled = 0
    while sampled < batch_samples:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % int(job["sample_every_n_frames"]) != 0:
            frame_idx += 1
            continue

        gray_lines: List[Tuple[str, float]] = []
        white_lines_raw: List[Tuple[str, float]] = []
        for roi, mode in get_target_rois(frame):
            extracted = extract_lines_from_roi(
                reader=reader,
                roi_bgr=roi,
                mode=mode,
                min_core_chars=int(job["min_core_chars"]),
                conf_threshold=float(job["conf_threshold"]),
            )
            if mode == "white":
                white_lines_raw.extend(extracted)
            else:
                gray_lines.extend(extracted)

        gray_lines = dedupe_in_frame(gray_lines)
        white_lines_raw = dedupe_in_frame(white_lines_raw)
        if bool(job["use_dict_correction"]):
            white_lines_raw = [
                (apply_dict_correction(txt, cf, True), cf) for txt, cf in white_lines_raw
            ]
            white_lines_raw = dedupe_in_frame(white_lines_raw)

        history: List[Dict[str, object]] = job["white_history"]
        history.append({"frame": frame_idx, "lines": white_lines_raw})
        if len(history) > 120:
            del history[:-120]

        white_lines = white_consensus_lines(job, frame_idx)
        line_texts = dedupe_in_frame(gray_lines + white_lines)

        clusters: List[Dict[str, object]] = job["clusters"]
        for line_text, line_conf in line_texts:
            hit = None
            for cluster in clusters:
                if mergeable(line_text, str(cluster["rep_text"])):
                    hit = cluster
                    break

            score = text_quality(line_text, line_conf)
            if hit is None:
                clusters.append(
                    {
                        "rep_text": line_text,
                        "best_text": line_text,
                        "best_score": score,
                        "hits": 1,
                        "last_frame": frame_idx,
                        "emitted": False,
                        "variants": {
                            line_text: {
                                "count": 1.0,
                                "conf_sum": float(line_conf),
                                "core_len": float(len(re.sub(r"[^A-Za-z0-9]+", "", line_text))),
                            }
                        },
                    }
                )
            else:
                hit["hits"] = int(hit["hits"]) + 1
                hit["last_frame"] = frame_idx
                if score > float(hit["best_score"]):
                    hit["best_score"] = score
                    hit["best_text"] = line_text

                old_core = len(re.sub(r"[^A-Za-z0-9]+", "", str(hit["rep_text"])))
                new_core = len(re.sub(r"[^A-Za-z0-9]+", "", line_text))
                if new_core >= old_core:
                    hit["rep_text"] = line_text

                variants: Dict[str, Dict[str, float]] = hit["variants"]
                if line_text not in variants:
                    variants[line_text] = {
                        "count": 0.0,
                        "conf_sum": 0.0,
                        "core_len": float(len(re.sub(r"[^A-Za-z0-9]+", "", line_text))),
                    }
                variants[line_text]["count"] += 1.0
                variants[line_text]["conf_sum"] += float(line_conf)

        finalize_old_clusters(job, frame_idx, force=False)

        sampled += 1
        frame_idx += 1

    cap.release()

    job["frame_idx"] = frame_idx
    if frame_idx >= int(job["total_frames"]):
        finalize_old_clusters(job, frame_idx, force=True)
        job["finished"] = True
        job["active"] = False


def render_results() -> None:
    job = st.session_state.job
    df = pd.DataFrame(job["rows"], columns=["视频名称", "视频原字幕"])
    st.dataframe(df, use_container_width=True, height=420)

    if not df.empty:
        csv_data = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="导出 CSV",
            data=csv_data,
            file_name="subtitle_results.csv",
            mime="text/csv",
        )

        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="results")
        st.download_button(
            label="导出 Excel",
            data=bio.getvalue(),
            file_name="subtitle_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def render_batch_results() -> None:
    batch = st.session_state.batch
    summary_df = pd.DataFrame(batch["results"], columns=["视频名称", "字幕条数", "CSV", "Excel"])
    if not summary_df.empty:
        st.dataframe(summary_df[["视频名称", "字幕条数"]], use_container_width=True, height=320)

    st.caption(f"批处理输出目录：{batch['output_dir']}")
    if batch["notes"]:
        for note in batch["notes"]:
            st.warning(note)

    if batch["zip_bytes"]:
        zip_name = f"subtitle_batch_{os.path.basename(str(batch['output_dir']))}.zip"
        st.download_button(
            label="下载全部结果 ZIP",
            data=batch["zip_bytes"],
            file_name=zip_name,
            mime="application/zip",
        )


def reset_job() -> None:
    ensure_batch_state()
    uploaded_dir = str(st.session_state.batch.get("uploaded_dir", ""))
    if uploaded_dir and os.path.isdir(uploaded_dir):
        current_dir = os.path.normcase(os.path.abspath(uploaded_dir))
        managed_roots = [
            os.path.normcase(os.path.abspath(UPLOAD_ROOT_DIR)),
            os.path.normcase(os.path.abspath(DOWNLOAD_ROOT_DIR)),
        ]
        if any(current_dir.startswith(root) for root in managed_roots):
            shutil.rmtree(uploaded_dir, ignore_errors=True)
    init_job_state()
    init_batch_state()


ensure_job_state()
ensure_batch_state()
job = st.session_state.job
batch = st.session_state.batch

with st.sidebar:
    st.subheader("参数配置")
    processing_mode = st.radio("处理模式", ["单视频", "批处理", "在线链接"], horizontal=True)
    if processing_mode == "单视频":
        video_path_input = st.text_input(
            "本地视频绝对路径",
            value=r"D:\视频\your_video.mp4",
            help="示例: D:\\videos\\demo.mp4",
        )
        batch_path_input = ""
        uploaded_videos = []
        online_url_input = ""
    elif processing_mode == "批处理":
        uploaded_videos = st.file_uploader(
            "拖入视频文件（可多选）",
            type=[ext.lstrip(".") for ext in sorted(VIDEO_EXTENSIONS)],
            accept_multiple_files=True,
            help="适合直接拖多个视频到页面。文件会先缓存到本地工作目录再识别；文件夹请在下面填写绝对路径。",
        )
        batch_path_input = st.text_area(
            "本地文件/文件夹绝对路径（每行一个，可混填）",
            value="",
            height=120,
            help="支持单个视频路径，也支持文件夹路径；文件夹会递归扫描视频。",
        )
        st.caption(f"批处理最多 {MAX_BATCH_VIDEOS} 个视频，超出请分批执行。")
        online_url_input = ""
    else:
        uploaded_videos = []
        batch_path_input = ""
        online_url_input = st.text_area(
            "在线视频链接（每行一个）",
            value="",
            height=120,
            help="系统会先下载到本地临时缓存，再复用现有批处理识别流程。",
        )
        st.caption(f"在线链接模式最多 {MAX_BATCH_VIDEOS} 个视频，超出请分批执行。")
    use_gpu = st.checkbox("启用 EasyOCR GPU", value=False)
    sample_every_n_frames = st.number_input("每 N 帧识别一次", min_value=1, max_value=60, value=5, step=1)
    min_core_chars = st.number_input("最少有效字符数", min_value=2, max_value=20, value=10, step=1)
    conf_threshold = st.slider("OCR置信度阈值", min_value=0.10, max_value=0.95, value=0.90, step=0.05)
    dedupe_global = st.checkbox("输出前全局去重", value=True)
    min_stable_hits = st.number_input("稳定输出最少命中次数", min_value=1, max_value=10, value=2, step=1)
    white_vote_window = st.number_input("白字投票帧数", min_value=5, max_value=10, value=8, step=1)
    white_vote_min_hits = st.number_input("白字最少命中", min_value=2, max_value=10, value=2, step=1)
    use_dict_correction = st.checkbox("启用英文字典纠错(pyEnchant)", value=True)

st.info("当前为纯字幕提取模式：灰底与白色大字分开识别。单视频保留实时结果导出；批处理支持多视频或文件夹路径；在线链接模式会先下载到本地临时缓存，再逐视频生成表格并提供 ZIP 下载。")
if use_dict_correction and get_en_dict() is None:
    st.warning("未检测到 pyenchant/en_US 词典，字典纠错将自动跳过。")

col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 1.6])
start_clicked = col1.button("开始处理", type="primary", disabled=bool(job["active"]) and not bool(job["paused"]))
pause_clicked = col2.button("暂停", disabled=not (bool(job["active"]) and not bool(job["paused"])))
resume_clicked = col3.button("继续处理", disabled=not bool(job["paused"]))
end_clicked = col4.button("结束处理（重置）")

if end_clicked:
    reset_job()
    st.rerun()

if start_clicked:
    config = make_job_config(
        use_gpu=bool(use_gpu),
        sample_every_n_frames=int(sample_every_n_frames),
        min_core_chars=int(min_core_chars),
        conf_threshold=float(conf_threshold),
        dedupe_global=bool(dedupe_global),
        min_stable_hits=int(min_stable_hits),
        white_vote_window=int(white_vote_window),
        white_vote_min_hits=int(white_vote_min_hits),
        use_dict_correction=bool(use_dict_correction),
    )
    if processing_mode == "单视频":
        path = ensure_video_path(video_path_input)
        if not path or not os.path.exists(path):
            st.error("请先填写有效的视频路径。")
        elif not is_video_file(path):
            st.error("当前路径不是支持的视频文件。")
        else:
            try:
                init_batch_state()
                start_job(video_path=path, **config)
                st.rerun()
            except Exception as exc:
                st.error(f"启动失败: {exc}")
    elif processing_mode == "批处理":
        try:
            video_paths, notes = collect_batch_video_paths(batch_path_input, uploaded_videos)
            if not video_paths:
                st.error("批处理未找到可识别的视频。请拖入视频文件，或填写有效的视频/文件夹路径。")
            else:
                start_batch_run(video_paths=video_paths, config=config, notes=notes)
                st.rerun()
        except Exception as exc:
            st.error(f"启动失败: {exc}")
    else:
        try:
            with st.spinner("正在下载在线视频到本地临时缓存..."):
                video_paths, notes = collect_batch_video_urls(online_url_input)
            if not video_paths:
                st.error("在线链接模式未下载到可识别的视频。请检查链接后重试。")
            else:
                start_batch_run(video_paths=video_paths, config=config, notes=notes)
                st.rerun()
        except Exception as exc:
            st.error(f"启动失败: {exc}")

if pause_clicked:
    st.session_state.job["paused"] = True
    st.session_state.job["active"] = False
    st.rerun()

if resume_clicked:
    st.session_state.job["paused"] = False
    st.session_state.job["active"] = True
    st.rerun()

job = st.session_state.job
batch = st.session_state.batch
if bool(job["active"]) and not bool(job["paused"]) and not bool(job["finished"]):
    try:
        process_batch(batch_samples=35)
        job = st.session_state.job
        batch = st.session_state.batch
        total = max(1, int(job["total_frames"]))
        cur = min(int(job["frame_idx"]), total)
        if bool(batch["enabled"]):
            finalize_finished_batch_video()
            job = st.session_state.job
            batch = st.session_state.batch
            total = max(1, int(job["total_frames"]))
            cur = min(int(job["frame_idx"]), total)
            overall_total = max(1, len(batch["queue"]))
            current_video_no = min(int(batch["current_index"]) + 1, overall_total)
            overall_done = int(batch["completed_count"])
            frame_progress = cur / total if total else 0.0
            overall_progress = min((overall_done + frame_progress) / overall_total, 1.0)
            st.progress(
                overall_progress,
                text=f"批处理中：第 {current_video_no}/{overall_total} 个视频，当前帧 {cur}/{total}",
            )
            render_batch_results()
            job = st.session_state.job
            if bool(job["active"]) and not bool(job["finished"]):
                st.rerun()
            elif bool(batch["finished"]):
                st.rerun()
        else:
            st.progress(cur / total, text=f"处理中: {cur}/{total} 帧")
            render_results()
            if bool(job["active"]) and not bool(job["finished"]):
                st.rerun()
    except Exception as exc:
        st.error(f"处理失败: {exc}")
elif bool(job["paused"]):
    total = max(1, int(job["total_frames"]))
    cur = min(int(job["frame_idx"]), total)
    if bool(batch["enabled"]):
        st.warning(f"批处理已暂停：第 {int(batch['current_index']) + 1}/{len(batch['queue'])} 个视频，当前帧 {cur}/{total}")
        overall_total = max(1, len(batch["queue"]))
        overall_done = int(batch["completed_count"])
        frame_progress = cur / total if total else 0.0
        st.progress(min((overall_done + frame_progress) / overall_total, 1.0))
        render_batch_results()
    else:
        st.warning(f"已暂停：{cur}/{total} 帧")
        st.progress(cur / total)
        render_results()
elif bool(job["finished"]):
    if bool(batch["enabled"]):
        finalize_finished_batch_video()
        batch = st.session_state.batch
        started = batch["started_at"]
        elapsed = "-"
        if started:
            elapsed = str((datetime.now() - datetime.fromisoformat(str(started))).seconds)
        st.success(
            f"批处理完成，共处理 {len(batch['results'])}/{len(batch['queue'])} 个视频，耗时 {elapsed} 秒。"
        )
        st.progress(1.0)
        render_batch_results()
    else:
        started = job["started_at"]
        elapsed = "-"
        if started:
            elapsed = str((datetime.now() - datetime.fromisoformat(str(started))).seconds)
        st.success(f"处理完成，共提取 {len(job['rows'])} 条字幕，耗时 {elapsed} 秒。")
        st.progress(1.0)
        render_results()
else:
    if bool(batch["finished"]):
        started = batch["started_at"]
        elapsed = "-"
        if started:
            elapsed = str((datetime.now() - datetime.fromisoformat(str(started))).seconds)
        st.success(
            f"批处理完成，共处理 {len(batch['results'])}/{len(batch['queue'])} 个视频，耗时 {elapsed} 秒。"
        )
        st.progress(1.0)
        render_batch_results()
    else:
        st.caption("等待开始处理。")
