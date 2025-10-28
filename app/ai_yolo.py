# app/ai_yolo.py
# YOLO 세그멘테이션 & 포즈 (REST) + 레퍼런스 영상 분석 + 세그멘트 영상 미리보기
from fastapi import APIRouter, HTTPException, Body, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Union
import base64, re, os, tempfile, uuid, math
import numpy as np
import cv2
from ultralytics import YOLO

router = APIRouter(prefix="/ai", tags=["ai"])

# =========================
# 요청/응답 모델
# =========================
class ImageReq(BaseModel):
    # "data:image/jpeg;base64,..." 또는 순수 base64 문자열
    image: str

class Keypoint(BaseModel):
    id: int
    x: float
    y: float
    conf: float

class PoseRes(BaseModel):
    image: Optional[str] = None          # dataURL (jpg)
    keypoints: List[Keypoint] = Field(default_factory=list)
    bbox: Optional[List[float]] = None   # [x1, y1, x2, y2]

# =========================
# 유틸
# =========================
def _clean_b64(s: str) -> str:
    """dataURL/순수 base64 모두 허용, 공백/개행 제거, 패딩 보정"""
    if not s:
        raise HTTPException(400, "image/base64 required")
    if s.strip().lower().startswith("data:"):
        parts = s.split(",", 1)
        s = parts[1] if len(parts) > 1 else ""
    s = re.sub(r"\s+", "", s)
    missing = len(s) % 4
    if missing:
        s += "=" * (4 - missing)
    return s

def _decode_to_bgr(data: str) -> np.ndarray:
    """base64 -> OpenCV BGR 이미지"""
    try:
        b64 = _clean_b64(data)
        img_bytes = base64.b64decode(b64, validate=False)
    except Exception as e:
        raise HTTPException(400, f"Invalid image/base64: {e}")
    arr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "Unable to decode image")
    return bgr

def _encode_to_dataurl(bgr: np.ndarray, q: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise HTTPException(500, "JPEG encode failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")

def _resize_for_infer(bgr: np.ndarray, max_size: int = 640):
    """긴 변 기준 max_size로 리사이즈(속도↑). (resized, scale, origH, origW) 반환"""
    H, W = bgr.shape[:2]
    s = max_size / max(H, W)
    if s < 1.0:
        resized = cv2.resize(bgr, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
    else:
        resized = bgr
        s = 1.0
    return resized, s, H, W

# =========================
# 모델 로드(11m 없으면 v8n으로 폴백)
# =========================
try:
    SEG = YOLO("yolo11m-seg.pt")
except Exception:
    SEG = YOLO("yolov8n-seg.pt")

try:
    POSE = YOLO("yolo11m-pose.pt")
except Exception:
    POSE = YOLO("yolov8n-pose.pt")

# =========================
# 세그멘테이션 (이미지, REST)
# =========================
@router.post("/segment", summary="Segment")
def segment(payload: Union[ImageReq, str] = Body(...)):
    img_str = payload if isinstance(payload, str) else payload.image
    bgr = _decode_to_bgr(img_str)
    resized, s, H, W = _resize_for_infer(bgr)

    res = SEG(resized, verbose=False)[0]
    vis = resized.copy()

    # 마스크 시각화
    if getattr(res, "masks", None) is not None and len(res.masks) > 0:
        overlay = vis.copy()
        for seg_xy in res.masks.xy:
            poly = np.array(seg_xy, dtype=np.int32).reshape((-1, 1, 2))
            color = np.random.randint(0, 255, size=3).tolist()
            cv2.fillPoly(overlay, [poly], color)
        vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

    # 박스/신뢰도
    if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
        for b in res.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            conf = float(b.conf[0].item()) if getattr(b, "conf", None) is not None else 0.0
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(vis, f"{conf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    # 원본 크기 복원
    if s != 1.0:
        vis = cv2.resize(vis, (W, H), interpolation=cv2.INTER_LINEAR)

    return {"image": _encode_to_dataurl(vis, 82)}

# =========================
# COCO 기본 연결(필요 시 수정)
# =========================
EDGES = [
    (5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),(0,1),(1,2),(2,3),(3,4),(1,5),(1,6)
]

# =========================
# 포즈 (이미지, REST)
# =========================
@router.post("/pose", response_model=PoseRes, summary="Pose")
def pose(payload: Union[ImageReq, str] = Body(...)):
    img_str = payload if isinstance(payload, str) else payload.image
    bgr = _decode_to_bgr(img_str)
    resized, s, H, W = _resize_for_infer(bgr)

    res = POSE(resized, verbose=False)[0]
    vis = resized.copy()
    keypoints: List[Keypoint] = []

    # 키포인트 (첫 번째 객체 기준)
    try:
        if getattr(res, "keypoints", None) is not None and len(res.keypoints) > 0:
            xy = res.keypoints.xy[0].cpu().numpy()            # [K, 2]
            conf = (res.keypoints.conf[0].cpu().numpy()
                    if res.keypoints.conf is not None else np.ones((xy.shape[0],), np.float32))

            inv = 1.0 / s
            for i, (x, y) in enumerate(xy):
                c = float(conf[i])
                keypoints.append(Keypoint(id=int(i), x=float(x * inv), y=float(y * inv), conf=c))
                if c > 0.3:
                    cv2.circle(vis, (int(x), int(y)), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)

            # 스켈레톤 라인
            for a, b in EDGES:
                if a < len(xy) and b < len(xy) and conf[a] > 0.3 and conf[b] > 0.3:
                    xa, ya = xy[a]; xb, yb = xy[b]
                    cv2.line(vis, (int(xa), int(ya)), (int(xb), int(yb)), (255, 0, 0), 2, lineType=cv2.LINE_AA)
    except Exception:
        pass

    # 첫 번째 박스
    bbox: Optional[List[float]] = None
    if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
        x1, y1, x2, y2 = res.boxes.xyxy[0].tolist()
        bbox = [float(x1 / s), float(y1 / s), float(x2 / s), float(y2 / s)]

    if s != 1.0:
        vis = cv2.resize(vis, (W, H), interpolation=cv2.INTER_LINEAR)

    return PoseRes(
        image=_encode_to_dataurl(vis, 82),
        keypoints=keypoints,
        bbox=bbox
    )

# =========================
# 레퍼런스 분석용 헬퍼 (각도/정규화/클래스)
# =========================
ANGLE_RANGE = {
    "knee":  (60.0, 180.0),
    "hip":   (50.0, 180.0),
    "trunk": (0.0,   45.0),
}
REF_N = 100

EX_DB = [
  { "id":"squat",  "name":"스쿼트", "joints":["knee","hip","trunk"], "proto":{ "kneeROM":110, "hipROM":90, "trunkROM":20 } },
  { "id":"lunge",  "name":"런지",   "joints":["knee","hip","trunk"], "proto":{ "kneeROM":100, "hipROM":80, "trunkROM":25 } },
  { "id":"pushup", "name":"푸시업", "joints":["trunk"],              "proto":{ "kneeROM":10,  "hipROM":15, "trunkROM":10 } },
  { "id":"situp",  "name":"싯업",   "joints":["trunk","hip"],        "proto":{ "kneeROM":30,  "hipROM":60, "trunkROM":45 } },
  { "id":"plank",  "name":"플랭크", "joints":["trunk"],              "proto":{ "kneeROM":10,  "hipROM":10, "trunkROM":8  } },
]

def _clamp(x, a, b): return a if x < a else b if x > b else x

def _angle(a, b, c):
    v1 = np.array([a[0]-b[0], a[1]-b[1]], np.float32)
    v2 = np.array([c[0]-b[0], c[1]-b[1]], np.float32)
    n = (np.linalg.norm(v1)*np.linalg.norm(v2)) + 1e-8
    cos = float(np.dot(v1, v2) / n)
    cos = _clamp(cos, -1.0, 1.0)
    return math.degrees(math.acos(cos))

def _trunk_flex(ls, rs, lh, rh):
    sm = ((ls[0]+rs[0])/2.0, (ls[1]+rs[1])/2.0)
    hm = ((lh[0]+rh[0])/2.0, (lh[1]+rh[1])/2.0)
    v = np.array([sm[0]-hm[0], sm[1]-hm[1]], np.float32)
    up = np.array([0.0, -1.0], np.float32)
    n = np.linalg.norm(v) + 1e-8
    cos = float(np.dot(v, up) / n)
    cos = _clamp(cos, -1.0, 1.0)
    return math.degrees(math.acos(cos))

def _ema(seq, a=0.3):
    if not seq: return seq
    out = [seq[0]]
    p = np.array(seq[0], np.float32)
    for i in range(1, len(seq)):
        c = np.array(seq[i], np.float32)
        p = a*c + (1-a)*p
        out.append(p.tolist())
    return out

def _resample(seq, N):
    if not seq: return []
    out, L = [], len(seq)-1
    for i in range(N):
        t = i*(L/(N-1))
        i0, i1 = int(math.floor(t)), min(L, int(math.floor(t))+1)
        r = t - i0
        a, b = np.array(seq[i0], np.float32), np.array(seq[i1], np.float32)
        v = a + (b-a)*r
        out.append(v.tolist())
    return out

def _norm_angles(seq):
    out = []
    for k, h, t in seq:
        k0, k1 = ANGLE_RANGE["knee"]
        h0, h1 = ANGLE_RANGE["hip"]
        t0, t1 = ANGLE_RANGE["trunk"]
        nk = (_clamp(k, k0, k1)-k0)/(k1-k0)
        nh = (_clamp(h, h0, h1)-h0)/(h1-h0)
        nt = (_clamp(t, t0, t1)-t0)/(t1-t0)
        out.append([float(nk), float(nh), float(nt)])
    return out

def _rom(vals):
    f = [v for v in vals if np.isfinite(v)]
    if not f: return 0.0
    return float(max(f) - min(f))

def _classify_by_db(romK, romH, romT):
    def N(v, m): return (v-m)/max(m, 1.0)
    best, bestD = None, 1e9
    for ex in EX_DB:
        p = ex["proto"]
        d = (N(romK, p["kneeROM"])**2 +
             N(romH, p["hipROM"])**2  +
             N(romT, p["trunkROM"])**2)
        if d < bestD: bestD, best = d, ex
    conf = int(round(100*(1-min(bestD, 1.0))))
    return {**best, "conf": conf}

def _weights_for_ex(joints):
    base = {"knee": .2, "hip": .2, "trunk": .2}
    pick = [j for j in joints if j in ("knee","hip","trunk")]
    if pick:
        for j in pick: base[j] += 0.6/len(pick)
    else:
        base = {"knee": .4, "hip": .35, "trunk": .25}
    s = base["knee"]+base["hip"]+base["trunk"]
    return {k: base[k]/s for k in ("knee","hip","trunk")}

# =========================
# 레퍼런스 영상 업로드 → 포즈 시퀀스/ROM/운동추정 (REST)
# =========================
@router.post("/pose_video_ref", summary="Pose Video Ref")
async def pose_video_ref(
    file: UploadFile = File(...),
    frameskip: int = 5,
    max_frames: int = 300,
    max_size: int = 640,
):
    data = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(data); tmp.close()
    path = tmp.name

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        os.unlink(path)
        raise HTTPException(400, "Invalid video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    ok, frame = cap.read()
    if not ok:
        cap.release(); os.unlink(path)
        raise HTTPException(400, "Empty video")

    H, W = frame.shape[:2]
    s = max_size / max(H, W)
    if s < 1.0:
        out_w, out_h = int(W * s), int(H * s)
    else:
        out_w, out_h = W, H
        s = 1.0

    seq = []
    count, i = 0, 0

    def infer_angles(img):
        res = POSE(img, verbose=False)[0]
        if getattr(res, "keypoints", None) is None or len(res.keypoints) == 0:
            return None, None
        xy = res.keypoints.xy[0].cpu().numpy()
        sc = (res.keypoints.conf[0].cpu().numpy()
              if res.keypoints.conf is not None else np.ones((xy.shape[0],), np.float32))
        try:
            rhip = xy[12]; rknee = xy[14]; rank = xy[16]; rsho = xy[6]
            lsho = xy[5]; lhip = xy[11]; rhip2 = xy[12]
        except Exception:
            return None, None

        kneeA  = _angle(rhip, rknee, rank)
        hipA   = _angle(rsho, rhip, rknee)
        trunkA = _trunk_flex(lsho, rsho, lhip, rhip2)
        conf   = float(min(sc[14] if len(sc)>14 else 1.0,
                           sc[12] if len(sc)>12 else 1.0,
                           sc[16] if len(sc)>16 else 1.0))
        return [kneeA, hipA, trunkA], conf

    # 첫 프레임 포함 루프
    while True:
        if i % max(1, frameskip) == 0:
            img = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA) if s < 1.0 else frame
            ang, conf = infer_angles(img)
            if ang is not None:
                seq.append(ang)
                count += 1
            if count >= max_frames:
                break
        i += 1
        ok, frame = cap.read()
        if not ok:
            break

    cap.release()
    try: os.unlink(path)
    except: pass

    if not seq:
        raise HTTPException(400, "No pose detected in video")

    s1 = _ema(seq, .3)
    s2 = _resample(s1, REF_N)
    nA = _norm_angles(s2)

    romK = _rom([a[0] for a in s2])
    romH = _rom([a[1] for a in s2])
    romT = _rom([a[2] for a in s2])

    picked = _classify_by_db(romK, romH, romT)
    weights = _weights_for_ex(picked.get("joints", []))

    return {
        "ref_id": str(uuid.uuid4()),
        "fps": int(round(fps)),
        "rom": {
            "knee":  float(romK),
            "hip":   float(romH),
            "trunk": float(romT),
        },
        "exercise": {
            "id":     picked["id"],
            "name":   picked["name"],
            "conf":   picked["conf"],
            "joints": picked["joints"],
        },
        "weights_per_joint": {
            "knee":  float(weights["knee"]),
            "hip":   float(weights["hip"]),
            "trunk": float(weights["trunk"]),
        },
        "seq_len": REF_N,
        # "seq_norm": nA,  # 필요 시 주석 해제(용량 ↑)
    }

# =========================
# 세그멘테이션 영상 업로드 → 미리보기 MP4 dataURL (REST)
# =========================
@router.post("/segment_video_ref", summary="Segment Video Ref")
async def segment_video_ref(
    file: UploadFile = File(...),
    frameskip: int = 2,
    max_frames: int = 300,
    max_size: int = 640,
    mode: str = "overlay",       # "overlay" | "person"
):
    data = await file.read()
    in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    in_tmp.write(data); in_tmp.close()
    in_path = in_tmp.name

    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        try: os.unlink(in_path)
        except: pass
        raise HTTPException(400, "Invalid video file")

    ok, frame = cap.read()
    if not ok:
        cap.release(); os.unlink(in_path)
        raise HTTPException(400, "Empty video")

    H, W = frame.shape[:2]
    s = max_size / max(H, W)
    if s < 1.0:
        out_w, out_h = int(W * s), int(H * s)
    else:
        out_w, out_h = W, H
        s = 1.0

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out_path = out_tmp.name; out_tmp.close()
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, max(1.0, fps / max(1, frameskip)), (out_w, out_h))

    def render_seg(img_bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(img_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA) if s < 1.0 else img_bgr
        res = SEG(img, verbose=False)[0]
        vis = img.copy()

        if getattr(res, "masks", None) is not None and len(res.masks) > 0:
            if mode == "person":
                mask = np.zeros((out_h, out_w), np.uint8)
                for seg_xy in res.masks.xy:
                    poly = np.array(seg_xy, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [poly], 255)
                vis = cv2.bitwise_and(vis, vis, mask=mask)
            else:  # overlay
                overlay = vis.copy()
                for seg_xy in res.masks.xy:
                    poly = np.array(seg_xy, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [poly], (36, 92, 255))
                vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)

        if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
            for b in res.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
        return vis

    # 첫 프레임 기록
    written = 0
    vw.write(render_seg(frame)); written += 1

    i = 1
    while written < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % max(1, frameskip) != 0:
            i += 1
            continue
        vw.write(render_seg(frame))
        written += 1
        i += 1

    cap.release(); vw.release()

    with open(out_path, "rb") as f:
        mp4_bytes = f.read()
    try: os.unlink(in_path)
    except: pass
    try: os.unlink(out_path)
    except: pass

    b64 = base64.b64encode(mp4_bytes).decode("ascii")
    return {
        "video": f"data:video/mp4;base64,{b64}",
        "fps": round(fps / max(1, frameskip), 2),
        "frames": int(written),
        "w": out_w, "h": out_h,
        "mode": mode,
    }
