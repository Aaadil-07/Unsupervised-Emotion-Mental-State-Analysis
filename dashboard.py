import os
import time
import cv2
import base64
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import threading

from data_loader       import load_single_frame, get_face_bbox
from feature_extractor import encode_single, ShallowAutoencoder
from clustering        import (
    load_models, interpret_mental_state,
    EMOTION_COLORS, MENTAL_MAP, PLOTS_DIR
)
from utils import EmotionTracker

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# GLOBAL STATE
class AppState:
    running = False
    cap = None
    loaded = False
    km = None
    density = {}
    metrics = {}
    cluster_map = {}
    outlier_thresh = {}
    ae = None
    emotion = "—"
    state = "—"
    cluster = -1
    alert = None
    tracker = EmotionTracker(window_size=10)
    fc = 0
    t0 = time.time()
    latest_frame_bgr = None
    latest_frame_jpeg = None
    fps = 0
    plot_b64 = None
    baseline_geo = None
    
S = AppState()

def init_models():
    if not S.loaded:
        result = load_models()
        if result:
            km, db, density, metrics, cluster_map, outlier_thresh = result
            ae_path = os.path.join("models", "autoencoder.pkl")
            if os.path.exists(ae_path):
                ae = ShallowAutoencoder.load(ae_path)
                S.km = km
                S.density = density
                S.metrics = metrics
                S.cluster_map = cluster_map
                S.outlier_thresh = outlier_thresh
                S.ae = ae
                S.loaded = True

def _cluster_fig_b64():
    if not S.km or not S.cluster_map:
        return None
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#12122a")

    centres = S.km.cluster_centers_
    for cid, emo in S.cluster_map.items():
        col = EMOTION_COLORS.get(emo, "#aaa")
        ax.scatter(float(centres[cid, 0]), float(centres[cid, 1]),
                   s=180, color=col, marker="X",
                   edgecolors="white", linewidth=1.2,
                   zorder=9, label=emo)

    ax.set_title("Cluster Map", color="white", fontsize=9)
    ax.tick_params(colors="#555", labelsize=7)
    for s in ax.spines.values():
        s.set_edgecolor("#333")
    fig.tight_layout(pad=0.2)
    
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#12122a", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def predict_frame(frame):
    flat = load_single_frame(frame)
    if flat is None or S.ae is None:
        return None
    try:
        from feature_extractor import extract_geometric_features
        geo = extract_geometric_features(flat)
        
        if S.baseline_geo is None:
            S.baseline_geo = geo.copy()
        else:
            S.baseline_geo = 0.99 * S.baseline_geo + 0.01 * geo
            
        rel_geo = geo - S.baseline_geo
        
        # Extracted relative movements
        movement_energy = float(np.mean(np.abs(rel_geo)))
        smile_energy    = float(rel_geo[54] + rel_geo[55] + rel_geo[11] * 1.5 + rel_geo[13] * 1.5)
        brow_increase   = float(rel_geo[25])
        mouth_open_inc  = float(rel_geo[13])
        
        latent_vec = S.ae.encode(geo.reshape(1, -1))[0].reshape(1, -1)
        cluster    = int(S.km.predict(latent_vec)[0])
        dist       = float(np.linalg.norm(latent_vec[0] - S.km.cluster_centers_[cluster]))
        thresh     = S.outlier_thresh.get(cluster, 5.0)
        is_outlier = dist > (thresh * 4.0)  # Effectively disable the outlier bias

        info = interpret_mental_state(
            cluster_id      = cluster,
            is_outlier      = is_outlier,
            recent_history  = S.tracker.recent_history(),
            cluster_density = S.density,
            cluster_map     = S.cluster_map,
        )
        
        # ----------------------------------------------------
        # DEFINITIVE BIAS FIX (Action Unit Energy Overrides)
        # ----------------------------------------------------
        orig_emo = info["emotion"]
        
        # 1. Happy (Lip corners up + mouth open/bright)
        if smile_energy > 0.035 or geo[16] >= 1:
            info["emotion"] = "Happy"
            
        # 2. Surprised (Mouth sharply opens, but not a smile)
        elif mouth_open_inc > 0.08:
            info["emotion"] = "Surprised"
            
        # 3. Stressed / Angry (Brows tighten noticeably)
        elif brow_increase > 0.01:
            info["emotion"] = "Stressed"
            
        # 4. Calm / Neutral (Face is resting, minimal deviation from baseline)
        elif movement_energy < 0.012:
            info["emotion"] = "Calm"
        elif movement_energy < 0.02:
            info["emotion"] = "Neutral"

        if info["emotion"] != orig_emo:
            info["mental_state"] = MENTAL_MAP.get(info["emotion"], "Balanced")
            if info["emotion"] in ["Happy", "Calm", "Neutral"]:
                info["alert"] = None
                
        info["cluster"] = cluster
        return info
    except Exception:
        return None

def capture_loop():
    skip = 3
    last_t = time.time()
    frames = 0
    while S.running and S.cap and S.cap.isOpened():
        ret, frame = S.cap.read()
        if not ret:
            break
            
        S.fc += 1
        info = None
        if S.fc % skip == 0:
            info = predict_frame(frame)
            
        if info:
            emotion = info["emotion"]
            state   = info["mental_state"]
            cluster = info["cluster"]
            alert   = info["alert"]

            S.emotion = emotion
            S.state   = state
            S.cluster = cluster
            S.alert   = alert

            S.tracker.update(cluster, emotion, state, alert)
            smoothed = S.tracker.smoothed_emotion()
            
            # draw
            bbox = get_face_bbox(frame)
            emo_col_bgr = {
                "Happy":     (0,   215, 255),
                "Calm":      (80,  200, 80),
                "Neutral":   (180, 180, 180),
                "Sad":       (200, 80,  80),
                "Stressed":  (50,  150, 255),
                "Angry":     (50,  50,  244),
                "Surprised": (200, 64,  224),
            }.get(emotion, (0, 230, 180))

            if bbox:
                x, y, bw, bh = bbox
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), emo_col_bgr, 2, cv2.LINE_AA)

            S.emotion = smoothed

        # calculate FPS
        frames += 1
        now = time.time()
        if now - last_t >= 1.0:
            S.fps = frames
            frames = 0
            last_t = now

        # HUD overlay
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (frame.shape[1], 82), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, f"Emotion: {S.emotion}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,230,180), 2, cv2.LINE_AA)
        
        _, jpeg = cv2.imencode('.jpg', frame)
        S.latest_frame_jpeg = jpeg.tobytes()
    
    S.running = False
    if S.cap:
        S.cap.release()
        S.cap = None

@app.on_event("startup")
def startup():
    init_models()
    S.plot_b64 = _cluster_fig_b64()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/start")
def start_cam():
    return {"status": "ok"}

@app.post("/stop")
def stop_cam():
    S.running = False
    return {"status": "ok"}

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    S.running = True
    S.tracker = EmotionTracker(window_size=10)
    frames = 0
    last_t = time.time()
    skip = 5  # Increased skip to save CPU
    last_bbox = None
    try:
        while True:
            data = await websocket.receive_text()
            S.fc += 1
            
            if "," in data:
                b64_data = data.split(",")[1]
            else:
                b64_data = data
                
            np_arr = np.frombuffer(base64.b64decode(b64_data), np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue

            info = None
            if S.fc % skip == 0:
                info = predict_frame(frame)
                last_bbox = get_face_bbox(frame)  # Only run heavy face detection here
                
            if info:
                S.emotion = info["emotion"]
                S.state   = info["mental_state"]
                S.cluster = info["cluster"]
                S.alert   = info["alert"]

                S.tracker.update(S.cluster, S.emotion, S.state, S.alert)
                S.emotion = S.tracker.smoothed_emotion()

            bbox = last_bbox
            emo_col_bgr = {
                "Happy":     (0,   215, 255),
                "Calm":      (80,  200, 80),
                "Neutral":   (180, 180, 180),
                "Sad":       (200, 80,  80),
                "Stressed":  (50,  150, 255),
                "Angry":     (50,  50,  244),
                "Surprised": (200, 64,  224),
            }.get(S.emotion, (0, 230, 180))

            if bbox:
                x, y, bw, bh = bbox
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), emo_col_bgr, 2, cv2.LINE_AA)

            ov = frame.copy()
            cv2.rectangle(ov, (0, 0), (frame.shape[1], 82), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, f"Emotion: {S.emotion}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,230,180), 2, cv2.LINE_AA)
            
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64_out = base64.b64encode(buffer).decode('utf-8')

            frames += 1
            now = time.time()
            if now - last_t >= 1.0:
                S.fps = frames
                frames = 0
                last_t = now

            await websocket.send_text("data:image/jpeg;base64," + b64_out)
    except WebSocketDisconnect:
        S.running = False

def _to_py(obj):
    if isinstance(obj, np.generic): return obj.item()
    if isinstance(obj, dict): return {k: _to_py(v) for k, v in obj.items()}
    return obj

@app.get("/state")
async def get_state():
    color = EMOTION_COLORS.get(S.emotion, "#00e5ff") if S.emotion != "—" else "var(--cyan)"
    return JSONResponse({
        "emotion": S.emotion,
        "state": S.state,
        "cluster": _to_py(S.cluster),
        "fps": S.fps,
        "alert": S.alert,
        "color": color,
        "metrics": _to_py(S.metrics),
        "plot_b64": S.plot_b64
    })

def gen_frames():
    while True:
        if S.latest_frame_jpeg:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + S.latest_frame_jpeg + b'\r\n')
            time.sleep(0.03)  # limit stream to ~30 FPS to save bandwidth
        else:
            time.sleep(0.1)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting API on http://0.0.0.0:{port} (Access locally at http://127.0.0.1:{port})")
    uvicorn.run("dashboard:app", host="0.0.0.0", port=port, reload=False)
