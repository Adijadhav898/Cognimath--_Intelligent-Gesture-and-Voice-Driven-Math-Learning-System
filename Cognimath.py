import os
import time
import base64
from io import BytesIO

import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Optional dependencies: wrap imports so the app can still load
try:
    from cvzone.HandTrackingModule import HandDetector
except Exception:
    HandDetector = None

try:
    import pymongo
except Exception:
    pymongo = None

try:
    import speech_recognition as sr
except Exception:
    sr = None

try:
    from werkzeug.security import generate_password_hash, check_password_hash
except Exception:
    generate_password_hash = None
    check_password_hash = None

try:
    import google.generativeai as genai
except Exception:
    genai = None

# -------------------- App Config -------------------- #
st.set_page_config(
    page_title="AirMath - Interactive Math Learning",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Helpers & Caching -------------------- #

@st.cache_resource(show_spinner=False)
def configure_ai():
    """Configure Gemini model using env var GENAI_API_KEY.
    Falls back to a helpful error if not available.
    """
    if genai is None:
        st.warning("google-generativeai not installed. AI features are disabled.")
        return None

    api_key = "AIzaSyBndiNy1xQEfP4JUSMU4LX7Cy6rTXWWHWQ"

    if not api_key:
        st.warning("GENAI_API_KEY not set. Set it in your environment to enable AI.")
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        st.error(f"Failed to configure AI: {e}")
        return None


@st.cache_resource(show_spinner=False)
def get_db():
    """Get Mongo database connection. If Mongo isn't available, return None safely."""
    uri = "mongodb://localhost:27017/"

    if pymongo is None:
        st.info("pymongo not installed. Auth & persistence are disabled.")
        return None, None, None
    try:
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=3000)
        # ping to ensure reachable
        client.admin.command("ping")
        db = client[os.getenv("AIRMATH_DB", "AirMath")] 
        users = db["users"]
        notes = db["notes"]
        # ensure indexes
        users.create_index("username", unique=True)
        users.create_index("email", unique=True)
        notes.create_index([("username", 1), ("created_at", -1)])
        return db, users, notes
    except Exception as e:
        st.warning(f"MongoDB not reachable: {e}. Auth & persistence disabled.")
        return None, None, None


@st.cache_resource(show_spinner=False)
def get_hand_detector():
    if HandDetector is None:
        st.info("cvzone not installed. Gesture page will be limited.")
        return None
    return HandDetector(staticMode=False, maxHands=1, detectionCon=0.7)


@st.cache_data(show_spinner=False)
def cached_ai_text(model_name: str, prompt: str) -> str:
    model = configure_ai()
    if model is None:
        return "AI model not configured. Set GENAI_API_KEY to enable responses."
    try:
        resp = model.generate_content(prompt)
        return getattr(resp, "text", "(No response text)")
    except Exception as e:
        return f"AI error: {e}"


def cached_ai_vision(prompt: str, pil_image: Image.Image) -> str:
    model = configure_ai()
    if model is None:
        return "AI model not configured. Set GENAI_API_KEY to enable responses."
    try:
        resp = model.generate_content([prompt, pil_image])
        return getattr(resp, "text", "(No response text)")
    except Exception as e:
        return f"AI error: {e}"


# -------------------- Session State -------------------- #

def init_state():
    ss = st.session_state
    ss.setdefault("page", "Welcome")
    ss.setdefault("logged_in", False)
    ss.setdefault("username", None)
    ss.setdefault("Blackboard", np.zeros((720, 1280, 3), dtype=np.uint8))
    ss.setdefault("bb_x", 50)
    ss.setdefault("bb_y", 100)
    ss.setdefault("canvas", None)
    ss.setdefault("run_gesture", False)
    ss.setdefault("last_ai_text", "")


init_state()

MODEL = configure_ai()
DB, USERS, NOTES = get_db()
DETECTOR = get_hand_detector()

# -------------------- Styles -------------------- #

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
      .main-header { font-size: 3rem; color: white; text-align: center; margin-bottom: 1.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); }
      .card { background: rgba(255,255,255,0.92); border-radius: 12px; padding: 20px; box-shadow: 0 6px 16px rgba(0,0,0,0.15); margin-bottom: 18px; }
      .feature-icon { font-size: 2.2rem; margin-bottom: 8px; }
      .stButton button { width: 100%; border-radius: 8px; background: linear-gradient(45deg,#667eea,#764ba2); color: white; font-weight: 600; border: none; padding: 10px; }
      .stButton button:hover { filter: brightness(1.05); transform: translateY(-1px); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Utility functions -------------------- #

MAX_BB_WIDTH = 1200
FONT = cv2.FONT_HERSHEY_SIMPLEX
LINE_TYPE = cv2.LINE_AA
LINE_HEIGHT = 40


def add_text_to_blackboard(text: str, font_scale: float = 1.0, thickness: int = 2, color=(255, 255, 255)):
    """Add text with word-wrapping to the blackboard image stored in session state."""
    if not text:
        return
    words = text.split()
    for word in words:
        (w, h), _ = cv2.getTextSize(word, FONT, font_scale, thickness)
        if st.session_state.bb_x + w > MAX_BB_WIDTH:
            st.session_state.bb_x = 50
            st.session_state.bb_y += LINE_HEIGHT
        cv2.putText(
            st.session_state.Blackboard,
            word,
            (st.session_state.bb_x, st.session_state.bb_y),
            FONT,
            font_scale,
            color,
            thickness,
            LINE_TYPE,
        )
        st.session_state.bb_x += w + 10


def reset_blackboard():
    st.session_state.Blackboard = np.zeros((720, 1280, 3), dtype=np.uint8)
    st.session_state.bb_x = 50
    st.session_state.bb_y = 100


def save_blackboard_png() -> bytes:
    bgr = st.session_state.Blackboard
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def process_speech() -> str:
    if sr is None:
        st.error("speech_recognition not installed.")
        return ""
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening... Speak now!")
            audio = recognizer.listen(source, timeout=5)
            query = recognizer.recognize_google(audio)
            st.success(f"You said: {query}")
            return query
    except sr.WaitTimeoutError:
        st.error("No speech detected within the timeout period.")
    except sr.UnknownValueError:
        st.error("Could not understand the audio.")
    except sr.RequestError as e:
        st.error(f"Google Speech service error: {e}")
    except Exception as e:
        st.error(f"Speech error: {e}")
    return ""


# -------------------- Pages -------------------- #

PAGES = {
    "Welcome": "🏠 Welcome",
    "Register/Login": "🔐 Register/Login",
    "Blackboard": "📝 Blackboard",
    "Gesture Recognition": "👋 Gesture Recognition",
    "My Notes": "📚 My Notes"
}



def render_sidebar():
    with st.sidebar:
        st.image("C:\\Users\\Aditya Jadhav\\Desktop\\datasets\\Cognimath\\Cognimath--_Intelligent-Gesture-and-Voice-Driven-Math-Learning-System\\Cognimathlogo.png", width=140)


        st.title("AirMath Navigation")
        if st.session_state.logged_in:
            st.success(f"Welcome, {st.session_state.username}!")
            if st.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.session_state.username = None
                st.session_state.page = "Welcome"
                st.rerun()
        st.markdown("---")
        for key, label in PAGES.items():
            if st.button(label, key=f"nav_{key}"):
                st.session_state.page = key
                st.rerun()


# ---------- Auth ---------- #

def strong_password(pw: str) -> bool:
    if len(pw) < 8:
        return False
    has_upper = any(c.isupper() for c in pw)
    has_lower = any(c.islower() for c in pw)
    has_digit = any(c.isdigit() for c in pw)
    has_sym = any(c in "!@#$%^&*()-_=+[]{};:'\",.<>/?|\\" for c in pw)
    return has_upper and has_lower and has_digit and has_sym


def render_auth_page():
    st.markdown("<h1 style='text-align:center;color:Black'>Authentication</h1>", unsafe_allow_html=True)

    mode = st.radio("", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    can_auth = USERS is not None and generate_password_hash is not None and check_password_hash is not None

    if not can_auth:
        st.warning("Auth disabled (missing MongoDB or werkzeug). You can still explore the app.")

    if mode == "Register":
        st.subheader("Create a New Account")
        username = st.text_input("Username", key="reg_user")
        email = st.text_input("Email", key="reg_email")
        password = st.text_input("Password", type="password", key="reg_pw")
        confirm = st.text_input("Confirm Password", type="password", key="reg_cpw")

        if st.button("Register", type="primary", use_container_width=True):
            if not can_auth:
                st.error("Registration unavailable in this environment.")
            elif not all([username, email, password, confirm]):
                st.error("All fields are required.")
            elif "@" not in email:
                st.error("Enter a valid email address.")
            elif password != confirm:
                st.error("Passwords do not match.")
            elif not strong_password(password):
                st.error("Weak password. Use 8+ chars with upper/lower/digit/symbol.")
            elif USERS.find_one({"username": username}):
                st.error("Username already exists.")
            elif USERS.find_one({"email": email}):
                st.error("Email already exists.")
            else:
                try:
                    USERS.insert_one({
                        "username": username,
                        "email": email,
                        "password": generate_password_hash(password),
                        "created_at": time.time(),
                    })
                    st.success("Registration successful!")
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.page = "Blackboard"
                    st.rerun()
                except Exception as e:
                    st.error(f"Registration failed: {e}")

    else:
        st.subheader("Login to Your Account")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pw")
        if st.button("Login", type="primary", use_container_width=True):
            if not can_auth:
                st.error("Login unavailable in this environment.")
            else:
                try:
                    user = USERS.find_one({"username": username})
                    if user and check_password_hash(user.get("password", ""), password):
                        st.success("Login successful!")
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.page = "Blackboard"
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                except Exception as e:
                    st.error(f"Login error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------- Welcome ---------- #

def render_welcome_page():
    st.markdown("<h1 class='main-header'>Welcome to AirMath</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='background:rgba(255,255,255,0.85);padding:16px;border-radius:12px;margin-bottom:70px'>
          <p style='font-size:1.1rem;text-align:center;color:black'>
            AirMath blends gesture recognition, voice input, and an AI tutor to make math learning fun and interactive.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='card' style='text-align:center;color:black'>
          <div class='feature-icon'>👋</div>
          <h4>Gesture Control</h4>
          <p>Draw problems with your hand.</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='card' style='text-align:center;color:black'>
          <div class='feature-icon'>🎤</div>
          <h4>Voice Input</h4>
          <p>Speak questions, get solutions.</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='card' style='text-align:center;color:black'>
          <div class='feature-icon'>🤖</div>
          <h4>AI Tutor</h4>
          <p>Step-by-step explanations.</p>
        </div>""", unsafe_allow_html=True)


# ---------- Blackboard ---------- #

def render_blackboard_page():
    if not st.session_state.logged_in:
        st.warning("Please log in to access the blackboard.")
        st.session_state.page = "Register/Login"
        st.rerun()
        return

    st.title("📝 Interactive Blackboard")
    st.write("Use speech-to-text, AI tutor, and export your notes.")

    col1, col2 = st.columns([3, 1])
    with col1:
        bb_placeholder = st.empty()
        bb_placeholder.image(st.session_state.Blackboard, channels="BGR", use_container_width=True)
    with col2:
        st.subheader("Tools")
        if st.button("🧹 Clear Blackboard", use_container_width=True):
            reset_blackboard()
            st.rerun()

        if st.button("🎤 Speak and Write", use_container_width=True):
            query = process_speech()
            if query:
                add_text_to_blackboard(query)
                st.rerun()

        st.divider()
        st.subheader("AI Assistance")
        uploaded_file = st.file_uploader("Upload an image for AI analysis", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_container_width=True)
            if st.button("🔍 Analyze Image", use_container_width=True):
                with st.spinner("AI is analyzing your image..."):
                    resp = cached_ai_vision("Analyze this image and solve/explain any math present.", img)
                    st.session_state.last_ai_text = resp
                    st.success("AI Analysis:")
                    st.write(resp)
                    add_text_to_blackboard(resp)
                    st.rerun()

        st.subheader("Virtual Tutor")
        q = st.text_input("Ask the tutor a question:", key="tutor_q")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📤 Ask Tutor", use_container_width=True):
                if q:
                    with st.spinner("Tutor is thinking..."):
                        resp = cached_ai_text("gemini-1.5-flash", q)
                        st.session_state.last_ai_text = resp
                        st.success("Tutor's Response:")
                        st.write(resp)
                        add_text_to_blackboard(resp)
                        st.rerun()
                else:
                    st.warning("Please enter a question first.")
        with c2:
            if st.button("🎤 Voice Ask", use_container_width=True):
                query = process_speech()
                if query:
                    with st.spinner("Tutor is thinking..."):
                        resp = cached_ai_text("gemini-1.5-flash", query)
                        st.session_state.last_ai_text = resp
                        st.success("Tutor's Response:")
                        st.write(resp)
                        add_text_to_blackboard(resp)
                        st.rerun()

        st.divider()
        st.subheader("Save / Export")
        png_bytes = save_blackboard_png()
        st.download_button("💾 Download Blackboard (PNG)", data=png_bytes, file_name="blackboard.png", mime="image/png", use_container_width=True)

        if NOTES is not None:
            if st.button("🗂️ Save to My Notes", use_container_width=True):
                try:
                    NOTES.insert_one({
                        "username": st.session_state.username,
                        "png": png_bytes,
                        "last_ai_text": st.session_state.last_ai_text,
                        "created_at": time.time(),
                    })
                    st.success("Saved to your notes.")
                except Exception as e:
                    st.error(f"Save failed: {e}")

            st.markdown("**Recent Notes**")
            try:
                for doc in NOTES.find({"username": st.session_state.username}).sort("created_at", -1).limit(3):
                    ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(doc.get("created_at", 0)))
                    st.caption(f"Saved at {ts}")
            except Exception:
                pass

    # refresh image view
    bb_placeholder.image(st.session_state.Blackboard, channels="BGR", use_container_width=True)

def render_notes_page():
    if not st.session_state.logged_in:
        st.warning("Please log in to access your notes.")
        st.session_state.page = "Register/Login"
        st.rerun()
        return

    st.title("📚 My Notes")
    st.write("Here you can see your recently saved blackboard notes.")

    if NOTES is None:
        st.info("Notes feature is disabled (MongoDB not connected).")
        return

    try:
        docs = NOTES.find({"username": st.session_state.username}).sort("created_at", -1).limit(5)
        for doc in docs:
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(doc.get("created_at", 0)))
            st.markdown(f"**Saved at:** {ts}")
            if "last_ai_text" in doc:
                st.markdown(f"**AI Note:** {doc['last_ai_text']}")
            if "png" in doc:
                try:
                    st.image(doc["png"], caption=f"Blackboard snapshot ({ts})", use_container_width=True)
                except Exception:
                    st.caption("[Image could not be displayed]")
            st.divider()
    except Exception as e:
        st.error(f"Could not load notes: {e}")



# ---------- Gesture Recognition ---------- #

def render_gesture_page():
    if not st.session_state.logged_in:
        st.warning("Please log in to access gesture recognition.")
        st.session_state.page = "Register/Login"
        st.rerun()
        return

    st.title("👋 Gesture Recognition")
    st.write("Draw with hand gestures, then send to AI.")

    col1, col2 = st.columns([2, 1])
    with col1:
        run = st.checkbox("Enable Camera", value=st.session_state.get("run_gesture", False))
        frame_window = st.empty()
        status_text = st.empty()
    with col2:
        st.subheader("Instructions")
        st.info("""
        - 👆 Index finger up: Draw on canvas
        - 👍 Thumb up: Clear canvas
        - ✋ All fingers up: Send canvas to AI
        """)
        st.subheader("AI Response")
        output_area = st.empty()

    if DETECTOR is None:
        st.error("Gesture features unavailable (cvzone not installed).")
        return

    # Initialize webcam
    if run and "_cap" not in st.session_state:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access the webcam. Check camera permissions.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        st.session_state._cap = cap
        st.session_state.canvas = np.zeros((480, 720, 3), dtype=np.uint8)
        st.session_state.run_gesture = True

    if not run and "_cap" in st.session_state:
        # Release when checkbox is turned off
        try:
            st.session_state._cap.release()
        except Exception:
            pass
        st.session_state.pop("_cap", None)
        st.session_state.run_gesture = False
        status_text.info("Camera stopped.")
        return

    # Process a bounded number of frames per run to avoid infinite loop
    if run and "_cap" in st.session_state:
        cap = st.session_state._cap
        prev_pos = None
        max_frames = 60  # ~3 seconds at ~20fps per script run
        processed = 0
        output_text = ""

        while processed < max_frames:
            success, img = cap.read()
            if not success:
                status_text.error("Failed to capture image.")
                break
            img = cv2.flip(img, 1)

            hands, _ = DETECTOR.findHands(img, draw=False, flipType=True)
            if hands:
                hand = hands[0]
                fingers = DETECTOR.fingersUp(hand)
                lmList = hand["lmList"]

                # Index finger up - draw
                if fingers == [0, 1, 0, 0, 0]:
                    current_pos = tuple(lmList[8][0:2])
                    if prev_pos is not None:
                        cv2.line(st.session_state.canvas, prev_pos, current_pos, (255, 0, 255), 8)
                    prev_pos = current_pos
                    status_text.info("Drawing mode")
                # Thumb up - clear canvas
                elif fingers == [1, 0, 0, 0, 0]:
                    st.session_state.canvas[:] = 0
                    prev_pos = None
                    status_text.info("Canvas cleared")
                    time.sleep(0.2)
                # All fingers up - send to AI
                elif fingers == [1, 1, 1, 1, 1]:
                    if np.any(st.session_state.canvas):
                        status_text.info("Sending to AI...")
                        pil_image = Image.fromarray(cv2.cvtColor(st.session_state.canvas, cv2.COLOR_BGR2RGB))
                        resp = cached_ai_vision("Solve this math problem:", pil_image)
                        output_text = resp
                        output_area.success(output_text)
                        time.sleep(0.6)
            else:
                prev_pos = None
                status_text.info("No hand detected")

            # Overlay canvas on webcam feed
                        # Overlay canvas on webcam feed (resize fix)
            if st.session_state.canvas.shape[:2] != img.shape[:2]:
                st.session_state.canvas = cv2.resize(
                    st.session_state.canvas,
                    (img.shape[1], img.shape[0])
                )

            img_combined = cv2.addWeighted(img, 0.7, st.session_state.canvas, 0.3, 0)
            frame_window.image(img_combined, channels="BGR")
            processed += 1
            time.sleep(0.03)


        # re-run to fetch next batch of frames if still enabled
        if run:
            st.rerun()


# -------------------- Main -------------------- #

render_sidebar()

page = st.session_state.page
if page == "Welcome":
    render_welcome_page()
elif page == "Register/Login":
    render_auth_page()
elif page == "Blackboard":
    render_blackboard_page()
elif page == "Gesture Recognition":
    render_gesture_page()
elif page == "My Notes":
    render_notes_page()

else:
    render_welcome_page()

