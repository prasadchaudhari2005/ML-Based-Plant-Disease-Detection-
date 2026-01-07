import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# ------------------------------
# Pillow Compatibility Fix
# ------------------------------
if hasattr(Image, 'Resampling'):
    RESAMPLE = Image.Resampling.LANCZOS
else:
    RESAMPLE = Image.ANTIALIAS

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="Plant Disease Recognition",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Class Names
# ------------------------------
CLASS_NAMES = [
    'Apple Scab', 'Apple Black Rot', 'Apple Cedar Rust',
    'Healthy Apple', 'Potato Early Blight', 'Potato Late Blight', 'Healthy Potato'
]

# ------------------------------
# Farmer + Learner Friendly Disease Info
# ------------------------------
# Diseases of Field Crops book , ICAR-Indian Agricultural Research Institute (IARI)

DISEASE_INFO = {
    'Apple Scab': {
        "description": (
            "Apple scab is one of the most common and destructive fungal diseases of apple trees, "
            "caused by the fungus *Venturia inaequalis*. It mainly affects the leaves, fruits, and sometimes young twigs. "
            "During spring, the fungus releases spores from infected fallen leaves left on the ground. "
            "These spores are carried by wind and rain to young leaves and fruits, starting new infections. "
            "Early infections appear as small olive-green spots on leaves which later turn dark brown and velvety. "
            "As the disease progresses, the spots enlarge, merge, and cause leaf curling, leading to early leaf drop. "
            "On fruits, the scab lesions form dark, cracked patches that make them unattractive and sometimes unmarketable. "
            "Repeated infections can weaken the tree, reduce fruit size, and severely impact the yield in future seasons. "
            "The disease thrives best in **cool (15–25°C)** and **wet** conditions, especially during rainy springs."
        ),
        "symptoms": [
            "Dark brown or black patches on leaves and fruits.",
            "Rough, cracked, or scabby fruit skin.",
            "Premature leaf fall in severe cases."
        ],
        "causes": [
            "Fungus: *Venturia inaequalis*.",
            "Spreads from infected leaves on the ground during rain or wind.",
            "Favored by cool, wet weather and poor orchard hygiene."
        ],
        "solutions": [
            "👉 Collect and destroy all fallen leaves and infected fruits immediately after harvest.",
            "👉 During winter pruning, remove weak and crowded branches to increase airflow and sunlight.",
            "👉 Before bud break, spray fungicides like **Mancozeb** or **Captan**. Repeat every 10–14 days during wet weather.",
            "👉 Grow resistant apple varieties such as Liberty or Enterprise for long-term protection."
        ],
        "prevention": [
            "Keep orchard floors clean and dry.",
            "Avoid overhead watering.",
            "Apply preventive sprays before rainy seasons begin."
        ]
    },

    'Apple Black Rot': {
        "description": (
            "Black rot is a serious fungal disease of apple trees caused by *Botryosphaeria obtusa*. "
            "It infects not only the fruits but also the leaves, twigs, and branches. "
            "The disease begins with small purple spots on leaves that develop into ‘frog-eye’ patterns—brown centers surrounded by lighter rings. "
            "On fruits, the disease causes circular, dark, sunken lesions that can grow over time, making fruits rot on the tree or in storage. "
            "The fungus also causes cankers on branches, which appear as sunken, blackened, or cracked areas of bark. "
            "These cankers serve as a long-term source of infection, releasing spores during warm, humid weather. "
            "The fungus overwinters in **dead wood, old cankers, or mummified fruit**, and spreads through wind and rain. "
            "Black rot can cause major yield loss and tree decline if not properly managed."
        ),
        "symptoms": [
            "Round black spots on fruit (‘frog-eye’ look).",
            "Dark, sunken cankers on branches and twigs.",
            "Leaves with small purple-brown spots."
        ],
        "causes": [
            "Fungus: *Botryosphaeria obtusa*.",
            "Infects through wounds or old pruning cuts.",
            "Thrives in warm (25–30°C), humid weather."
        ],
        "solutions": [
            "👉 During winter, **cut out all dead and infected wood** at least 8–10 inches below the canker and burn it.",
            "👉 Collect and destroy any old or mummified fruits on the tree and ground.",
            "👉 Disinfect pruning tools using spirit or bleach before reusing.",
            "👉 Apply fungicides such as **Captan or Thiophanate-methyl** during the growing season."
        ],
        "prevention": [
            "Regular pruning to remove dead wood.",
            "Avoid mechanical injuries to bark.",
            "Maintain orchard cleanliness year-round."
        ]
    },

    'Apple Cedar Rust': {
        "description": (
            "Cedar-apple rust is a fungal disease caused by *Gymnosporangium juniperi-virginianae* that affects apple and crabapple trees. "
            "The unique thing about this disease is that it needs **two hosts** to complete its life cycle: an apple (or crabapple) and a juniper (or cedar) tree. "
            "In early spring, orange jelly-like horns appear on cedar trees (called galls). These horns release spores that travel through the air and infect apple leaves. "
            "After a few weeks, yellow-orange spots appear on apple leaves and fruits. "
            "The disease rarely kills the tree, but it reduces photosynthesis and weakens plant health. "
            "Infected fruits become misshapen and may fall prematurely. "
            "Without proper management, the disease continues to cycle between apple and cedar trees year after year."
        ),
        "symptoms": [
            "Orange or yellow spots on apple leaves.",
            "Deformed fruits with raised yellow spots.",
            "Orange galls with jelly-like horns on nearby cedar trees."
        ],
        "causes": [
            "Fungus: *Gymnosporangium juniperi-virginianae*.",
            "Spreads between apple and cedar trees during wet weather.",
            "Spores travel up to 2–3 km through wind and rain."
        ],
        "solutions": [
            "👉 Remove or prune nearby cedar/juniper trees within 1 km of the orchard.",
            "👉 Use rust-resistant apple varieties such as Redfree and Priscilla.",
            "👉 Apply fungicides like **Myclobutanil or Mancozeb** from pink-bud to petal-fall stage."
        ],
        "prevention": [
            "Avoid planting apple and cedar trees close to each other.",
            "Remove cedar galls before spring rains.",
            "Regularly inspect trees for rust symptoms."
        ]
    },

    'Potato Early Blight': {
        "description": (
            "Potato early blight, caused by the fungus *Alternaria solani*, is one of the most common diseases of potato plants. "
            "It typically starts on older, lower leaves as small brown spots with concentric rings (‘target-like’ pattern). "
            "As the disease spreads, leaves turn yellow and drop, reducing photosynthesis and overall plant vigor. "
            "The fungus survives on plant debris left in the soil and spreads by wind and rain splashes. "
            "Early blight is favored by **warm (25–30°C)** and **humid** conditions and often occurs when plants are stressed — such as by drought or lack of nutrients. "
            "If left uncontrolled, it can cause up to 40% yield loss, particularly in older or weak crops."
        ),
        "symptoms": [
            "Dark circular spots with ring patterns on lower leaves.",
            "Yellowing and early leaf drop.",
            "Brown lesions on stems or tubers."
        ],
        "causes": [
            "Fungus: *Alternaria solani*.",
            "Thrives in hot, humid environments.",
            "Overwinters in soil and plant remains."
        ],
        "solutions": [
            "👉 After harvest, **collect and burn infected plant residues** instead of leaving them in the soil.",
            "👉 **Rotate crops for at least 2 years** with non-host plants (like maize or beans).",
            "👉 Use **drip irrigation** and water plants early in the morning to keep leaves dry.",
            "👉 Spray protective fungicides like **Mancozeb or Chlorothalonil** when the first symptoms appear.",
            "👉 Maintain balanced nutrition with compost or potassium fertilizer to strengthen resistance."
        ],
        "prevention": [
            "Use healthy, certified seed potatoes.",
            "Avoid planting in the same field year after year.",
            "Regularly inspect and remove infected leaves."
        ]
    },

    'Potato Late Blight': {
        "description": (
            "Potato late blight, caused by *Phytophthora infestans*, is a devastating disease known for causing the historic Irish Potato Famine. "
            "It spreads very rapidly under cool and wet weather conditions. "
            "The disease affects all parts of the potato plant — leaves, stems, and tubers. "
            "On leaves, it appears as dark, water-soaked patches that expand quickly and turn black. "
            "In humid weather, white mold appears under the leaves — a clear sign of active infection. "
            "The pathogen survives in infected tubers left in the soil or discarded piles and spreads via wind, rain, and irrigation water. "
            "Once it infects a field, it can destroy an entire crop within a week if left unmanaged. "
            "Late blight thrives best in **cool (15–21°C)**, **humid**, and **cloudy** conditions."
        ),
        "symptoms": [
            "Dark, wet-looking spots on leaves and stems.",
            "White fungal growth under leaves during humidity.",
            "Brown, rotted potatoes with a foul smell."
        ],
        "causes": [
            "Water mold: *Phytophthora infestans*.",
            "Spreads via air, soil, and infected tubers.",
            "Cool and humid weather favors rapid growth."
        ],
        "solutions": [
            "👉 Plant only **certified, healthy seed tubers**.",
            "👉 **Remove infected plants immediately** to stop the spread.",
            "👉 Provide enough space between rows for airflow and sunlight.",
            "👉 Spray **Metalaxyl, Cymoxanil, or Copper-based fungicides** before and after rainy spells.",
            "👉 Destroy leftover tubers and clean fields after harvest."
        ],
        "prevention": [
            "Rotate potato crops with cereals or legumes every year.",
            "Harvest early when vines start dying.",
            "Store harvested tubers in a dry, ventilated place."
        ]
    }
}

# ------------------------------
# Load TensorFlow Model
# ------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("trained_model_new.h5")
    # Warmup the model to avoid slow first prediction
    dummy_input = np.zeros((1, 128, 128, 3))
    model.predict(dummy_input)
    return model

def predict(image):
    model = load_model()
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions), np.max(predictions)

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# ------------------------------
# Home Page
# ------------------------------
if app_mode == "Home":
    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")
    st.image("image.jpg", width=700)
    st.markdown("""
    Welcome to the **Plant Disease Recognition System**! 🌾  

    Upload a photo of a leaf to identify the disease and learn about its causes, symptoms, and treatments.  
    This system is helpful for **farmers**, **students**, and **researchers** to improve both practical knowledge and crop health.

    ### 🧩 How It Works:
    1. Go to the *Disease Recognition* page.  
    2. Upload a clear image of the affected leaf.  
    3. Get accurate disease prediction and detailed learning information.
    """)

# ------------------------------
# About Page
# ------------------------------
elif app_mode == "About":
    st.header("📘 About the Project")
    st.markdown("""
    This project combines **AI and Agriculture** to help identify plant diseases using Deep Learning.  
    It’s trained on over **26,000+ images** of healthy and diseased apple and potato leaves.  

    ### 🌱 Key Objectives:
    - Early detection of crop diseases  
    - Educating farmers on prevention and control  
    - Supporting students in agricultural learning  
    """)

# ------------------------------
# Disease Recognition Page
# ------------------------------
elif app_mode == "Disease Recognition":
    st.header("🌱 Disease Recognition")
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.session_state['uploaded_image'] = uploaded_file

    if 'uploaded_image' in st.session_state:
        image_to_display = st.session_state['uploaded_image']
        col1, col2 = st.columns([1, 3])

        with col1:
            st.image(image_to_display, caption="Uploaded Leaf", use_container_width=True)

        with col2:
            st.write("### 🌿 Step 2: Check Plant Health")
            st.write("Click below to analyze and learn about your plant’s condition:")

            if st.button("🔍 Predict Disease"):
                try:
                    img = Image.open(image_to_display)
                    img = ImageOps.fit(img, (128, 128), RESAMPLE)

                    with st.spinner('Analyzing the leaf... Please wait...'):
                        idx, confidence = predict(img)
                        prediction = CLASS_NAMES[idx]

                    st.success(f"✅ Prediction Result: **{prediction}**")
                    st.info(f"📊 Confidence: **{confidence * 100:.2f}%**")

                    if "Healthy" in prediction:
                    
                        st.info("""
                        🟢 Your plant looks **healthy!**
                        - Keep regular watering and sunlight.  
                        - Maintain field hygiene to avoid infections.  
                        """)
                    else:
                        info = DISEASE_INFO.get(prediction)
                        if info:
                            st.markdown("### 🧠 About the Disease")
                            st.write(info['description'])

                            st.markdown("### 👀 Symptoms (What to Notice)")
                            for s in info['symptoms']:
                                st.markdown(f"- {s}")

                            st.markdown("### ❓ Causes (Why it Happens)")
                            for c in info['causes']:
                                st.markdown(f"- {c}")

                            st.markdown("### 💪 Step-by-Step Solution (What to Do Now)")
                            for s in info['solutions']:
                                st.markdown(f"- {s}")

                            st.markdown("### 🌾 Future Prevention Tips")
                            for p in info['prevention']:
                                st.markdown(f"- {p}")

                except Exception as e:
                    st.error(f"⚠️ Error during prediction: {e}")
    else:
        st.info("📸 Please upload a clear leaf image to begin the analysis.")
