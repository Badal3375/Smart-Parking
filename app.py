import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# -------------------------------
# App Header
# -------------------------------
st.set_page_config(page_title="Smart Parking Detection", layout="wide")

st.title("🅿️ Smart Parking Detection (Stable & Error-Free)")
st.write("✅ Detects **FREE 🟢 / OCCUPIED 🔴** parking slots safely.")

uploaded_file = st.file_uploader("Upload parking image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    w, h = image.size

    st.image(image, caption=f"Image Size: {w} × {h}", use_column_width=True)

    # -------------------------------
    # Sidebar Controls
    # -------------------------------
    st.sidebar.header("Detection Settings")

    empty_std = st.sidebar.slider("Empty Max Std", 5, 60, 20)
    occ_std = st.sidebar.slider("Occupied Min Std", 15, 100, 35)
    empty_br = st.sidebar.slider("Empty Min Brightness", 150, 255, 200)
    occ_br = st.sidebar.slider("Occupied Max Brightness", 80, 200, 140)

    st.sidebar.header("Parking Spaces")
    num_spaces = st.sidebar.number_input("Number of Spaces", 1, 10, 4)

    # -------------------------------
    # Safe coordinate generator
    # -------------------------------
    def safe_val(idx, maxv):
        base = 40 + idx * 60
        return min(base, maxv - 60)

    spaces = []

    for i in range(num_spaces):
        with st.sidebar.expander(f"Space {i+1}"):
            x1 = st.number_input(f"X1", 0, w - 60, safe_val(i, w), key=f"x1_{i}")
            y1 = st.number_input(f"Y1", 0, h - 60, safe_val(i, h), key=f"y1_{i}")
            x2 = st.number_input(f"X2", 60, w, safe_val(i + 1, w), key=f"x2_{i}")
            y2 = st.number_input(f"Y2", 60, h, safe_val(i + 1, h), key=f"y2_{i}")

            if x2 > x1 + 30 and y2 > y1 + 30:
                spaces.append((x1, y1, x2, y2))

    # -------------------------------
    # Analysis
    # -------------------------------
    if st.button("🚀 ANALYZE"):
        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)

        free = occ = 0
        data = []

        for i, (x1, y1, x2, y2) in enumerate(spaces):
            roi = image.crop((x1, y1, x2, y2))
            gray = roi.convert("L")
            pixels = np.array(gray)

            stdv = np.std(pixels)
            bright = np.mean(pixels)

            gx = np.gradient(pixels, axis=1)
            gy = np.gradient(pixels, axis=0)
            edge_density = np.mean(np.sqrt(gx**2 + gy**2)) / 255

            # Scores
            free_score = (
                max(0, (empty_std - stdv) / empty_std) * 0.5 +
                max(0, (bright - empty_br) / 60) * 0.3 +
                (1 - edge_density) * 0.2
            )

            occ_score = (
                max(0, (stdv - occ_std) / 60) * 0.5 +
                max(0, (occ_br - bright) / occ_br) * 0.3 +
                edge_density * 0.2
            )

            if free_score > 0.65:
                status, color = "FREE 🟢", (0, 255, 0)
                free += 1
            elif occ_score > 0.60:
                status, color = "OCCUPIED 🔴", (255, 0, 0)
                occ += 1
            else:
                status, color = "UNCLEAR ⚪", (150, 150, 150)

            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            draw.text((x1, max(0, y1 - 18)), status, fill=color)

            thumb = roi.resize((100, 60))
            st.image(thumb, caption=f"Space {i+1}")

            data.append({
                "Space": i + 1,
                "Std Dev": stdv,
                "Brightness": bright,
                "Edge Density": edge_density,
                "Free Score": free_score,
                "Occ Score": occ_score,
                "Status": status
            })

        st.image(draw_img, use_column_width=True)

        df = pd.DataFrame(data)
        st.dataframe(df.round(2))

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Spaces", len(df))
        c2.metric("FREE 🟢", free)
        c3.metric("OCCUPIED 🔴", occ)

         # ------------------------------- 
        st.subheader("📊 Parking Occupancy Overview")
        
        chart_data = pd.DataFrame({
            "Status": ["FREE", "OCCUPIED", "UNCLEAR"],
            "Count": [
                free,
                occ,
                len(df) - free - occ
            ]
        })
        
        st.bar_chart(chart_data.set_index("Status"))