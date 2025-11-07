# ----------------------------------------------------------
# Streamlit App: Hybrid Product Recommendation System (SVD + KNN)
# Dataset: Women's E-Commerce Clothing Reviews
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import math
from surprise import Dataset, Reader, SVD, KNNWithMeans
from surprise.model_selection import train_test_split

# ----------------------------------------------------------
# APP CONFIGURATION
# ----------------------------------------------------------
st.set_page_config(page_title="Hybrid Product Recommender", layout="wide")

st.title("🛍️ Hybrid Product Recommendation System")
st.markdown(
    """
    This app uses a **Hybrid Machine Learning model (SVD + KNNWithMeans)**  
    trained on the *Women's E-Commerce Clothing Reviews* dataset to recommend products.  
    The hybrid model blends collaborative filtering and similarity-based predictions.
    """
)

# ----------------------------------------------------------
# STEP 1 — DATA UPLOAD / LOAD DEFAULT
# ----------------------------------------------------------
st.header("📂 Step 1 — Load Dataset")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.info("Using default dataset: Women's Clothing E-Commerce Reviews.")
    df = pd.read_csv("Womens Clothing E-Commerce Reviews.csv")

st.write("### Dataset Preview", df.head())

# ----------------------------------------------------------
# STEP 2 — DATA CLEANING
# ----------------------------------------------------------
st.header("🧹 Step 2 — Clean and Prepare Data")

df = df[['Clothing ID', 'Rating']].rename(columns={'Clothing ID': 'product_id'})
df.dropna(inplace=True)
df['user_id'] = range(1, len(df) + 1)

st.success("✅ Data cleaned successfully. Ready for modeling.")

# ----------------------------------------------------------
# STEP 3 — TRAIN HYBRID MODEL
# ----------------------------------------------------------
st.header("🤖 Step 3 — Train Hybrid Model (SVD + KNN)")

if st.button("Train Model"):
    with st.spinner("Training models... please wait ⏳"):

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[['user_id', 'product_id', 'Rating']], reader)
        trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

        # --- Train SVD ---
        svd = SVD(n_factors=100, lr_all=0.005, reg_all=0.02, n_epochs=30)
        svd.fit(trainset)

        # --- Train KNNWithMeans ---
        sim_options = {"name": "pearson_baseline", "user_based": False}
        knn = KNNWithMeans(k=30, sim_options=sim_options, verbose=False)
        knn.fit(trainset)

        # --- Hybrid prediction ---
        alpha = 0.6
        hybrid_predictions = []
        for uid, iid, true_r in testset:
            svd_pred = svd.predict(uid, iid).est
            knn_pred = knn.predict(uid, iid).est
            hybrid_est = alpha * svd_pred + (1 - alpha) * knn_pred
            hybrid_predictions.append((true_r, hybrid_est))

        mse = sum([(true_r - est) ** 2 for (true_r, est) in hybrid_predictions]) / len(hybrid_predictions)
        rmse = math.sqrt(mse)

        st.success(f"✅ Hybrid Model trained successfully! RMSE: **{rmse:.4f}**")

        # Store in session state
        st.session_state['svd'] = svd
        st.session_state['knn'] = knn
        st.session_state['alpha'] = alpha
        st.session_state['df'] = df

# ----------------------------------------------------------
# STEP 4 — RECOMMEND PRODUCTS
# ----------------------------------------------------------
if 'svd' in st.session_state:
    st.header("🎯 Step 4 — Get Product Recommendations")

    user_id = st.number_input("Enter a user ID:", min_value=1, max_value=len(st.session_state['df']), value=100)
    num_recommendations = st.slider("Number of recommendations:", 1, 10, 5)

    if st.button("Recommend Products"):
        df = st.session_state['df']
        svd = st.session_state['svd']
        knn = st.session_state['knn']
        alpha = st.session_state['alpha']

        all_products = df['product_id'].unique()
        rated_products = df[df['user_id'] == user_id]['product_id'].unique()
        products_to_predict = [pid for pid in all_products if pid not in rated_products]

        predictions = []
        for pid in products_to_predict:
            svd_pred = svd.predict(user_id, pid).est
            knn_pred = knn.predict(user_id, pid).est
            hybrid_est = alpha * svd_pred + (1 - alpha) * knn_pred
            predictions.append((pid, hybrid_est))

        top_recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:num_recommendations]

        st.subheader(f"Top {num_recommendations} Recommended Products for User {user_id}")
        rec_df = pd.DataFrame(top_recommendations, columns=["Product ID", "Predicted Rating"])
        st.dataframe(rec_df)

        st.balloons()
