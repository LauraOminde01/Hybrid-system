#  Hybrid Product Recommendation System (SVD + KNN)

An **AI-powered hybrid recommender system** that suggests fashion products to users based on their preferences and ratings.  
It blends **Collaborative Filtering (SVD)** and **Similarity-Based Filtering (KNNWithMeans)** into an **interactive Streamlit web app**.

---

##  Business Context

In the fast-paced world of **e-commerce**, personalization drives engagement and sales.  
Fashion retailers, in particular, face challenges due to **large product catalogs** and **diverse customer preferences**.  

The company behind this dataset noticed:
- Customers were leaving after browsing only a few items  
- The “You may also like” section had low engagement (<3% click rate)  
- 70% of users never interacted with recommendations  

---

##  Business Problem

Without a smart recommendation system, the business struggled with:
- Low product discovery and engagement  
- Generic recommendations based on popularity  
- Missed opportunities for **cross-selling and upselling**

---

##  Business Objective

Build a **machine learning system** that:
1. Learns from past user–product interactions  
2. Predicts which products a user is most likely to rate highly  
3. Provides personalized recommendations in real time  
4. Improves engagement and customer lifetime value (CLV)

---

##  Solution Approach

### 1. Collaborative Filtering (SVD)
- Learns hidden user and product patterns (latent features)
- Predicts unseen ratings using matrix factorization

### 2. Similarity-Based Filtering (KNNWithMeans)
- Finds similar items using Pearson baseline similarity
- Recommends products that resemble items liked by the user

###  Hybrid Approach
Combines both models for stronger predictions:
\[
\text{Hybrid Score} = 0.6 \times \text{SVD Prediction} + 0.4 \times \text{KNN Prediction}
\]

This balances **latent preference learning** and **item similarity**, improving overall accuracy.

---


