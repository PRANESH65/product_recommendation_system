# 🛍️ Image-Based Product Recommendation System

A two-stage intelligent system that identifies a visually similar product from an image and recommends related products based on product metadata. This solution is designed to enhance user experience in e-commerce platforms by automating **product recognition** and **personalized product recommendations**.

---

## 🎯 Objective

> **To automatically identify and recommend similar and related products based on an input image.**

Given a query image (like a product photo), the system performs:
1. **Visual similarity search** using a pre-trained CNN (ResNet50).
2. **Contextual recommendations** using product descriptions and specifications via NLP techniques (TF-IDF + cosine similarity).

---

## 🧠 Key Features

### 🔍 Image Recognition Module
- Utilizes a **pre-trained ResNet50** model from Keras for feature extraction.
- Denoises and resizes input images using **OpenCV**.
- Identifies the most visually similar product from a product image database using **cosine similarity**.

### 🎯 Recommendation Engine
- Uses **TF-IDF vectorization** on product metadata: title, specifications, and details.
- Recommends contextually related products by computing **textual similarity**.
- Displays product images directly from URLs using **PIL** and **IPython** (if run in a Jupyter notebook).

---

## 🏗️ System Architecture

```plaintext
[User Query Image]
        |
        v
[Image Preprocessing + Denoising]
        |
        v
[ResNet50 Feature Extraction]
        |
        v
[Cosine Similarity Matching]
        |
        v
[Matched Product ID]
        |
        v
[TF-IDF on Product Metadata]
        |
        v
[Related Product Recommendations]
        |
        v
[Image Display + Product Info]

