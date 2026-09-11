import streamlit as st
import requests

st.title("Movie Review Sentiment Analyzer")

with st.form("review_form"):

    review = st.text_area(
        "Enter your review",
        placeholder="Write your movie review here..."
    )

    submitted = st.form_submit_button("Predict Sentiment")

if submitted:

    if not review.strip():
        st.warning("Please enter a review.")
    else:

        response = requests.post(
            "http://backend:8000/predict",
            json={"review": review}
        )

        if response.status_code == 200:

            result = response.json()

            st.success("Prediction completed!")

            st.write("### Result")
            st.write("Prediction:", result["Prediction"])
            st.write("Confidence:", result["Confidence"])

        else:
            st.error(
                f"API request failed: {response.status_code}"
            )