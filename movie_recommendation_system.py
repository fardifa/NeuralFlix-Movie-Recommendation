import streamlit as st
import pandas as pd
import pickle  # Use to load the pre-trained model or data

# Import your backend functions
from backend import load_data, preprocess_data, get_recommendations, train_model

# Set up the page configuration
st.set_page_config(page_title="Movie Recommendations", layout="wide")

# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

# Load and preprocess the data once
@st.cache_resource
def load_and_prepare():
    ratings, movies, keywords = load_data()
    ratings, movies, keywords = preprocess_data(ratings, movies, keywords)
    svd_model, _, _ = train_model(ratings)  # Train the model
    return ratings, movies, svd_model

ratings, movies, svd_model = load_and_prepare()

# Common title function for all pages
def render_title():
    st.markdown(
        """
        <style>
        .title {
            color: red;
            font-size: 50px;
            font-weight: bold;
            text-align: left;
            margin-top: 0;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="title">🎬 NEURALFLIX</div>', unsafe_allow_html=True)

def home_page():
    # Render title
    render_title()

    # Apply custom styles using CSS for description and button
    st.markdown(
        """
        <style>
        .description {
            font-size: 20px;
            color: white;
            text-align: center;
            margin-top: 20px;  
            margin-bottom: 50px;  
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Layout for the description
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="description">Discover movies tailored to your preferences. Click the button below to get started!</div>', unsafe_allow_html=True)

    # Button centered using columns
    col4, col5, col6 = st.columns([1, 1, 1])
    with col5:
        if st.button("Get Recommendations", key="home_recommend_button"):
            st.session_state.page = "user_id"

def user_id_page():
    # Render title
    render_title()

    # Adjust styles to ensure all elements align properly
    st.markdown(
        """
        <style>
        .center-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start; /* Bring content closer to the top */
            height: 20vh; /* Reduce the height for better vertical alignment */
            text-align: center;
            gap: 20px; /* Add spacing between elements */
            padding-top: 10px; /* Reduced padding to move elements further up */
        }
        .input-box {
            width: 300px; /* Set a consistent width for the input box */
        }
        .button-group {
            display: flex;
            justify-content: center;
            gap: 15px; /* Add spacing between buttons */
            margin-top: 10px; /* Reduced margin to bring buttons closer */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Centered layout for all elements
    st.markdown(
        """
        <div class="center-content">
            <h2>🔑 Enter Your User ID</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input box and buttons using Streamlit
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Input box for User ID
        user_id = st.number_input(
            "Enter your User ID:", 
            min_value=1, 
            step=1, 
            key="user_id", 
            help="Enter a valid user ID."
        )

        # Buttons positioned centrally
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Submit", key="submit_user_id"):
                st.session_state.page = "recommendations"
        with col_b:
            if st.button("Back to Home", key="back_to_home"):
                st.session_state.page = "home"

def recommendations_page():
    # Render title
    render_title()

    # Smaller title for recommendations page
    st.markdown(
        """
        <h3 style='text-align: center;'>🎥 Your Personalized Recommendations</h3>
        """,
        unsafe_allow_html=True,
    )

    # Display recommendations only if user_id exists
    if "user_id" in st.session_state and st.session_state.user_id:
        user_id = st.session_state.user_id

        # Get recommendations
        recommendations = get_recommendations(user_id, svd_model, ratings, movies, n_recommendations=4)

        if not recommendations.empty:
            st.markdown(f"Recommendations for User ID: **{user_id}**")

            # Display recommendations dynamically
            for index, row in recommendations.iterrows():
                st.markdown(f"- {row['title']}")
        else:
            st.warning("No recommendations available for this user.")
    else:
        st.warning("Redirecting to Home...")

    # Back to Home button
    if st.button("Back to Home", key="back_to_home_from_recommendations"):
        # Reset session state and navigate to home page
        st.session_state.page = "home"
        if "user_id" in st.session_state:
            del st.session_state["user_id"]

# Navigation Logic
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "user_id":
    user_id_page()
elif st.session_state.page == "recommendations":
    recommendations_page()
