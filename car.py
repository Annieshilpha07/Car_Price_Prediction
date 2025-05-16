import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import matplotlib.pyplot as plt

# Set up page configuration
def main():
    st.set_page_config(
        page_title='Car Price Prediction',
        page_icon='🚗',
        initial_sidebar_state='expanded',
        layout='wide',
        menu_items={"about": 'This is a Car Price Prediction app developed using machine learning models.'}
    )

    # Display the page title at the top of your app
    st.title(':rainbow[🚗 Car Price Prediction]')

    # Set up the sidebar with option menu
    selected = option_menu("Car Price Prediction App",
                            options=["Home", "Get Prediction", "Explore Data"],
                            icons=["house", "lightbulb", "bar-chart-line"],
                            default_index=1, menu_icon="car-front",
                            orientation="horizontal")

    # Home Page Section
    if selected == "Home":
        st.markdown('<h2 style="color:gray;text-align:center;">Welcome to the Car Price Prediction App!</h2>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 2], gap="large")

        with col1:
            st.markdown("<h3 style='color:#e67300;'>Skills and Techniques</h3>", unsafe_allow_html=True)
            st.markdown("""
                <ul style='font-size:17px;'>
                    <li>🔧 Data Preprocessing & Cleaning</li>
                    <li>📊 Feature Engineering & Model Training</li>
                    <li>🤖 Price Prediction using ML (Linear Regression, Random Forest)</li>
                    <li>📈 Exploratory Data Analysis (EDA)</li>
                    <li>🌐 Streamlit App Deployment</li>
                </ul>
            """, unsafe_allow_html=True)

            st.markdown("<h3 style='color:#e67300;'>Domain</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:17px;'>🚗 Automobile pricing analysis using features like brand, year, mileage, fuel type, etc.</p>", unsafe_allow_html=True)

            st.markdown("<h3 style='color:#e67300;'>Problem Statement</h3>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:17px;'>To predict the resale price of a car based on its features using machine learning models.</p>", unsafe_allow_html=True)

            st.markdown("<h3 style='color:#e67300;'>Solution Overview</h3>", unsafe_allow_html=True)
            st.markdown("""
                <ul style='font-size:17px;'>
                    <li>📥 <b>Data Collection</b>: Dataset containing car attributes and prices</li>
                    <li>🧹 <b>Data Preprocessing</b>: Cleaned data, handled missing values, encoded categories</li>
                    <li>📊 <b>Modeling</b>: Trained ML models (e.g., Random Forest, Decision Tree)</li>
                    <li>💻 <b>Deployment</b>: Streamlit app for interactive price prediction</li>
                </ul>
            """, unsafe_allow_html=True)

        with col2:
            st.write(" ")
            st.image("https://cdn.dribbble.com/userupload/23462936/file/original-f64595c0c5eea8cfc1cfd50c5e713796.gif", use_container_width=True)
            st.write(" ")
            st.write("----")
            st.image("https://jpinfotech.org/wp-content/uploads/2023/01/JPPY2233-Prediction-Of-Used-Car-Prices.jpg", use_container_width=True)
            st.write(" ")
            st.write(" ")


    # Prediction Section
    if selected == "Get Prediction":
        st.header("🚗 Car Price Prediction")
        
        # Load the dataset once here
        df = pd.read_csv("car data.csv")

        # Load the model
        with open("Random.pkl", "rb") as f:
            model = pickle.load(f)

        # Manual class lists (these should match the ones used during training)
        df['Brand'] = df['Car_Name'].apply(lambda x: x.split()[0])
        df.drop('Car_Name', axis=1, inplace=True)

        # Get the unique values for categorical features
        brand_classes = df['Brand'].unique()
        fuel_classes = ['Petrol', 'Diesel', 'CNG']
        seller_classes = ['Dealer', 'Individual']
        transmission_classes = ['Manual', 'Automatic']

        # Form to take user input
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                brand = st.selectbox("Car Brand", options=brand_classes)
                year = st.number_input("Year of Manufacture", min_value=2000, max_value=2024)
                present_price = st.number_input("Present Price (in Lakhs)", min_value=0.0, step=0.5)
                driven_kms = st.number_input("Kilometers Driven", min_value=0, step=100)
                owner = st.selectbox("Number of Previous Owners", options=[0, 1, 2, 3])
            with col2:
                fuel_type = st.selectbox("Fuel Type", options=fuel_classes)
                selling_type = st.selectbox("Seller Type", options=seller_classes)
                transmission = st.selectbox("Transmission", options=transmission_classes)

            submit_button = st.form_submit_button("🚀 Predict Selling Price")

            if submit_button:
                # Encoding categorical features
                le_brand = LabelEncoder()
                le_brand.fit(brand_classes)
                le_fuel = LabelEncoder()
                le_fuel.fit(fuel_classes)
                le_seller = LabelEncoder()
                le_seller.fit(seller_classes)
                le_trans = LabelEncoder()
                le_trans.fit(transmission_classes)

                # Create the input dataframe for prediction
                input_df = pd.DataFrame([{
                    "Year": year,
                    "Present_Price": present_price,
                    "Driven_kms": driven_kms,
                    "Fuel_Type": le_fuel.transform([fuel_type])[0],
                    "Selling_type": le_seller.transform([selling_type])[0],
                    "Transmission": le_trans.transform([transmission])[0],
                    "Owner": str(owner),
                    "Brand": le_brand.transform([brand])[0]
                }])

                # Reorder the columns to match the order of features in the training data
                input_df = input_df[['Year', 'Present_Price', 'Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner', 'Brand']]

                # Predict the price using the trained model
                prediction = model.predict(input_df)[0]
                st.subheader(f"💰 Predicted Resale Price: :green[₹ {prediction:.2f} Lakhs]")

    # Explore Data Section (EDA)
    if selected == "Explore Data":
        df = pd.read_csv("car data.csv")
        st.subheader("📊 Exploratory Data Analysis")
        st.sidebar.header("📊 EDA Topics")
        eda_topics = [
            "1. Histogram for Each Feature",
            "2. Distribution of Selling and Present Prices",
            "3. Top 20 Car Models by Average Selling Price",
            "4. Car Price vs Year (Selling & Present)",
            "5. Correlation Heatmap"
        ]
        selected_eda = st.sidebar.selectbox("Select a Topic", eda_topics)

        # Show selected topic as subheader
        st.subheader(f"🔍 {selected_eda}")

        # Create a dropdown selectbox
        df = pd.read_csv("car data.csv")

        # Based on selection, render corresponding EDA visualization
        if selected_eda == eda_topics[0]:  # Histogram
            st.subheader("📈 Histogram for Each Feature")
            df.hist(figsize=(12, 10), bins=20, color='skyblue', edgecolor='black')
            plt.suptitle('Histograms of Features')
            st.pyplot(plt)

        elif selected_eda == eda_topics[1]:  # Distribution
            st.subheader("🏷️ Distribution of Selling and Present Prices")
            plt.figure(figsize=(8, 6))
            sns.histplot(data=df['Selling_Price'], bins=20, kde=True, color='blue', label='Selling Price', alpha=0.6)
            sns.histplot(data=df['Present_Price'], bins=20, kde=True, color='green', label='Present Price', alpha=0.6)
            plt.title('Distribution of Selling and Present Prices')
            plt.xlabel('Price (Lakh ₹)')
            plt.ylabel('Count')
            plt.legend()
            st.pyplot(plt)

        elif selected_eda == eda_topics[2]:  # Top 20 Models
            st.subheader("🚘 Top 20 Car Models by Average Selling Price")
            n = 20
            top_car_models_price = df.groupby('Car_Name')['Selling_Price'].mean().sort_values(ascending=False).head(n)
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(x=top_car_models_price.values, y=top_car_models_price.index, palette='plasma')
            plt.title(f'Top {n} Car Models by Average Selling Price')
            plt.xlabel('Average Selling Price (Lakhs)')
            plt.ylabel('Car Model')
            for i, (value, name) in enumerate(zip(top_car_models_price.values, top_car_models_price.index)):
                ax.text(value - 0.2, i, f'{value:.2f}', va='center', ha='right', color='white', fontsize=10, fontweight='bold')
            st.pyplot(plt)

        elif selected_eda == eda_topics[3]:  # Price vs Year
            st.subheader("⚙️ Car Price vs Year")
            plt.figure(figsize=(8, 6))
            sns.histplot(data=df['Selling_Price'], bins=20, kde=True, color='blue', label='Selling Price', alpha=0.6)
            sns.histplot(data=df['Present_Price'], bins=20, kde=True, color='green', label='Present Price', alpha=0.6)
            plt.title('Car Price Distribution Over the Years')
            plt.xlabel('Price (Lakh ₹)')
            plt.ylabel('Count')
            plt.legend()
            st.pyplot(plt)
            

        elif selected_eda == eda_topics[4]:  # Correlation
            st.subheader("🔥 Correlation Heatmap")
            numerical_df = df.select_dtypes(include=['int64', 'float64'])
            correlation_matrix = numerical_df.corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
            plt.title('Correlation Between Numerical Features')
            st.pyplot(plt)


    st.markdown(" ")

if __name__ == "__main__":
    main()
