import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Employee Burnout Analysis",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the employee burnout data"""
    try:
        data = pd.read_excel("employee_burnout_analysis-AI.xlsx")
        data["Date of Joining"] = pd.to_datetime(data["Date of Joining"])
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def train_model(data):
    """Train the Random Forest model"""
    # Create a copy to avoid modifying cached data
    data_copy = data.copy()
    
    # Label encoding with separate encoders
    gender_encoder = preprocessing.LabelEncoder()
    company_encoder = preprocessing.LabelEncoder()
    wfh_encoder = preprocessing.LabelEncoder()
    
    data_copy['GenderLabel'] = gender_encoder.fit_transform(data_copy['Gender'].values)
    data_copy['Company_TypeLabel'] = company_encoder.fit_transform(data_copy['Company Type'].values)
    data_copy['WFH_Setup_AvailableLabel'] = wfh_encoder.fit_transform(data_copy['WFH Setup Available'].values)
    
    # Feature selection
    columns = ['Designation', 'Resource Allocation', 'Mental Fatigue Score', 
               'GenderLabel', 'Company_TypeLabel', 'WFH_Setup_AvailableLabel']
    x = data_copy[columns].fillna(data_copy[columns].median())
    y = data_copy['Burn Rate'].fillna(data_copy['Burn Rate'].median())
    
    # PCA
    pca = PCA(0.95)
    x_pca = pca.fit_transform(x)
    
    # Train-test split
    x_train_pca, x_test, y_train, y_test = train_test_split(
        x_pca, y, test_size=0.25, random_state=10
    )
    
    # Train model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(x_train_pca, y_train)
    
    # Calculate metrics
    train_pred = rf_model.predict(x_train_pca)
    test_pred = rf_model.predict(x_test)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    return {
        'model': rf_model,
        'pca': pca,
        'label_encoders': {
            'gender': gender_encoder,
            'company_type': company_encoder,
            'wfh': wfh_encoder
        },
        'train_r2': train_r2,
        'test_r2': test_r2,
        'x_train': x_train_pca,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test
    }

def main():
    st.markdown('<h1 class="main-header">🔥 Employee Burnout Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    if data is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Choose a page",
        ["📊 Overview", "📈 Visualizations", "🤖 Predict Burnout", "📋 Data Explorer"]
    )
    
    if page == "📊 Overview":
        show_overview(data)
    elif page == "📈 Visualizations":
        show_visualizations(data)
    elif page == "🤖 Predict Burnout":
        show_predictor(data)
    elif page == "📋 Data Explorer":
        show_data_explorer(data)

def show_overview(data):
    st.header("📊 Dataset Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", f"{len(data):,}")
    
    with col2:
        avg_burn_rate = data['Burn Rate'].mean()
        st.metric("Avg Burn Rate", f"{avg_burn_rate:.2%}")
    
    with col3:
        avg_fatigue = data['Mental Fatigue Score'].mean()
        st.metric("Avg Mental Fatigue", f"{avg_fatigue:.2f}")
    
    with col4:
        avg_resource = data['Resource Allocation'].mean()
        st.metric("Avg Resource Allocation", f"{avg_resource:.2f}")
    
    st.divider()
    
    # Dataset info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Information")
        st.write(f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns")
        st.write(f"**Memory Usage:** {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.subheader("Missing Values")
        missing = data.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Percentage': (missing.values / len(data) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values!")
    
    with col2:
        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
            'Column': data.dtypes.index,
            'Data Type': data.dtypes.values
        })
        st.dataframe(dtype_df, use_container_width=True)
        
        st.subheader("Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)
    
    # Model performance
    st.divider()
    st.subheader("🤖 Model Performance")
    
    model_data = train_model(data)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training R² Score", f"{model_data['train_r2']:.4f}", 
                 f"{model_data['train_r2']*100:.2f}%")
    with col2:
        st.metric("Test R² Score", f"{model_data['test_r2']:.4f}",
                 f"{model_data['test_r2']*100:.2f}%")

def show_visualizations(data):
    st.header("📈 Data Visualizations")
    
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Correlation Heatmap", "Gender Distribution", "Company Type Distribution",
         "WFH Setup Distribution", "Burn Rate Distribution", "Mental Fatigue Distribution",
         "Resource Allocation Distribution", "Designation Distribution"]
    )
    
    if viz_type == "Correlation Heatmap":
        st.subheader("Correlation Heatmap")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr = data[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", 
                       color_continuous_scale="RdBu_r",
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Gender Distribution":
        st.subheader("Gender Distribution")
        fig = px.histogram(data, x="Gender", color="Gender",
                          title="Distribution of Gender",
                          color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
        
        # Burn rate by gender
        fig2 = px.box(data, x="Gender", y="Burn Rate",
                     title="Burn Rate by Gender",
                     color="Gender",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig2, use_container_width=True)
    
    elif viz_type == "Company Type Distribution":
        st.subheader("Company Type Distribution")
        fig = px.histogram(data, x="Company Type", color="Company Type",
                          title="Distribution of Company Type",
                          color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
        
        # Burn rate by company type
        fig2 = px.box(data, x="Company Type", y="Burn Rate",
                     title="Burn Rate by Company Type",
                     color="Company Type",
                     color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig2, use_container_width=True)
    
    elif viz_type == "WFH Setup Distribution":
        st.subheader("WFH Setup Available Distribution")
        fig = px.histogram(data, x="WFH Setup Available", color="WFH Setup Available",
                          title="Distribution of WFH Setup Available",
                          color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)
        
        # Burn rate by WFH
        fig2 = px.box(data, x="WFH Setup Available", y="Burn Rate",
                     title="Burn Rate by WFH Setup Availability",
                     color="WFH Setup Available",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2, use_container_width=True)
    
    elif viz_type == "Burn Rate Distribution":
        st.subheader("Burn Rate Distribution")
        fig = px.histogram(data, x="Burn Rate", nbins=50,
                          title="Distribution of Burn Rate",
                          color_discrete_sequence=['indianred'])
        fig.update_layout(bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Mental Fatigue Distribution":
        st.subheader("Mental Fatigue Score Distribution")
        fig = px.histogram(data, x="Mental Fatigue Score", nbins=50,
                          title="Distribution of Mental Fatigue Score",
                          color_discrete_sequence=['orange'])
        fig.update_layout(bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot: Mental Fatigue vs Burn Rate
        fig2 = px.scatter(data, x="Mental Fatigue Score", y="Burn Rate",
                         title="Mental Fatigue Score vs Burn Rate",
                         trendline="ols",
                         color_discrete_sequence=['purple'])
        st.plotly_chart(fig2, use_container_width=True)
    
    elif viz_type == "Resource Allocation Distribution":
        st.subheader("Resource Allocation Distribution")
        fig = px.histogram(data, x="Resource Allocation", nbins=50,
                          title="Distribution of Resource Allocation",
                          color_discrete_sequence=['teal'])
        fig.update_layout(bargap=0.2)
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot: Resource Allocation vs Burn Rate
        fig2 = px.scatter(data, x="Resource Allocation", y="Burn Rate",
                         title="Resource Allocation vs Burn Rate",
                         trendline="ols",
                         color_discrete_sequence=['green'])
        st.plotly_chart(fig2, use_container_width=True)
    
    elif viz_type == "Designation Distribution":
        st.subheader("Designation Distribution")
        fig = px.histogram(data, x="Designation", color="Designation",
                          title="Distribution of Designation",
                          color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig, use_container_width=True)
        
        # Burn rate by designation
        fig2 = px.box(data, x="Designation", y="Burn Rate",
                     title="Burn Rate by Designation",
                     color="Designation",
                     color_discrete_sequence=px.colors.sequential.Viridis)
        st.plotly_chart(fig2, use_container_width=True)

def show_predictor(data):
    st.header("🤖 Burnout Rate Predictor")
    st.write("Enter employee details to predict their burnout rate:")
    
    model_data = train_model(data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        designation = st.selectbox("Designation", [1, 2, 3, 4, 5])
        resource_allocation = st.slider("Resource Allocation", 0.0, 10.0, 5.0, 0.1)
        mental_fatigue = st.slider("Mental Fatigue Score", 0.0, 10.0, 5.0, 0.1)
    
    with col2:
        gender = st.selectbox("Gender", data['Gender'].unique())
        company_type = st.selectbox("Company Type", data['Company Type'].unique())
        wfh_setup = st.selectbox("WFH Setup Available", data['WFH Setup Available'].unique())
    
    if st.button("🔮 Predict Burnout Rate", type="primary"):
        # Encode categorical variables
        gender_encoded = model_data['label_encoders']['gender'].transform([gender])[0]
        company_encoded = model_data['label_encoders']['company_type'].transform([company_type])[0]
        wfh_encoded = model_data['label_encoders']['wfh'].transform([wfh_setup])[0]
        
        # Prepare input
        input_features = np.array([[designation, resource_allocation, mental_fatigue,
                                   gender_encoded, company_encoded, wfh_encoded]])
        
        # Apply PCA
        input_pca = model_data['pca'].transform(input_features)
        
        # Predict
        prediction = model_data['model'].predict(input_pca)[0]
        
        # Display result
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background-color: #f0f2f6; 
                        border-radius: 1rem; border: 3px solid #FF6B6B;'>
                <h2 style='color: #FF6B6B; margin-bottom: 1rem;'>Predicted Burnout Rate</h2>
                <h1 style='font-size: 4rem; color: #FF6B6B; margin: 0;'>{prediction:.2%}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Interpretation
        st.divider()
        if prediction < 0.3:
            st.success("✅ **Low Risk**: Employee shows low burnout risk. Continue current support measures.")
        elif prediction < 0.5:
            st.warning("⚠️ **Moderate Risk**: Employee may be experiencing moderate stress. Consider additional support.")
        elif prediction < 0.7:
            st.error("🔴 **High Risk**: Employee shows high burnout risk. Immediate intervention recommended.")
        else:
            st.error("🚨 **Critical Risk**: Employee shows critical burnout levels. Urgent action required!")

def show_data_explorer(data):
    st.header("📋 Data Explorer")
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender_filter = st.multiselect("Gender", data['Gender'].unique(), default=data['Gender'].unique())
    with col2:
        company_filter = st.multiselect("Company Type", data['Company Type'].unique(), 
                                        default=data['Company Type'].unique())
    with col3:
        wfh_filter = st.multiselect("WFH Setup Available", data['WFH Setup Available'].unique(),
                                   default=data['WFH Setup Available'].unique())
    
    # Apply filters
    filtered_data = data[
        (data['Gender'].isin(gender_filter)) &
        (data['Company Type'].isin(company_filter)) &
        (data['WFH Setup Available'].isin(wfh_filter))
    ]
    
    st.write(f"**Showing {len(filtered_data)} of {len(data)} employees**")
    
    # Display data
    st.subheader("Employee Data")
    st.dataframe(filtered_data, use_container_width=True, height=400)
    
    # Download button
    csv = filtered_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_employee_data.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()

