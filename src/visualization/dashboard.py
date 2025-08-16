"""
Warranty Claims Fraud Detection - Visualization Dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import yaml
import streamlit as st
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Custom imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.evaluate_model import ModelEvaluator
from data.preprocessing import DataPreprocessor

# Set page configuration
st.set_page_config(
    page_title="Warranty Claims Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FraudDetectionDashboard:
    """
    Interactive dashboard for warranty claims fraud detection analysis
    """
    
    def __init__(self):
        """Initialize the dashboard"""
        self.config = self._load_config()
        self.data = None
        self.processed_data = None
        self.model_results = None
        
    def _load_config(self):
        """Load configuration"""
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    @st.cache_data
    def load_data(_self):
        """Load and cache data"""
        try:
            # Load raw data
            raw_data_path = _self.config['data']['raw_data_path']
            data = pd.read_csv(raw_data_path)
            
            # Remove index column if exists
            if 'Unnamed: 0' in data.columns:
                data = data.drop('Unnamed: 0', axis=1)
            
            return data
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def create_eda_plots(self, data):
        """Create exploratory data analysis plots"""
        st.header("📊 Exploratory Data Analysis")
        
        # Data overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Claims", data.shape[0])
        
        with col2:
            fraud_count = data['Fraud'].sum()
            st.metric("Fraudulent Claims", fraud_count)
        
        with col3:
            fraud_rate = (fraud_count / len(data)) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        
        with col4:
            avg_claim_value = data['Claim_Value'].mean()
            st.metric("Avg Claim Value", f"${avg_claim_value:,.2f}")
        
        # Class distribution
        st.subheader("Class Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for fraud distribution
            fraud_counts = data['Fraud'].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Legitimate', 'Fraudulent'],
                values=fraud_counts.values,
                hole=0.3,
                marker_colors=['lightblue', 'salmon']
            )])
            fig_pie.update_layout(title="Fraud vs Legitimate Claims")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart for product types
            product_fraud = data.groupby(['Product_type', 'Fraud']).size().unstack(fill_value=0)
            fig_bar = go.Figure(data=[
                go.Bar(name='Legitimate', x=product_fraud.index, y=product_fraud[0]),
                go.Bar(name='Fraudulent', x=product_fraud.index, y=product_fraud[1])
            ])
            fig_bar.update_layout(title="Claims by Product Type", barmode='stack')
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Geographic analysis
        st.subheader("Geographic Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            # Region-wise fraud rate
            region_stats = data.groupby('Region').agg({
                'Fraud': ['count', 'sum', 'mean']
            }).round(3)
            region_stats.columns = ['Total_Claims', 'Fraud_Count', 'Fraud_Rate']
            region_stats = region_stats.reset_index()
            
            fig_region = px.bar(
                region_stats, x='Region', y='Fraud_Rate',
                title="Fraud Rate by Region",
                color='Fraud_Rate',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_region, use_container_width=True)
        
        with col2:
            # Area-wise distribution
            area_stats = data.groupby('Area').agg({
                'Fraud': ['count', 'sum', 'mean'],
                'Claim_Value': 'mean'
            }).round(2)
            area_stats.columns = ['Total_Claims', 'Fraud_Count', 'Fraud_Rate', 'Avg_Claim_Value']
            area_stats = area_stats.reset_index()
            
            fig_area = go.Figure(data=[
                go.Bar(name='Total Claims', x=area_stats['Area'], y=area_stats['Total_Claims']),
                go.Bar(name='Fraud Count', x=area_stats['Area'], y=area_stats['Fraud_Count'])
            ])
            fig_area.update_layout(title="Claims Distribution: Urban vs Rural", barmode='group')
            st.plotly_chart(fig_area, use_container_width=True)
        
        # Claim value analysis
        st.subheader("Claim Value Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Claim value distribution by fraud status
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=data[data['Fraud'] == 0]['Claim_Value'],
                name='Legitimate',
                opacity=0.7,
                nbinsx=30
            ))
            fig_hist.add_trace(go.Histogram(
                x=data[data['Fraud'] == 1]['Claim_Value'],
                name='Fraudulent',
                opacity=0.7,
                nbinsx=30
            ))
            fig_hist.update_layout(
                title="Claim Value Distribution",
                barmode='overlay',
                xaxis_title="Claim Value",
                yaxis_title="Frequency"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot for claim values
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=data[data['Fraud'] == 0]['Claim_Value'],
                name='Legitimate',
                boxpoints='outliers'
            ))
            fig_box.add_trace(go.Box(
                y=data[data['Fraud'] == 1]['Claim_Value'],
                name='Fraudulent',
                boxpoints='outliers'
            ))
            fig_box.update_layout(
                title="Claim Value Box Plot",
                yaxis_title="Claim Value"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Product age analysis
        st.subheader("Product Age Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Product age vs fraud
            fig_scatter = px.scatter(
                data, x='Product_Age', y='Claim_Value',
                color='Fraud', size='Call_details',
                title="Product Age vs Claim Value",
                color_discrete_map={0: 'blue', 1: 'red'},
                labels={'Fraud': 'Fraud Status'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Service center analysis
            service_stats = data.groupby('Service_Centre').agg({
                'Fraud': ['count', 'sum', 'mean']
            }).round(3)
            service_stats.columns = ['Total_Claims', 'Fraud_Count', 'Fraud_Rate']
            service_stats = service_stats.reset_index()
            
            fig_service = px.scatter(
                service_stats, x='Service_Centre', y='Fraud_Rate',
                size='Total_Claims', title="Service Center Risk Analysis",
                hover_data=['Fraud_Count']
            )
            st.plotly_chart(fig_service, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numerical_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig_corr.update_layout(title="Feature Correlation Heatmap")
        st.plotly_chart(fig_corr, use_container_width=True)
    
    def create_model_performance_plots(self):
        """Create model performance visualization"""
        st.header("🤖 Model Performance Analysis")
        
        try:
            # Load model evaluation results
            models_dir = Path(self.config['paths']['models_dir'])
            cv_results_path = models_dir / "cv_results.json"
            
            if cv_results_path.exists():
                with open(cv_results_path, 'r') as f:
                    cv_results = json.load(f)
                
                # Create model comparison chart
                st.subheader("Cross-Validation Results")
                
                models = list(cv_results.keys())
                metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
                
                # Prepare data for plotting
                comparison_data = []
                for model in models:
                    for metric in metrics:
                        test_key = f'{metric}_test_mean'
                        if test_key in cv_results[model]:
                            comparison_data.append({
                                'Model': model,
                                'Metric': metric.upper(),
                                'Score': cv_results[model][test_key],
                                'Std': cv_results[model].get(f'{metric}_test_std', 0)
                            })
                
                if comparison_data:
                    df_comparison = pd.DataFrame(comparison_data)
                    
                    # Create grouped bar chart
                    fig_cv = px.bar(
                        df_comparison, x='Model', y='Score', color='Metric',
                        title="Model Performance Comparison (Cross-Validation)",
                        barmode='group',
                        error_y='Std'
                    )
                    st.plotly_chart(fig_cv, use_container_width=True)
                
                # Model details table
                st.subheader("Detailed Performance Metrics")
                
                detailed_results = []
                for model_name, results in cv_results.items():
                    detailed_results.append({
                        'Model': model_name,
                        'Accuracy': f"{results.get('accuracy_test_mean', 0):.3f} ± {results.get('accuracy_test_std', 0):.3f}",
                        'Precision': f"{results.get('precision_test_mean', 0):.3f} ± {results.get('precision_test_std', 0):.3f}",
                        'Recall': f"{results.get('recall_test_mean', 0):.3f} ± {results.get('recall_test_std', 0):.3f}",
                        'F1-Score': f"{results.get('f1_test_mean', 0):.3f} ± {results.get('f1_test_std', 0):.3f}",
                        'ROC-AUC': f"{results.get('roc_auc_test_mean', 0):.3f} ± {results.get('roc_auc_test_std', 0):.3f}"
                    })
                
                st.dataframe(pd.DataFrame(detailed_results), use_container_width=True)
            
            else:
                st.warning("No cross-validation results found. Run the training pipeline first.")
        
        except Exception as e:
            st.error(f"Error loading model performance data: {str(e)}")
    
    def create_business_insights(self, data):
        """Create business insights and recommendations"""
        st.header("💼 Business Insights & Recommendations")
        
        # Key insights
        st.subheader("Key Findings")
        
        # Calculate insights
        total_claims = len(data)
        fraud_claims = data['Fraud'].sum()
        fraud_rate = (fraud_claims / total_claims) * 100
        avg_fraud_value = data[data['Fraud'] == 1]['Claim_Value'].mean()
        avg_legit_value = data[data['Fraud'] == 0]['Claim_Value'].mean()
        
        # Service center analysis
        service_fraud_rate = data.groupby('Service_Centre')['Fraud'].agg(['count', 'sum', 'mean']).sort_values('mean', ascending=False)
        high_risk_centers = service_fraud_rate[service_fraud_rate['mean'] > fraud_rate/100].index.tolist()
        
        # Product type analysis
        product_fraud_rate = data.groupby('Product_type')['Fraud'].mean()
        riskiest_product = product_fraud_rate.idxmax()
        
        # Region analysis
        region_fraud_rate = data.groupby('Region')['Fraud'].mean()
        riskiest_region = region_fraud_rate.idxmax()
        
        # Display insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **📊 Fraud Statistics:**
            - Overall fraud rate: {fraud_rate:.2f}%
            - Average fraudulent claim value: ${avg_fraud_value:,.2f}
            - Average legitimate claim value: ${avg_legit_value:,.2f}
            - Value difference: {((avg_fraud_value - avg_legit_value) / avg_legit_value * 100):+.1f}%
            """)
            
            st.warning(f"""
            **⚠️ High-Risk Areas:**
            - Riskiest product type: **{riskiest_product}**
            - Riskiest region: **{riskiest_region}**
            - High-risk service centers: {len(high_risk_centers)} centers
            """)
        
        with col2:
            # Potential savings calculation
            potential_fraud_value = data[data['Fraud'] == 1]['Claim_Value'].sum()
            detection_rate = 0.8  # Assume 80% detection rate
            potential_savings = potential_fraud_value * detection_rate
            
            st.success(f"""
            **💰 Potential Impact:**
            - Total fraudulent claims value: ${potential_fraud_value:,.2f}
            - Potential savings (80% detection): ${potential_savings:,.2f}
            - ROI of fraud detection system: High
            """)
            
            # Recommendations
            st.info(f"""
            **🎯 Recommendations:**
            - Focus monitoring on {riskiest_product} products
            - Increase scrutiny in {riskiest_region} region
            - Review processes at {len(high_risk_centers)} high-risk service centers
            - Implement early warning system for claims > ${avg_fraud_value:,.0f}
            """)
        
        # Service center risk analysis
        st.subheader("Service Center Risk Analysis")
        
        service_analysis = data.groupby('Service_Centre').agg({
            'Fraud': ['count', 'sum', 'mean'],
            'Claim_Value': ['mean', 'sum']
        }).round(3)
        service_analysis.columns = ['Total_Claims', 'Fraud_Count', 'Fraud_Rate', 'Avg_Claim_Value', 'Total_Claim_Value']
        service_analysis = service_analysis.reset_index()
        service_analysis['Risk_Score'] = (service_analysis['Fraud_Rate'] * service_analysis['Total_Claims']).round(2)
        
        # Color code by risk level
        service_analysis['Risk_Level'] = pd.cut(
            service_analysis['Fraud_Rate'],
            bins=[0, 0.05, 0.15, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        color_map = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        
        fig_service_risk = px.scatter(
            service_analysis,
            x='Total_Claims',
            y='Fraud_Rate',
            size='Total_Claim_Value',
            color='Risk_Level',
            hover_data=['Service_Centre', 'Fraud_Count'],
            title="Service Center Risk Matrix",
            color_discrete_map=color_map
        )
        st.plotly_chart(fig_service_risk, use_container_width=True)
        
        # Detailed service center table
        st.dataframe(
            service_analysis.sort_values('Risk_Score', ascending=False),
            use_container_width=True
        )
    
    def create_prediction_interface(self):
        """Create interactive prediction interface"""
        st.header("🔮 Real-time Fraud Prediction")
        
        st.subheader("Enter Claim Details")
        
        # Create input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            region = st.selectbox("Region", ['North', 'South', 'East', 'West', 'North East'])
            state = st.text_input("State", value="Karnataka")
            area = st.selectbox("Area", ['Urban', 'Rural'])
            city = st.text_input("City", value="Bangalore")
        
        with col2:
            consumer_profile = st.selectbox("Consumer Profile", ['Business', 'Personal'])
            product_category = st.selectbox("Product Category", ['Entertainment', 'Household'])
            product_type = st.selectbox("Product Type", ['TV', 'AC'])
            purchased_from = st.selectbox("Purchased From", ['Manufacturer', 'Dealer', 'Online'])
        
        with col3:
            claim_value = st.number_input("Claim Value ($)", min_value=0.0, value=15000.0, step=100.0)
            service_centre = st.number_input("Service Centre", min_value=10, max_value=20, value=10)
            product_age = st.number_input("Product Age (days)", min_value=0, value=60)
            call_details = st.number_input("Call Details", min_value=0.0, value=0.5, step=0.1)
            purpose = st.selectbox("Purpose", ['Claim', 'Complaint', 'Inquiry'])
        
        if st.button("Predict Fraud Risk", type="primary"):
            # Create prediction data
            prediction_data = {
                'Region': region,
                'State': state,
                'Area': area,
                'City': city,
                'Consumer_profile': consumer_profile,
                'Product_category': product_category,
                'Product_type': product_type,
                'Claim_Value': claim_value,
                'Service_Centre': service_centre,
                'Product_Age': product_age,
                'Purchased_from': purchased_from,
                'Call_details': call_details,
                'Purpose': purpose
            }
            
            # Simulate prediction (in real implementation, this would call the actual model)
            # For demo purposes, we'll create a simple rule-based prediction
            risk_factors = 0
            
            if claim_value > 20000:
                risk_factors += 0.3
            if product_age < 30:
                risk_factors += 0.2
            if service_centre in [10, 15, 16]:  # High-risk centers from data analysis
                risk_factors += 0.3
            if consumer_profile == 'Business' and product_type == 'TV':
                risk_factors += 0.2
            
            fraud_probability = min(risk_factors, 0.95)
            
            # Determine risk level
            if fraud_probability < 0.3:
                risk_level = "Low"
                risk_color = "green"
            elif fraud_probability < 0.7:
                risk_level = "Medium"
                risk_color = "orange"
            else:
                risk_level = "High"
                risk_color = "red"
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Fraud Probability", f"{fraud_probability:.1%}")
            
            with col2:
                st.metric("Risk Level", risk_level)
            
            with col3:
                confidence = 1 - abs(fraud_probability - 0.5) * 2
                st.metric("Confidence", f"{confidence:.1%}")
            
            # Risk explanation
            if fraud_probability > 0.5:
                st.error(f"⚠️ HIGH RISK CLAIM - Requires manual review")
            elif fraud_probability > 0.3:
                st.warning(f"⚡ MEDIUM RISK CLAIM - Consider additional verification")
            else:
                st.success(f"✅ LOW RISK CLAIM - Can be processed normally")
            
            # Show input data
            with st.expander("View Input Data"):
                st.json(prediction_data)
    
    def run_dashboard(self):
        """Run the complete dashboard"""
        st.title("🔍 Warranty Claims Fraud Detection Dashboard")
        st.markdown("---")
        
        # Sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Select Page",
            ["📊 Data Analysis", "🤖 Model Performance", "💼 Business Insights", "🔮 Fraud Prediction"]
        )
        
        # Load data
        if self.data is None:
            with st.spinner("Loading data..."):
                self.data = self.load_data()
        
        if self.data is None:
            st.error("Could not load data. Please check the data file path.")
            return
        
        # Show selected page
        if page == "📊 Data Analysis":
            self.create_eda_plots(self.data)
        
        elif page == "🤖 Model Performance":
            self.create_model_performance_plots()
        
        elif page == "💼 Business Insights":
            self.create_business_insights(self.data)
        
        elif page == "🔮 Fraud Prediction":
            self.create_prediction_interface()
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center'>
                <p>Warranty Claims Fraud Detection System | Built with ❤️ using Streamlit</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# Main execution
if __name__ == "__main__":
    dashboard = FraudDetectionDashboard()
    dashboard.run_dashboard()
