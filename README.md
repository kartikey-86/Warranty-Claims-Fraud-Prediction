# Warranty Claims Fraud Detection

## 📌 Project Overview
Fraudulent warranty claims inflate service costs and erode margins. This project detects suspicious claims and surfaces investigation priorities.

## 🎯 Business Problem
Identify and prioritize likely fraudulent claims to reduce payouts and investigation time.

## 🗂️ Dataset (example schema)
`claim_id`, `customer_id`, `product_type`, `purchase_date`, `claim_date`, `failure_code`, `service_center`, `claim_amount`, `approved` (Y/N), `fraud_label` (Y/N or NULL)

## 🛠 Tools
Python (Pandas, Scikit‑learn), SQL, Power BI

## 🔬 Method
- Data quality checks (duplicates, date validity, claim‑after‑purchase logic).  
- Feature engineering (days‑to‑failure, repeat claims, center frequency).  
- Modeling (Logistic Regression / Random Forest).  
- Threshold tuning for high **precision** to minimize false positives.  
- BI: Ops dashboard for fraud hotspots & savings estimate.

## 🔑 Example Insights
- Claims within **< 14 days** of purchase and specific failure codes have elevated fraud probability.  
- 5% of service centers contribute 40% of suspicious volume.

## 📊 Power BI Dashboard (Layout)
- KPIs: #Claims, Suspicious %, Est. Savings, Precision@K.  
- Fraud Heatmap: Service Center × Product Type.  
- Time Trend: weekly suspicious claims.  
- Drill‑through: claim details with risk score.


---
**Folders**
```
Warranty_Fraud_Detection/
├─ README.md
├─ scripts/
│  └─ fraud_analysis.sql
└─ bi/
   ├─ dashboard_layout.md
   └─ dax_measures.md
```
