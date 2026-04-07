import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;}
.stApp{background:#f1f5f9;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f172a,#1e293b)!important;}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] span,[data-testid="stSidebar"] div,[data-testid="stSidebar"] input{color:#cbd5e1!important;}
[data-testid="stSidebar"] label{color:#94a3b8!important;font-size:0.72rem!important;font-weight:600!important;text-transform:uppercase!important;letter-spacing:0.07em!important;}
[data-testid="stSidebar"] .stSelectbox>div>div{background:#1e293b!important;border:1px solid #334155!important;border-radius:8px!important;}
[data-testid="collapsedControl"]{background:#1e3a5f!important;border:2px solid #3b82f6!important;border-radius:0 8px 8px 0!important;opacity:1!important;}
.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:.8rem;}
.metric-card{background:#fff;border-radius:10px;padding:.85rem 1rem;border:1px solid #e2e8f0;box-shadow:0 1px 4px rgba(0,0,0,.05);}
.metric-card .ml{font-size:.68rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;margin:0 0 .2rem;}
.metric-card .mv{font-size:1.5rem;font-weight:700;color:#0f172a;margin:0;font-family:'JetBrains Mono',monospace;}
.summary-bar{background:white;border-radius:12px;padding:1rem 1.4rem;border:1px solid #e2e8f0;display:flex;}
.si{flex:1;text-align:center;padding:0 .5rem;border-right:1px solid #f1f5f9;}
.si:last-child{border-right:none;}
.sl{font-size:.68rem;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:.06em;margin:0;}
.sv{font-size:1.05rem;font-weight:700;color:#0f172a;margin:.15rem 0 0;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1rem!important;}

/* Tab labels always visible */
.stTabs [data-baseweb="tab-list"]{gap:4px;background:#f1f5f9!important;border-radius:10px;padding:4px;}
.stTabs [data-baseweb="tab"]{background:#f1f5f9!important;border-radius:8px!important;padding:8px 16px!important;color:#475569!important;font-weight:500!important;font-size:.85rem!important;border:none!important;}
.stTabs [aria-selected="true"]{background:#ffffff!important;color:#1d4ed8!important;font-weight:700!important;box-shadow:0 1px 4px rgba(0,0,0,.1)!important;}
.stTabs [data-baseweb="tab"]:hover{color:#1d4ed8!important;background:#e8f0fe!important;}
.stTabs [data-baseweb="tab-highlight"]{display:none!important;}
.stTabs [data-baseweb="tab-border"]{display:none!important;}
</style>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    path = Path(__file__).parent / "model" / "churn_model.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

md = load_model()
model=md["model"]; model_name=md["model_name"]
scaler=md["scaler"]; feat_cols=md["feature_cols"]
metrics=md["metrics"]; importances=md["importances"]

with st.sidebar:
    st.markdown("## 📉 Customer Profile")
    st.caption("Adjust inputs to predict churn risk.")
    st.markdown("---")
    def sdiv(t): st.markdown(f'<p style="font-size:.68rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.1em;margin:.8rem 0 .2rem;padding-top:.5rem;border-top:1px solid #334155;">{t}</p>', unsafe_allow_html=True)
    sdiv("Account")
    tenure=st.slider("Tenure (months)",1,72,12)
    contract=st.selectbox("Contract Type",["Month-to-month","One year","Two year"])
    payment_method=st.selectbox("Payment Method",["Electronic check","Mailed check","Bank transfer","Credit card"])
    paperless=st.selectbox("Paperless Billing",["Yes","No"])
    sdiv("Charges")
    monthly_charges=st.slider("Monthly Charges ($)",18.0,118.0,65.0,step=0.5)
    total_charges=st.slider("Total Charges ($)",18.0,8500.0,float(monthly_charges*tenure),step=10.0)
    sdiv("Services")
    internet_service=st.selectbox("Internet Service",["Fiber optic","DSL","No"])
    multiple_lines=st.selectbox("Multiple Lines",["Yes","No","No phone service"])
    online_security=st.selectbox("Online Security",["Yes","No","No internet service"])
    online_backup=st.selectbox("Online Backup",["Yes","No","No internet service"])
    device_protect=st.selectbox("Device Protection",["Yes","No","No internet service"])
    tech_support=st.selectbox("Tech Support",["Yes","No","No internet service"])
    streaming_tv=st.selectbox("Streaming TV",["Yes","No","No internet service"])
    streaming_movies=st.selectbox("Streaming Movies",["Yes","No","No internet service"])
    sdiv("Demographics")
    gender=st.selectbox("Gender",["Male","Female"])
    senior_citizen=st.selectbox("Senior Citizen",["No","Yes"])
    partner=st.selectbox("Has Partner",["Yes","No"])
    dependents=st.selectbox("Has Dependents",["Yes","No"])
    phone_service=st.selectbox("Phone Service",["Yes","No"])

svc={
    "MultipleLines_Yes":multiple_lines=="Yes","OnlineSecurity_Yes":online_security=="Yes",
    "OnlineBackup_Yes":online_backup=="Yes","DeviceProtection_Yes":device_protect=="Yes",
    "TechSupport_Yes":tech_support=="Yes","StreamingTV_Yes":streaming_tv=="Yes",
    "StreamingMovies_Yes":streaming_movies=="Yes"
}
sc=sum(svc.values())
row={
    "tenure":tenure,"MonthlyCharges":monthly_charges,"TotalCharges":total_charges,
    "gender":1 if gender=="Female" else 0,"SeniorCitizen":1 if senior_citizen=="Yes" else 0,
    "Partner":1 if partner=="Yes" else 0,"Dependents":1 if dependents=="Yes" else 0,
    "PhoneService":1 if phone_service=="Yes" else 0,"PaperlessBilling":1 if paperless=="Yes" else 0,
    "Contract_Month":1 if contract=="Month-to-month" else 0,"Contract_OneYear":1 if contract=="One year" else 0,
    "Internet_Fiber":1 if internet_service=="Fiber optic" else 0,"Internet_DSL":1 if internet_service=="DSL" else 0,
    "Payment_Echeck":1 if payment_method=="Electronic check" else 0,
    "Payment_MailedCheck":1 if payment_method=="Mailed check" else 0,
    "AvgMonthlySpend":total_charges/(tenure+1),"ChargePerService":monthly_charges/(sc+1),
    **{k:int(v) for k,v in svc.items()}
}
X=pd.DataFrame([row])[feat_cols]
prob=model.predict_proba(scaler.transform(X) if model_name=="Logistic Regression" else X)[0][1]
risk_pct=round(prob*100,1)

if prob>=0.65: rl,rc,re,bc="HIGH CHURN RISK","linear-gradient(135deg,#991b1b,#dc2626)","🔴","#dc2626"; rd="High likelihood of churning. Immediate intervention recommended."
elif prob>=0.40: rl,rc,re,bc="MODERATE CHURN RISK","linear-gradient(135deg,#92400e,#d97706)","🟡","#d97706"; rd="Moderate risk. Proactive retention outreach advised."
else: rl,rc,re,bc="LOW CHURN RISK","linear-gradient(135deg,#14532d,#16a34a)","🟢","#16a34a"; rd="Customer is likely to stay. Continue standard engagement."

seg,sc2=("New Customer","#dc2626") if tenure<=12 else ("Growing Customer","#d97706") if tenure<=36 else ("Loyal Customer","#16a34a")
clv=monthly_charges*max(24,tenure)

readable={
    "tenure":"Customer Tenure","MonthlyCharges":"Monthly Charges","TotalCharges":"Total Charges",
    "Contract_Month":"Month-to-Month Contract","Internet_Fiber":"Fiber Optic Internet",
    "Payment_Echeck":"Electronic Check Payment","AvgMonthlySpend":"Avg Monthly Spend",
    "ChargePerService":"Charge per Service","Contract_OneYear":"One-Year Contract",
    "OnlineSecurity_Yes":"Has Online Security","TechSupport_Yes":"Has Tech Support",
    "Internet_DSL":"DSL Internet","SeniorCitizen":"Senior Citizen",
    "MultipleLines_Yes":"Multiple Lines","OnlineBackup_Yes":"Online Backup",
    "DeviceProtection_Yes":"Device Protection","StreamingTV_Yes":"Streaming TV",
    "StreamingMovies_Yes":"Streaming Movies","Payment_MailedCheck":"Mailed Check",
    "PaperlessBilling":"Paperless Billing","Partner":"Has Partner",
    "Dependents":"Has Dependents","gender":"Gender","PhoneService":"Phone Service",
}
sorted_imp=sorted(importances.items(),key=lambda x:x[1],reverse=True)[:12]
labels=[readable.get(f,f.replace("_"," ")) for f,_ in sorted_imp]
values=[round(v,4) for _,v in sorted_imp]

strategies=[]
if contract=="Month-to-month": strategies.append(("🔒","Offer discounted annual contract — month-to-month customers churn 3× more often"))
if tenure<=12: strategies.append(("🎁","Enroll in new customer loyalty program with first-year perks"))
if monthly_charges>80: strategies.append(("💰","Offer personalized bill review — high monthly charges are #1 churn driver"))
if payment_method=="Electronic check": strategies.append(("💳","Incentivize switch to auto-pay — reduces payment friction and churn"))
if internet_service=="Fiber optic" and online_security=="No": strategies.append(("🔐","Bundle Online Security — fiber users without it churn significantly more"))
if tech_support=="No": strategies.append(("🛠️","Offer 3-month free Tech Support trial — proven to reduce churn"))
if senior_citizen=="Yes": strategies.append(("👴","Assign dedicated senior support agent — higher churn sensitivity"))
if not strategies:
    strategies=[("✅","Strong retention profile. Continue standard engagement."),("📞","Schedule quarterly satisfaction check-in"),("🎯","Eligible for loyalty upgrade or referral program")]

st.markdown(f"""
<div style="background:linear-gradient(135deg,#0f172a,#1e3a5f,#1d4ed8);padding:1.6rem 2rem;border-radius:16px;margin-bottom:1.2rem;display:flex;justify-content:space-between;align-items:center;box-shadow:0 4px 20px rgba(15,23,42,.3);">
    <div>
        <h1 style="font-size:1.6rem;font-weight:700;margin:0 0 .25rem;color:#fff;letter-spacing:-.02em;">📉 Customer Churn Predictor</h1>
        <p style="font-size:.82rem;color:#93c5fd;margin:0;">ML-powered risk assessment · {model_name} · Trained on 10,000 Telco records · 19 features</p>
    </div>
   
</div>
""", unsafe_allow_html=True)

left,right=st.columns([1,1.75],gap="large")

with left:
    st.markdown(f"""
    <div style="background:{rc};padding:1.6rem 1.4rem;border-radius:16px;text-align:center;margin-bottom:1rem;box-shadow:0 4px 16px rgba(0,0,0,.15);">
        <p style="font-size:.72rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:rgba(255,255,255,.8);margin:0;">{re} Churn Probability Score</p>
        <div style="font-size:3.5rem;font-weight:800;color:#fff;margin:.3rem 0;font-family:'JetBrains Mono',monospace;line-height:1;">{risk_pct}%</div>
        <p style="font-size:1rem;font-weight:700;color:rgba(255,255,255,.95);margin:.3rem 0 .2rem;">{rl}</p>
        <p style="font-size:.78rem;color:rgba(255,255,255,.75);margin:0;">{rd}</p>
    </div>
    <div style="text-align:center;margin-bottom:1rem;">
        <span style="display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border-radius:20px;font-size:.78rem;font-weight:600;border:1.5px solid {sc2};color:{sc2};background:{sc2}18;">
            👤 {seg} — {tenure} months tenure
        </span>
    </div>
    <p style="font-size:.78rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin:0 0 .5rem;padding-bottom:.3rem;border-bottom:2px solid #e2e8f0;">Model Performance</p>
    <div class="metric-grid">
        <div class="metric-card"><p class="ml">Accuracy</p><p class="mv">{metrics['accuracy']:.1%}</p></div>
        <div class="metric-card"><p class="ml">F1 Score</p><p class="mv">{metrics['f1']:.2f}</p></div>
        <div class="metric-card"><p class="ml">Precision</p><p class="mv">{metrics['precision']:.1%}</p></div>
        <div class="metric-card"><p class="ml">ROC-AUC</p><p class="mv">{metrics['roc_auc']:.2f}</p></div>
    </div>
    <div style="background:#eff6ff;border-radius:10px;padding:.85rem 1rem;border-left:3px solid #3b82f6;font-size:.79rem;color:#1e40af;line-height:1.5;">
        <strong> About this model</strong><br>
        {model_name} trained on 10,000 Telco records. Three models compared — {model_name} selected based on highest F1 score.
    </div>
    """, unsafe_allow_html=True)

with right:
    tab1,tab2,tab3=st.tabs(["📊 Feature Importance","💡 Retention Strategy","📈 Risk Gauge"])

    with tab1:
        st.markdown('<p style="font-size:.78rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin:0 0 .5rem;padding-bottom:.3rem;border-bottom:2px solid #e2e8f0;">Top Predictive Factors</p>', unsafe_allow_html=True)
        fig_imp=go.Figure(go.Bar(
            x=values,y=labels,orientation="h",
            marker=dict(color=values,colorscale=[[0,"#bfdbfe"],[.5,"#3b82f6"],[1,"#1d4ed8"]],showscale=False),
            text=[f"{v:.3f}" for v in values],textposition="outside",
            textfont=dict(size=11,color="#374151"),cliponaxis=False,
        ))
        fig_imp.update_layout(
            margin=dict(l=0,r=70,t=10,b=20),paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",height=390,
            yaxis=dict(autorange="reversed",tickfont=dict(size=12,color="#374151"),showgrid=False),
            xaxis=dict(showgrid=True,gridcolor="#f1f5f9",
                title=dict(text="Importance Score",font=dict(size=11,color="#64748b")),
                tickfont=dict(size=10,color="#94a3b8")),
            showlegend=False,
        )
        st.plotly_chart(fig_imp,use_container_width=True)

    with tab2:
        # ALL inline styles - guaranteed to render regardless of tab lazy loading
        BOX="background:#ffffff;border-radius:12px;padding:1.3rem 1.4rem;border:1px solid #e2e8f0;box-shadow:0 2px 8px rgba(0,0,0,0.07);margin-bottom:1rem;"
        HDR="font-size:.88rem;font-weight:700;color:#1e293b;margin:0 0 .8rem;padding-bottom:.45rem;border-bottom:2px solid #f1f5f9;display:block;"
        ITM="display:flex;gap:10px;align-items:flex-start;font-size:.84rem;color:#374151;margin-bottom:.55rem;line-height:1.5;padding:.3rem 0;"
        SEC="font-size:.78rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin:.9rem 0 .6rem;padding-bottom:.3rem;border-bottom:2px solid #e2e8f0;display:block;"
        ROW="display:flex;justify-content:space-between;align-items:center;padding:.5rem 0;border-bottom:1px solid #f8fafc;"
        LBL="font-size:.84rem;color:#64748b;"
        VAL="font-size:.86rem;font-weight:700;color:#0f172a;font-family:monospace;"

        st.markdown(f'<span style="{SEC}">Personalized Retention Actions</span>', unsafe_allow_html=True)
        html=f'<div style="{BOX}"><span style="{HDR}">Recommended Actions</span>'
        for icon,txt in strategies:
            html+=f'<div style="{ITM}"><span style="flex-shrink:0;font-size:1rem;">{icon}</span><span>{txt}</span></div>'
        html+='</div>'
        st.markdown(html,unsafe_allow_html=True)

        st.markdown(f'<span style="{SEC}">Revenue at Risk Analysis</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="{BOX}">
            <span style="{HDR}">💵 Financial Impact Estimate</span>
            <div style="{ROW}"><span style="{LBL}">Monthly Revenue</span><span style="{VAL}">${monthly_charges:,.2f}</span></div>
            <div style="{ROW}"><span style="{LBL}">Estimated 2-Year CLV</span><span style="{VAL}">${clv:,.0f}</span></div>
            <div style="{ROW}"><span style="{LBL}">Churn-Weighted Revenue Loss</span><span style="{VAL};color:#dc2626;">${clv*prob:,.0f}</span></div>
            <div style="{ROW};border-bottom:none;"><span style="{LBL}">Retention Cost (10% off × 6 mo)</span><span style="{VAL};color:#16a34a;">${monthly_charges*.1*6:,.0f}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        SEC3="font-size:.78rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin:.9rem 0 .6rem;padding-bottom:.3rem;border-bottom:2px solid #e2e8f0;display:block;"
        st.markdown(f'<span style="{SEC3}">Churn Risk Gauge</span>', unsafe_allow_html=True)
        fig_g=go.Figure(go.Indicator(
            mode="gauge+number",value=risk_pct,
            number={"suffix":"%","font":{"size":44,"color":"#0f172a","family":"JetBrains Mono"}},
            gauge={
                "axis":{"range":[0,100],"tickwidth":1,"tickfont":{"size":11,"color":"#64748b"},
                        "tickvals":[0,25,40,65,100],"ticktext":["0","25","40","65","100"]},
                "bar":{"color":bc,"thickness":0.3},"bgcolor":"white","borderwidth":0,
                "steps":[{"range":[0,40],"color":"#dcfce7"},{"range":[40,65],"color":"#fef9c3"},{"range":[65,100],"color":"#fee2e2"}],
                "threshold":{"line":{"color":"#0f172a","width":3},"thickness":0.8,"value":risk_pct},
            },
            title={"text":f"<b>{rl}</b>","font":{"size":13,"color":bc}},
        ))
        fig_g.update_layout(height=290,margin=dict(l=40,r=40,t=60,b=10),paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g,use_container_width=True)

        st.markdown(f'<span style="{SEC3}">Benchmark Comparison</span>', unsafe_allow_html=True)
        bmarks={"This Customer":risk_pct,"Month-to-Month Avg":68.0,"New Customer Avg":55.0,"Fiber Optic Avg":42.0,"2-Yr Contract Avg":11.0,"Overall Avg":26.5}
        bcols=[bc if k=="This Customer" else "#64748b" if "Overall" in k else "#cbd5e1" for k in bmarks]
        fig_b=go.Figure(go.Bar(
            x=list(bmarks.keys()),y=list(bmarks.values()),marker_color=bcols,
            text=[f"{v:.1f}%" for v in bmarks.values()],textposition="outside",
            textfont=dict(size=11,color="#374151"),cliponaxis=False,
        ))
        fig_b.update_layout(
            margin=dict(l=10,r=10,t=20,b=60),paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",height=240,
            yaxis=dict(showgrid=True,gridcolor="#f1f5f9",title=dict(text="Churn Rate (%)",font=dict(size=11,color="#64748b"))),
            xaxis=dict(tickfont=dict(size=10,color="#374151"),tickangle=-15),
            showlegend=False,
        )
        st.plotly_chart(fig_b,use_container_width=True)

st.markdown('<p style="font-size:.78rem;font-weight:700;color:#475569;text-transform:uppercase;letter-spacing:.08em;margin:1rem 0 .5rem;padding-bottom:.3rem;border-bottom:2px solid #e2e8f0;">Current Customer Summary</p>', unsafe_allow_html=True)
sc3=sum([multiple_lines=="Yes",online_security=="Yes",online_backup=="Yes",device_protect=="Yes",tech_support=="Yes",streaming_tv=="Yes",streaming_movies=="Yes"])
st.markdown(f"""
<div class="summary-bar">
    <div class="si"><p class="sl">Contract</p><p class="sv">{contract.split()[0]}</p></div>
    <div class="si"><p class="sl">Tenure</p><p class="sv">{tenure} mo</p></div>
    <div class="si"><p class="sl">Monthly $</p><p class="sv">${monthly_charges:.0f}</p></div>
    <div class="si"><p class="sl">Internet</p><p class="sv">{internet_service.split()[0]}</p></div>
    <div class="si"><p class="sl">Payment</p><p class="sv">{payment_method.split()[0]}</p></div>
    <div class="si"><p class="sl">Services</p><p class="sv">{sc3} active</p></div>
    <div class="si"><p class="sl">Segment</p><p class="sv" style="color:{sc2};">{seg.split()[0]}</p></div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div style="text-align:center;margin-top:1.5rem;padding-top:1rem;border-top:1px solid #e2e8f0;color:#94a3b8;font-size:.72rem;"> <a href="https://www.sunkarasaitejaprasad.com/" style="color:#3b82f6;text-decoration:none;"> Check My Portfolio</a> · Gradient Boosting · Telco Churn (10,000 records)</div>', unsafe_allow_html=True)
