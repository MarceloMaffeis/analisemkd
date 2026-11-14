"""
Estilos e componentes visuais modernos para a aplicação Streamlit.
Design voltado a dashboards executivos de Marketing Digital e Data Science.
"""

import streamlit as st

def apply_custom_styles():
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        /* Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, #4338CA 0%, #6366F1 50%, #818CF8 100%);
            padding: 2.2rem 2.4rem;
            border-radius: 16px;
            color: #ffffff;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.35);
        }
        
        .hero-banner h1 {
            color: #ffffff !important;
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }
        
        .hero-banner p {
            color: #E0E7FF !important;
            font-size: 1.05rem;
            max-width: 800px;
            margin: 0;
            line-height: 1.5;
        }

        /* Metric Cards */
        .metric-card {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1.2rem 1.4rem;
            backdrop-filter: blur(8px);
            transition: all 0.2s ease-in-out;
            margin-bottom: 1rem;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.4);
            box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.3);
        }

        .metric-label {
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #94A3B8;
            margin-bottom: 0.3rem;
        }

        .metric-value {
            font-size: 1.85rem;
            font-weight: 800;
            color: #F8FAFC;
            letter-spacing: -0.02em;
        }

        .metric-sub {
            font-size: 0.85rem;
            color: #10B981;
            font-weight: 500;
            margin-top: 0.2rem;
        }

        /* Section Cards */
        .section-card {
            background: #1E293B;
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 14px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        /* Pills and Badges */
        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .badge-reg {
            background: rgba(59, 130, 246, 0.2);
            color: #60A5FA;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }

        .badge-clf {
            background: rgba(16, 185, 129, 0.2);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .badge-clu {
            background: rgba(245, 158, 11, 0.2);
            color: #FBBF24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        /* Insight Callouts */
        .insight-box {
            background: rgba(99, 102, 241, 0.08);
            border-left: 4px solid #6366F1;
            padding: 1rem 1.2rem;
            border-radius: 0 10px 10px 0;
            margin: 1.2rem 0;
        }

        .insight-box h4 {
            color: #A5B4FC;
            margin: 0 0 0.4rem 0;
            font-size: 1rem;
            font-weight: 700;
        }

        .insight-box p {
            color: #CBD5E1;
            margin: 0;
            font-size: 0.92rem;
            line-height: 1.5;
        }
        
        /* Simulators Box */
        .simulator-container {
            background: #0F172A;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.5rem;
            margin-top: 1rem;
        }

        /* Custom buttons styling */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }

        /* Table styling improvements */
        .dataframe {
            font-size: 0.88rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
