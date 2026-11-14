"""
Utilitários de Métricas e KPIs de Marketing Digital.
Funções para cálculo, formatação e verificação de sanidade de dados de marketing.
"""

import pandas as pd
import numpy as np

def format_currency_brl(val):
    """Formata valor em Real Brasileiro (R$)."""
    if pd.isna(val):
        return "R$ 0,00"
    return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def format_percentage(val):
    """Formata valor percentual."""
    if pd.isna(val):
        return "0,00%"
    return f"{val:.2f}%".replace(".", ",")

def format_number(val):
    """Formata número inteiro ou flutuante com separador de milhar."""
    if pd.isna(val):
        return "0"
    if isinstance(val, (int, np.integer)):
        return f"{val:,}".replace(",", ".")
    return f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def calculate_marketing_kpis(df):
    """
    Calcula ou extrai os principais KPIs de Marketing Digital caso as colunas existam:
    - CTR (Click-Through Rate)
    - CPC (Cost Per Click)
    - CPA (Cost Per Acquisition)
    - ROAS (Return On Ad Spend)
    - Taxa de Conversão
    """
    kpis = {}
    cols_lower = {str(c).lower().strip(): c for c in df.columns}

    # Investimento / Spend
    spend_col = None
    for k in ['investimento_brl', 'investimento', 'spend', 'custo', 'cost', 'valor_gasto']:
        if k in cols_lower:
            spend_col = cols_lower[k]
            break
            
    # Receita / Revenue
    rev_col = None
    for k in ['receita_brl', 'receita_total_brl', 'receita', 'revenue', 'faturamento', 'valor_total_gasto_brl']:
        if k in cols_lower:
            rev_col = cols_lower[k]
            break

    # Cliques
    clicks_col = None
    for k in ['cliques', 'clicks']:
        if k in cols_lower:
            clicks_col = cols_lower[k]
            break

    # Impressões
    imp_col = None
    for k in ['impressoes', 'impressões', 'impressions']:
        if k in cols_lower:
            imp_col = cols_lower[k]
            break

    # Conversões
    conv_col = None
    for k in ['conversoes', 'conversões', 'conversions', 'vendas']:
        if k in cols_lower:
            conv_col = cols_lower[k]
            break

    total_spend = df[spend_col].sum() if spend_col else 0
    total_rev = df[rev_col].sum() if rev_col else 0
    total_clicks = df[clicks_col].sum() if clicks_col else 0
    total_imp = df[imp_col].sum() if imp_col else 0
    total_conv = df[conv_col].sum() if conv_col else 0

    kpis['total_investimento'] = total_spend
    kpis['total_receita'] = total_rev
    kpis['total_cliques'] = total_clicks
    kpis['total_impressoes'] = total_imp
    kpis['total_conversoes'] = total_conv

    # CTR Global
    if total_imp > 0 and total_clicks > 0:
        kpis['ctr_global'] = (total_clicks / total_imp) * 100
    else:
        kpis['ctr_global'] = None

    # CPC Global
    if total_clicks > 0 and total_spend > 0:
        kpis['cpc_global'] = total_spend / total_clicks
    else:
        kpis['cpc_global'] = None

    # CPA Global
    if total_conv > 0 and total_spend > 0:
        kpis['cpa_global'] = total_spend / total_conv
    else:
        kpis['cpa_global'] = None

    # ROAS Global
    if total_spend > 0 and total_rev > 0:
        kpis['roas_global'] = total_rev / total_spend
    else:
        kpis['roas_global'] = None

    # Taxa de Conversão Global
    if total_clicks > 0 and total_conv > 0:
        kpis['taxa_conversao_global'] = (total_conv / total_clicks) * 100
    else:
        kpis['taxa_conversao_global'] = None

    return kpis
