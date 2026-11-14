"""
Módulo de Análise Exploratória de Dados (EDA) para Marketing Digital.
Visualizações interativas, métricas de resumo, correlações e detecção de anomalias/outliers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.metrics import calculate_marketing_kpis, format_currency_brl, format_percentage, format_number

def render_eda_tab(df):
    st.markdown("## 📊 Análise Exploratória de Dados (EDA)")
    st.markdown("Compreenda a distribuição, integridade e correlações das suas métricas de marketing antes de modelar.")

    if df is None or df.empty:
        st.warning("Nenhum dado carregado. Faça o upload de um CSV ou selecione um dataset de exemplo.")
        return

    # 1. KPIs Globais de Marketing (se disponíveis)
    kpis = calculate_marketing_kpis(df)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Linhas / Registros</div>
                <div class="metric-value">{format_number(len(df))}</div>
                <div class="metric-sub">{len(df.columns)} colunas</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        val_spend = format_currency_brl(kpis['total_investimento']) if kpis['total_investimento'] > 0 else "-"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Investimento Total</div>
                <div class="metric-value">{val_spend}</div>
                <div class="metric-sub">Orçamento de mídia</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        val_rev = format_currency_brl(kpis['total_receita']) if kpis['total_receita'] > 0 else "-"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Receita Estimada</div>
                <div class="metric-value">{val_rev}</div>
                <div class="metric-sub">Retorno gerado</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        val_roas = f"{kpis['roas_global']:.2f}x" if kpis['roas_global'] is not None else "-"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ROAS Médio</div>
                <div class="metric-value">{val_roas}</div>
                <div class="metric-sub">Receita / Gasto</div>
            </div>
        """, unsafe_allow_html=True)
    with col5:
        val_ctr = format_percentage(kpis['ctr_global']) if kpis['ctr_global'] is not None else "-"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">CTR Médio</div>
                <div class="metric-value">{val_ctr}</div>
                <div class="metric-sub">Cliques / Impressões</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 2. Amostra e Inspeção dos Dados
    st.markdown("### 📋 Visualização da Tabela de Dados")
    n_rows = st.slider("Quantidade de linhas a exibir:", min_value=5, max_value=min(100, len(df)), value=10, step=5)
    st.dataframe(df.head(n_rows), use_container_width=True)

    # 3. Diagnóstico de Colunas e Valores Nulos
    with st.expander("🔍 Diagnóstico Detalhado de Colunas e Valores Nulos"):
        diag_df = pd.DataFrame({
            "Coluna": df.columns,
            "Tipo": [str(t) for t in df.dtypes],
            "Preenchidos": df.notnull().sum().values,
            "Nulos (Qtd)": df.isnull().sum().values,
            "Nulos (%)": [(df[col].isnull().sum() / len(df) * 100) for col in df.columns],
            "Valores Únicos": [df[col].nunique() for col in df.columns]
        })
        diag_df["Nulos (%)"] = diag_df["Nulos (%)"].map(lambda x: f"{x:.1f}%")
        st.dataframe(diag_df, use_container_width=True)

    # 4. Estatísticas Descritivas
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if num_cols:
        st.markdown("### 📈 Estatísticas Descritivas (Numéricas)")
        desc_df = df[num_cols].describe().T
        st.dataframe(desc_df.style.format("{:,.2f}"), use_container_width=True)

    # 5. Visualizações Gráficas: Correlação e Dispersão
    st.markdown("### 🔬 Relações e Correlações Entre Métricas")
    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.markdown("#### Matriz de Correlação Linear (Pearson)")
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Viridis",
                title="Correlações entre Variáveis Numéricas"
            )
            fig_corr.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("São necessárias pelo menos duas variáveis numéricas para exibir a matriz de correlação.")

    with col_chart2:
        st.markdown("#### Gráfico de Dispersão Customizado (Scatter Plot)")
        if len(num_cols) >= 2:
            x_default = "Investimento_BRL" if "Investimento_BRL" in num_cols else num_cols[0]
            y_default = "Receita_BRL" if "Receita_BRL" in num_cols else (num_cols[1] if len(num_cols) > 1 else num_cols[0])
            
            x_var = st.selectbox("Eixo X:", options=num_cols, index=num_cols.index(x_default) if x_default in num_cols else 0)
            y_var = st.selectbox("Eixo Y:", options=num_cols, index=num_cols.index(y_default) if y_default in num_cols else 1)
            
            color_opts = ["Nenhum"] + cat_cols + num_cols
            color_var = st.selectbox("Colorir por (Dimensão/Categoria):", options=color_opts, index=1 if cat_cols else 0)

            fig_scatter = px.scatter(
                df,
                x=x_var,
                y=y_var,
                color=None if color_var == "Nenhum" else color_var,
                trendline="ols",
                title=f"{y_var} vs {x_var}",
                template="plotly_dark"
            )
            fig_scatter.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Colunas numéricas insuficientes para gerar dispersão.")

    # 6. Distribuições e Boxplots (Detecção de Outliers)
    st.markdown("### 📊 Distribuição de Métricas e Detecção de Outliers")
    if num_cols:
        col_dist1, col_dist2 = st.columns([1, 1])
        with col_dist1:
            dist_feat = st.selectbox("Selecione a métrica para histograma:", options=num_cols, key="dist_feat")
            fig_hist = px.histogram(
                df,
                x=dist_feat,
                nbins=30,
                marginal="box",
                title=f"Distribuição & Boxplot: {dist_feat}",
                template="plotly_dark",
                color_discrete_sequence=["#6366F1"]
            )
            fig_hist.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_dist2:
            if cat_cols:
                cat_feat = st.selectbox("Selecione uma variável categórica:", options=cat_cols, key="cat_feat")
                cat_counts = df[cat_feat].value_counts().reset_index()
                cat_counts.columns = [cat_feat, 'Contagem']
                fig_bar = px.bar(
                    cat_counts,
                    x=cat_feat,
                    y='Contagem',
                    color='Contagem',
                    title=f"Frequência por Categoria: {cat_feat}",
                    template="plotly_dark",
                    color_continuous_scale="Plasma"
                )
                fig_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Nenhuma variável categórica encontrada para gráfico de frequência.")

    # 7. Storytelling & Insights de Negócio
    st.markdown("""
        <div class="insight-box">
            <h4>💡 Insights de Marketing & Boas Práticas</h4>
            <p>
                A análise exploratória é crucial para identificar <b>assimetrias</b> em métricas como CPA e ROAS, 
                detectar <b>outliers</b> (como campanhas com investimento excessivo sem conversão correspondente) e 
                verificar se variáveis como Cliques e Investimento possuem alta colinearidade com a Receita esperada.
            </p>
        </div>
    """, unsafe_allow_html=True)
