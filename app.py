"""
Aplicação Principal: Marketing AI Lab
Plataforma de Inteligência Artificial e Machine Learning aplicada a Marketing Digital.
Suporta:
1. Regressão (Previsão de Receita, ROAS e Conversões)
2. Classificação (Propensão de Conversão de Leads e Qualificação)
3. Clusterização (Segmentação Comportamental RFM de Clientes e Campanhas)
"""

import streamlit as st
import pandas as pd
import numpy as np

# Configuração da página Streamlit (deve ser o primeiro comando Streamlit)
st.set_page_config(
    page_title="Marketing AI Lab | Machine Learning para Marketing Digital",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importações dos módulos da aplicação
from modules.styles import apply_custom_styles
from modules.data_loader import parse_uploaded_csv, load_sample_dataset, SAMPLE_DATASETS, get_csv_download_link
from modules.eda import render_eda_tab
from modules.regression_module import render_regression_tab
from modules.classification_module import render_classification_tab
from modules.clustering_module import render_clustering_tab

# Aplicar estilos CSS personalizados
apply_custom_styles()

def main():
    # ------------------------------------------------------------------
    # BARRA LATERAL (SIDEBAR): Configuração de Dados & Navegação
    # ------------------------------------------------------------------
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 1.2rem;">
                <h2 style="color: #818CF8; margin: 0; font-size: 1.5rem; font-weight: 800;">🚀 Marketing AI Lab</h2>
                <p style="color: #94A3B8; font-size: 0.8rem; margin: 0;">Plataforma Integrada de ML</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📁 Fonte de Dados")
        
        fonte_dados = st.radio(
            "Selecione a origem da tabela:",
            options=[
                "Upload de Arquivo CSV",
                "Amostra 1: Regressão (Receita & ROAS)",
                "Amostra 2: Classificação (Leads)",
                "Amostra 3: Clusterização (Clientes)",
                "Amostra 4: Base Completa (Unificada)"
            ],
            index=1,
            help="Faça o upload do seu próprio arquivo CSV ou selecione um dos conjuntos prontos para demonstração imediata."
        )

        df = None

        if fonte_dados == "Upload de Arquivo CSV":
            uploaded_file = st.file_uploader(
                "Envie seu arquivo CSV:",
                type=["csv", "txt"],
                help="Selecione um arquivo CSV com cabeçalho contendo suas métricas de marketing."
            )
            
            with st.expander("🛠️ Opções Avançadas de Leitura CSV"):
                sep_choice = st.selectbox("Delimitador:", options=["Automático", ",", ";", "\\t", "|"], index=0)
                dec_choice = st.selectbox("Separador Decimal:", options=["Automático", ".", ","], index=0)
                enc_choice = st.selectbox("Codificação (Encoding):", options=["Automático", "utf-8", "latin-1", "cp1252"], index=0)

            if uploaded_file is not None:
                sep = None if sep_choice == "Automático" else (r'\t' if sep_choice == "\\t" else sep_choice)
                dec = None if dec_choice == "Automático" else dec_choice
                enc = None if enc_choice == "Automático" else enc_choice
                try:
                    df = parse_uploaded_csv(uploaded_file, sep=sep, decimal=dec, encoding=enc)
                    st.success(f"CSV carregado: {df.shape[0]} linhas x {df.shape[1]} colunas.")
                except Exception as e:
                    st.error(f"Erro ao processar CSV: {e}")
            else:
                st.info("Aguardando upload do arquivo CSV do cliente.")
        elif fonte_dados == "Amostra 1: Regressão (Receita & ROAS)":
            df = load_sample_dataset("regressao")
        elif fonte_dados == "Amostra 2: Classificação (Leads)":
            df = load_sample_dataset("classificacao")
        elif fonte_dados == "Amostra 3: Clusterização (Clientes)":
            df = load_sample_dataset("clusterizacao")
        elif fonte_dados == "Amostra 4: Base Completa (Unificada)":
            df = load_sample_dataset("completo")

        # Se houver dados carregados, exibir resumo rápido
        if df is not None and not df.empty:
            st.markdown("---")
            st.markdown(f"""
                <div style="background: rgba(99, 102, 241, 0.12); padding: 10px 14px; border-radius: 8px; border: 1px solid rgba(99, 102, 241, 0.3);">
                    <span style="color: #A5B4FC; font-size: 0.8rem; font-weight: 700; text-transform: uppercase;">Base Ativa</span>
                    <div style="font-size: 1.1rem; color: #FFFFFF; font-weight: 700;">{len(df):,} linhas</div>
                    <div style="color: #94A3B8; font-size: 0.8rem;">{len(df.columns)} variáveis mapeadas</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🧭 Menu de Navegação")
        menu_opt = st.radio(
            "Navegue pelos módulos:",
            options=[
                "🏠 Visão Geral & Início",
                "📊 Análise Exploratória (EDA)",
                "📈 Regressão: Previsão de Receita e ROAS",
                "🎯 Classificação: Propensão de Conversão",
                "🔮 Clusterização: Segmentação de Clientes",
                "📖 Documentação & Glossário de Métricas"
            ],
            index=0
        )

        st.markdown("---")
        st.markdown("""
            <div style="font-size: 0.75rem; color: #64748B; text-align: center;">
                Desenvolvido com Python & Streamlit<br>
                Pronto para Deploy no GitHub
            </div>
        """, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # CONTEÚDO PRINCIPAL (MAIN CONTENT)
    # ------------------------------------------------------------------

    # Banner Superior de Boas-Vindas
    st.markdown("""
        <div class="hero-banner">
            <h1>Marketing AI Lab</h1>
            <p>
                Plataforma analítica e preditiva desenvolvida para potencializar resultados de <b>Marketing Digital</b> 
                através dos três paradigmas fundamentais de Machine Learning: <b>Regressão</b>, <b>Classificação</b> e <b>Clusterização</b>.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Roteamento de telas
    if menu_opt == "🏠 Visão Geral & Início":
        render_home_tab(df)
    elif menu_opt == "📊 Análise Exploratória (EDA)":
        render_eda_tab(df)
    elif menu_opt == "📈 Regressão: Previsão de Receita e ROAS":
        render_regression_tab(df)
    elif menu_opt == "🎯 Classificação: Propensão de Conversão":
        render_classification_tab(df)
    elif menu_opt == "🔮 Clusterização: Segmentação de Clientes":
        render_clustering_tab(df)
    elif menu_opt == "📖 Documentação & Glossário de Métricas":
        render_docs_tab()

def render_home_tab(df):
    """Tela inicial com apresentação dos 3 modelos, templates para download e prévia da base."""
    st.markdown("### 🎯 Como Funciona Esta Aplicação?")
    st.markdown("""
        Esta solução foi estruturada para atender tomadores de decisão, analistas de tráfego pago, 
        gestores de CRM e cientistas de dados em suas análises de campanhas digitais.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="metric-card" style="border-top: 4px solid #3B82F6;">
                <div class="badge badge-reg" style="margin-bottom: 8px;">1. Regressão</div>
                <h3 style="color: #93C5FD; margin-top: 0;">Previsão de Receita & ROAS</h3>
                <p style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.5;">
                    Estime o faturamento e o retorno financeiro com base no investimento de mídia, 
                    volume de cliques e canais publicitários. Inclui simulador de orçamento <i>What-If</i>.
                </p>
                <ul style="color: #94A3B8; font-size: 0.82rem; padding-left: 18px;">
                    <li>Linear Regression & Ridge</li>
                    <li>Random Forest Regressor</li>
                    <li>Gradient Boosting Regressor</li>
                    <li>Métricas: R², MAE, RMSE, MAPE</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="metric-card" style="border-top: 4px solid #10B981;">
                <div class="badge badge-clf" style="margin-bottom: 8px;">2. Classificação</div>
                <h3 style="color: #6EE7B7; margin-top: 0;">Propensão de Conversão</h3>
                <p style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.5;">
                    Preveja se um lead ou visitante irá converter com base em sinais de intenção de compra e histórico de navegação. 
                    Inclui calculadora de Lead Scoring (Quente/Morno/Frio).
                </p>
                <ul style="color: #94A3B8; font-size: 0.82rem; padding-left: 18px;">
                    <li>Logistic Regression</li>
                    <li>Random Forest Classifier</li>
                    <li>Gradient Boosting Classifier</li>
                    <li>Métricas: Acurácia, F1, ROC-AUC</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="metric-card" style="border-top: 4px solid #F59E0B;">
                <div class="badge badge-clu" style="margin-bottom: 8px;">3. Clusterização</div>
                <h3 style="color: #FCD34D; margin-top: 0;">Segmentação & Personas</h3>
                <p style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.5;">
                    Descubra grupos naturais de comportamento (VIPs, Em Risco, Oportunistas) utilizando 
                    técnicas de agrupamento não supervisionado e redução de dimensionalidade PCA.
                </p>
                <ul style="color: #94A3B8; font-size: 0.82rem; padding-left: 18px;">
                    <li>K-Means Clustering</li>
                    <li>Método do Cotovelo & Silhueta</li>
                    <li>Visualização PCA 2D e 3D</li>
                    <li>Playbook tático de ativação</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Seção de Download de Templates CSV
    st.markdown("### 📥 Modelos de Planilhas CSV para Testar ou Fazer Upload")
    st.markdown("Você pode baixar os templates prontos abaixo para preencher com seus dados reais ou testar imediatamente:")

    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        reg_sample = load_sample_dataset("regressao")
        if reg_sample is not None:
            st.download_button(
                "⬇️ Baixar Template Regressão (CSV)",
                data=get_csv_download_link(reg_sample, "template_regressao_marketing.csv"),
                file_name="template_regressao_marketing.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    with col_d2:
        clf_sample = load_sample_dataset("classificacao")
        if clf_sample is not None:
            st.download_button(
                "⬇️ Baixar Template Classificação (CSV)",
                data=get_csv_download_link(clf_sample, "template_classificacao_leads.csv"),
                file_name="template_classificacao_leads.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    with col_d3:
        clu_sample = load_sample_dataset("clusterizacao")
        if clu_sample is not None:
            st.download_button(
                "⬇️ Baixar Template Clusterização (CSV)",
                data=get_csv_download_link(clu_sample, "template_clusterizacao_clientes.csv"),
                file_name="template_clusterizacao_clientes.csv",
                mime="text/csv",
                use_container_width=True
            )

    st.markdown("---")

    # Prévia da Base Atual Carregada
    st.markdown("### 📋 Prévia da Base de Dados Carregada no Momento")
    if df is not None and not df.empty:
        st.write(f"Exibindo as primeiras linhas do conjunto ativo ({len(df)} registros totais):")
        st.dataframe(df.head(8), use_container_width=True)
    else:
        st.warning("Nenhuma base carregada. Selecione um dataset de exemplo na barra lateral ou faça o upload de um CSV.")

def render_docs_tab():
    """Documentação técnica e glossário de métricas de marketing digital."""
    st.markdown("## 📖 Documentação & Glossário de Marketing Digital")
    st.markdown("""
        Esta seção detalha os conceitos de negócios, fórmulas matemáticas e os algoritmos empregados na plataforma.
    """)

    with st.expander("📊 Glossário de Métricas de Performance de Marketing", expanded=True):
        st.markdown("""
        - **Investimento (Spend):** Valor total em moeda investido na veiculação de anúncios (Google Ads, Meta Ads, TikTok, etc.).
        - **Impressões (Impressions):** Quantidade de vezes em que o anúncio foi exibido na tela do usuário.
        - **Cliques (Clicks):** Número de interações onde o usuário clicou no anúncio e foi direcionado para a página de destino.
        - **CTR (Click-Through Rate):** Taxa de cliques sobre as impressões:
          $$\\text{CTR (\\%)} = \\left( \\frac{\\text{Cliques}}{\\text{Impressões}} \\right) \\times 100$$
        - **CPC (Cost Per Click):** Custo médio pago por clique gerado:
          $$\\text{CPC} = \\frac{\\text{Investimento Total}}{\\text{Total de Cliques}}$$
        - **CPA (Cost Per Acquisition):** Custo médio para adquirir uma conversão ou venda:
          $$\\text{CPA} = \\frac{\\text{Investimento Total}}{\\text{Total de Conversões}}$$
        - **ROAS (Return On Ad Spend):** Retorno financeiro obtido para cada R$ 1,00 investido em publicidade:
          $$\\text{ROAS} = \\frac{\\text{Receita Gerada}}{\\text{Investimento em Mídia}}$$
        - **Taxa de Conversão (CR):** Percentual de visitantes que realizaram a ação desejada:
          $$\\text{Taxa de Conversão (\\%)} = \\left( \\frac{\\text{Conversões}}{\\text{Cliques / Sessões}} \\right) \\times 100$$
        """)

    with st.expander("🤖 Boas Práticas de Machine Learning Implementadas"):
        st.markdown("""
        1. **Strict Featurization Ordering (Sem Vazamento de Dados):**
           - A divisão entre conjuntos de treino e teste é realizada estritamente antes do cálculo de médias, medianas, encoders ou escalonamento.
        2. **Tratamento de Dados Ausentes:**
           - Variáveis numéricas utilizam imputação por mediana (robusta a outliers).
           - Variáveis categóricas utilizam imputação pela moda (valor mais frequente).
        3. **Codificação Categórica e Escalonamento:**
           - `OneHotEncoder(handle_unknown='ignore')` para dados categóricos nominais.
           - `StandardScaler()` para variáveis numéricas contínuas.
        4. **Benchmark Competitivo:**
           - Comparação simultânea de modelos lineares e ensembles baseados em árvores (Random Forest e Gradient Boosting).
        5. **Avaliação Holística:**
           - Regressão: Avaliação por $R^2$, MAE, RMSE e análise de resíduos.
           - Classificação: Avaliação por Acurácia, F1-Score, Precisão, Recall e Curva ROC-AUC.
           - Clusterização: Validação por Inércia (Cotovelo), Coeficiente de Silhueta e Davies-Bouldin.
        """)

if __name__ == "__main__":
    main()
