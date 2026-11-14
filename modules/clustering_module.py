"""
Módulo de Machine Learning: CLUSTERIZAÇÃO (AGRUPAMENTO NÃO SUPERVISIONADO)
Foco em Marketing Digital: Segmentação Comportamental, Matriz RFM e Personas de Campanhas/Clientes.
Segue estritamente as diretrizes de 'ml-best-practices':
- Padronização rigorosa com StandardScaler
- Cálculo do Método do Cotovelo (Inertia/Elbow) e Curva do Coeficiente de Silhueta para k em [2, 10]
- Redução de dimensionalidade com PCA 2D e 3D interativo
- Gráfico Radar / Barras de Centróides para caracterização de personas
- Download dos dados classificados com rótulos de cluster
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

from utils.metrics import format_number

def render_clustering_tab(df):
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span class="badge badge-clu">Não Supervisionado</span>
            <span style="color: #94A3B8; font-size: 0.9rem;">Segmentação de Audiências & Personas</span>
        </div>
        <h2>🔮 Clusterização: Segmentação Inteligente de Clientes & Campanhas</h2>
        <p style="color: #CBD5E1;">
            Descubra padrões ocultos de comportamento e crie clusters de clientes (VIP, Em Risco, Comprador Ocasionais) 
            ou grupos de campanhas para personalização de mensagens e otimização de verba.
        </p>
    """, unsafe_allow_html=True)

    if df is None or df.empty:
        st.warning("Carregue uma base de dados para iniciar a clusterização.")
        return

    # 1. Seleção de Variáveis
    st.markdown("### 1. Seleção de Variáveis para Agrupamento")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(num_cols) < 2:
        st.error("São necessárias pelo menos duas variáveis numéricas para realizar a clusterização.")
        return

    # Excluir potenciais IDs numéricos
    clean_num_cols = [c for c in num_cols if not any(id_word in c.lower() for id_word in ['_id', 'id_', 'cod_'])]

    selected_features = st.multiselect(
        "Selecione as variáveis numéricas para formar os clusters:",
        options=clean_num_cols,
        default=clean_num_cols if len(clean_num_cols) <= 6 else clean_num_cols[:5],
        help="Exemplos: Recência, Frequência de Compras, Valor Gasto, CTR, Taxa de Abandono."
    )

    if len(selected_features) < 2:
        st.warning("Selecione pelo menos duas variáveis para calcular os agrupamentos.")
        return

    # 2. Análise de Otimização de K (Cotovelo & Silhueta)
    st.markdown("### 2. Otimização do Número de Clusters (K-Means)")
    
    with st.spinner("Calculando inércia e coeficiente de silhueta para k de 2 a 8..."):
        eval_results = evaluate_clusters_range(df[selected_features], max_k=8)

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=eval_results['k_values'],
            y=eval_results['inertias'],
            mode='lines+markers',
            marker=dict(color='#6366F1', size=9),
            line=dict(color='#6366F1', width=3),
            name='Inércia (WCSS)'
        ))
        fig_elbow.update_layout(
            title="Método do Cotovelo (Elbow Method)",
            xaxis_title="Número de Clusters (k)",
            yaxis_title="Inércia Intra-Cluster",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_e2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=eval_results['k_values'],
            y=eval_results['silhouette_scores'],
            mode='lines+markers',
            marker=dict(color='#F59E0B', size=9),
            line=dict(color='#F59E0B', width=3),
            name='Silhouette Score'
        ))
        fig_sil.update_layout(
            title="Coeficiente de Silhueta Médio por k",
            xaxis_title="Número de Clusters (k)",
            yaxis_title="Silhouette Score (-1 a +1)",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    suggested_k = eval_results['best_k']
    st.info(f"💡 **Sugestão Algorítmica**: O k com melhor coeficiente de silhueta é **k = {suggested_k}**.")

    # 3. Execução do K-Means com k Escolhido
    col_k_sel, col_k_seed = st.columns([1, 1])
    with col_k_sel:
        chosen_k = st.slider("Escolha o número de clusters final (k):", min_value=2, max_value=8, value=suggested_k, step=1)
    with col_k_seed:
        random_state = st.number_input("Random Seed:", value=42, step=1, key="clu_seed")

    # Treinamento
    clu_results = run_kmeans_clustering(df, selected_features, k=chosen_k, random_state=random_state)
    
    st.markdown("---")
    st.markdown(f"### 3. Visualização dos Clusters (k = {chosen_k})")

    # Cards de Métricas
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Clusters Formados</div>
                <div class="metric-value">{chosen_k} Grupos</div>
                <div class="metric-sub">Segmentação Completa</div>
            </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Silhouette Score</div>
                <div class="metric-value">{clu_results['silhouette']:.3f}</div>
                <div class="metric-sub">Qualidade da separação</div>
            </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Davies-Bouldin Index</div>
                <div class="metric-value">{clu_results['davies_bouldin']:.3f}</div>
                <div class="metric-sub">Menor valor = melhor separação</div>
            </div>
        """, unsafe_allow_html=True)

    # 4. Gráficos PCA 2D e 3D
    tab_pca2d, tab_pca3d, tab_pair = st.tabs(["📊 Projeção PCA 2D", "🌐 Projeção PCA 3D", "🔍 Pares de Variáveis"])
    
    with tab_pca2d:
        fig_pca2d = px.scatter(
            clu_results['pca_df'],
            x='PCA1',
            y='PCA2',
            color='Cluster',
            hover_data=selected_features[:3],
            title=f"Projeção 2D dos Clusters (PCA: {clu_results['pca_var_explained']:.1f}% da variância explicada)",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pca2d.update_layout(margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_pca2d, use_container_width=True)

    with tab_pca3d:
        fig_pca3d = px.scatter_3d(
            clu_results['pca_df'],
            x='PCA1',
            y='PCA2',
            z='PCA3',
            color='Cluster',
            hover_data=selected_features[:2],
            title="Projeção 3D dos Clusters via PCA",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pca3d.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_pca3d, use_container_width=True)

    with tab_pair:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            var_x = st.selectbox("Eixo X:", options=selected_features, index=0)
        with col_p2:
            var_y = st.selectbox("Eixo Y:", options=selected_features, index=min(1, len(selected_features)-1))

        fig_pair = px.scatter(
            clu_results['df_with_clusters'],
            x=var_x,
            y=var_y,
            color='Cluster',
            title=f"{var_y} vs {var_x} por Cluster",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_pair, use_container_width=True)

    # 5. Perfis dos Clusters & Personas
    st.markdown("### 4. Perfis dos Clusters & Playbook de Marketing")
    cluster_means = clu_results['cluster_means']
    
    st.markdown("#### Médias Reais das Métricas por Cluster:")
    st.dataframe(cluster_means.style.format("{:,.2f}"), use_container_width=True)

    # Distribuição de tamanho dos clusters
    col_pie, col_radar = st.columns([1, 1])
    with col_pie:
        counts_df = clu_results['df_with_clusters']['Cluster'].value_counts().reset_index()
        counts_df.columns = ['Cluster', 'Qtd_Registros']
        fig_pie = px.pie(
            counts_df,
            values='Qtd_Registros',
            names='Cluster',
            title="Distribuição do Tamanho dos Clusters",
            template="plotly_dark",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_radar:
        # Gráfico de Barras Normalizado das Médias
        scaled_means = clu_results['scaled_means']
        fig_bars = px.bar(
            scaled_means.T.reset_index(),
            x='index',
            y=[f"Cluster {i}" for i in range(chosen_k)],
            barmode='group',
            title="Desvio Padrão Relativo por Métrica (Centroides Z-Score)",
            template="plotly_dark",
            labels={'index': 'Métrica', 'value': 'Z-Score'}
        )
        st.plotly_chart(fig_bars, use_container_width=True)

    # 6. Playbook Estratégico de Marketing
    st.markdown("### 5. 🎯 Ações Táticas Recomendadas para Cada Segmento")
    cols_play = st.columns(min(chosen_k, 4))
    
    # Gerar interpretação automática
    personas = generate_personas_playbook(cluster_means, selected_features)
    for idx, (c_name, p_info) in enumerate(personas.items()):
        if idx < len(cols_play):
            with cols_play[idx]:
                st.markdown(f"""
                    <div class="metric-card" style="border-top: 3px solid #6366F1;">
                        <h4 style="color: #A5B4FC; margin-bottom: 5px;">{c_name}</h4>
                        <p style="font-size: 0.82rem; color: #E2E8F0; font-weight: 600;">{p_info['titulo']}</p>
                        <p style="font-size: 0.78rem; color: #94A3B8;"><b>Tamanho:</b> {p_info['tamanho']} registros</p>
                        <hr style="border-color: rgba(255,255,255,0.08); margin: 8px 0;">
                        <p style="font-size: 0.78rem; color: #CBD5E1;"><b>Estratégia Recomendada:</b><br>{p_info['estrategia']}</p>
                    </div>
                """, unsafe_allow_html=True)

    # 7. Download do Dataset com Rótulos
    st.markdown("---")
    st.markdown("### 6. 📥 Exportar Resultados Segmentados")
    csv_bytes = clu_results['df_with_clusters'].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Baixar Tabela Completa com Clusters Atribuídos (.csv)",
        data=csv_bytes,
        file_name="marketing_clusters_resultado.csv",
        mime="text/csv",
        type="primary"
    )

def evaluate_clusters_range(X_features, max_k=8):
    """Calcula métricas de inércia e silhueta para k variando de 2 até max_k."""
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_imputed = imputer.fit_transform(X_features)
    X_scaled = scaler.fit_transform(X_imputed)

    k_values = list(range(2, min(max_k + 1, len(X_features) - 1)))
    inertias = []
    silhouette_scores = []

    best_k = 2
    best_sil = -1.0

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        
        inertias.append(km.inertia_)
        sil = silhouette_score(X_scaled, labels)
        silhouette_scores.append(sil)

        if sil > best_sil:
            best_sil = sil
            best_k = k

    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "best_k": best_k
    }

def run_kmeans_clustering(df, feature_cols, k=3, random_state=42):
    """Ajusta o KMeans, realiza PCA e calcula os centróides e métricas."""
    X = df[feature_cols].copy()
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=15)
    cluster_labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(X_scaled, cluster_labels)
    db = davies_bouldin_score(X_scaled, cluster_labels)

    # PCA 3D
    pca = PCA(n_components=min(3, len(feature_cols)))
    pca_comps = pca.fit_transform(X_scaled)

    df_out = df.copy()
    df_out['Cluster'] = [f"Cluster {c}" for c in cluster_labels]
    
    pca_df = pd.DataFrame(pca_comps, columns=[f"PCA{i+1}" for i in range(pca_comps.shape[1])])
    pca_df['Cluster'] = df_out['Cluster']
    for col in feature_cols:
        pca_df[col] = df[col].values

    # Médias Reais por Cluster
    means = df_out.groupby('Cluster')[feature_cols].mean()

    # Médias Z-score (centróides no espaço escalado)
    scaled_means = pd.DataFrame(
        kmeans.cluster_centers_,
        index=[f"Cluster {i}" for i in range(k)],
        columns=feature_cols
    )

    var_exp = (pca.explained_variance_ratio_[:2].sum()) * 100

    return {
        "kmeans": kmeans,
        "df_with_clusters": df_out,
        "cluster_means": means,
        "scaled_means": scaled_means,
        "pca_df": pca_df,
        "pca_var_explained": var_exp,
        "silhouette": sil,
        "davies_bouldin": db
    }

def generate_personas_playbook(cluster_means, feature_cols):
    """Gera personas e recomendações táticas automáticas baseadas nas médias dos clusters."""
    personas = {}
    for cluster_name in cluster_means.index:
        row = cluster_means.loc[cluster_name]
        
        # Heurística para descobrir se tem métricas de gasto/frequência
        titulo = "Segmento de Audiência"
        estrategia = "Campanha segmentada com teste A/B de criativos e ofertas personalizadas."

        # Checar se temos colunas RFM ou de mídia
        cols_lower = [c.lower() for c in feature_cols]
        if any('gasto' in c or 'valor' in c or 'receita' in c for c in cols_lower):
            # Identificar se é o cluster com maior gasto
            spend_col = [c for c in feature_cols if any(k in c.lower() for k in ['gasto', 'valor', 'receita'])][0]
            max_spend_cluster = cluster_means[spend_col].idxmax()
            min_spend_cluster = cluster_means[spend_col].idxmin()

            if cluster_name == max_spend_cluster:
                titulo = "💎 Clientes VIP / High-Value"
                estrategia = "Atendimento exclusivo, programa de fidelidade premium, pré-lançamentos e benefícios antecipados."
            elif cluster_name == min_spend_cluster:
                titulo = "🏷️ Oportunistas / Sensíveis a Preço"
                estrategia = "Cupons promocionais com tempo limitado, frete grátis condicional e produtos de ticket de entrada."
            else:
                titulo = "🔄 Recorrentes em Desenvolvimento"
                estrategia = "Campanhas de Cross-sell e Up-sell sugerindo produtos complementares ao histórico de compras."
        elif any('cpc' in c or 'ctr' in c or 'investimento' in c for c in cols_lower):
            titulo = "🎯 Grupo de Campanhas Específicas"
            estrategia = "Otimizar lances de CPC, pausar criativos de baixa conversão e realocar verba para os canais de alto ROAS."

        personas[cluster_name] = {
            "titulo": titulo,
            "tamanho": "-",
            "estrategia": estrategia
        }

    return personas
