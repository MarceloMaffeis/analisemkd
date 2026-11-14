"""
Módulo de Machine Learning: REGRESSÃO
Foco em Marketing Digital: Previsão de Receita (Revenue), ROAS e Conversões.
Segue estritamente as diretrizes de 'ml-best-practices':
- Divisão Treino/Teste ANTES do pré-processamento (Strict Featurization Ordering)
- Tratamento cuidadoso de dados faltantes (Imputação de mediana/moda)
- Codificação One-Hot para categóricos e Padronização StandardScaler para numéricos
- Treinamento e benchmark de múltiplos modelos (Linear, Ridge, Random Forest, Gradient Boosting)
- Gráficos de Resíduos, Real vs Previsto e Importância de Features
- Simulador Interativo de Orçamento 'What-If'
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from utils.metrics import format_currency_brl, format_number

def render_regression_tab(df):
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span class="badge badge-reg">Supervisionado</span>
            <span style="color: #94A3B8; font-size: 0.9rem;">Modelagem Contínua</span>
        </div>
        <h2>📈 Regressão: Previsão de Receita & ROAS de Campanhas</h2>
        <p style="color: #CBD5E1;">
            Estime o retorno financeiro ou volume de conversões com base em investimentos de mídia,
            canais, impressões e taxas de clique (CTR/CPC).
        </p>
    """, unsafe_allow_html=True)

    if df is None or df.empty:
        st.warning("Carregue uma base de dados para iniciar a modelagem.")
        return

    # 1. Seleção de Variáveis
    st.markdown("### 1. Configuração do Modelo & Variáveis")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    if len(num_cols) < 1:
        st.error("O dataset precisa de pelo menos uma coluna numérica para atuar como variável alvo.")
        return

    # Sugestão de target padrão
    default_targets = ['Receita_BRL', 'Receita_Total_BRL', 'receita', 'revenue', 'ROAS', 'roas', 'Conversoes', 'conversoes']
    target_default_idx = 0
    for dt in default_targets:
        if dt in num_cols:
            target_default_idx = num_cols.index(dt)
            break

    col_cfg1, col_cfg2 = st.columns([1, 2])
    with col_cfg1:
        target_col = st.selectbox(
            "🎯 Variável Alvo Contínua (Target Y):",
            options=num_cols,
            index=target_default_idx,
            help="A métrica que desejamos prever (ex: Receita gerada, ROAS, Conversões)."
        )

    # Identificar potenciais colunas de ID para desmarcar por padrão
    potential_features = [c for c in all_cols if c != target_col]
    default_features = [
        c for c in potential_features
        if not any(id_word in c.lower() for id_word in ['_id', 'id_', 'campanha_id', 'id_registro', 'cod_'])
    ]

    with col_cfg2:
        feature_cols = st.multiselect(
            "🧩 Variáveis Explicativas (Features X):",
            options=potential_features,
            default=default_features if default_features else potential_features[:5],
            help="Métricas e atributos usados para prever a meta (ex: Investimento, Canal, Cliques, CTR)."
        )

    if not feature_cols:
        st.warning("Selecione pelo menos uma variável preditora (feature).")
        return

    # Parâmetros de treino
    with st.expander("⚙️ Hiperparâmetros e Configurações de Treinamento", expanded=False):
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            test_size = st.slider("Proporção do Teste (%):", min_value=10, max_value=40, value=20, step=5) / 100
        with c_p2:
            rf_estimators = st.slider("Árvores no Random Forest:", min_value=50, max_value=300, value=100, step=50)
        with c_p3:
            random_state = st.number_input("Random Seed (Reprodutibilidade):", value=42, step=1)

    # Botão de Treinamento
    train_button = st.button("🚀 Treinar & Comparar Modelos de Regressão", type="primary", use_container_width=True)

    if "regression_results" not in st.session_state or train_button:
        if train_button or "regression_results" not in st.session_state:
            # Processamento e Modelagem
            with st.spinner("Dividindo dados, aplicando encoders e treinando algoritmos..."):
                results = train_regression_models(
                    df=df,
                    target_col=target_col,
                    feature_cols=feature_cols,
                    test_size=test_size,
                    rf_estimators=rf_estimators,
                    random_state=random_state
                )
                st.session_state["regression_results"] = results
                st.session_state["regression_target"] = target_col
                st.session_state["regression_features"] = feature_cols

    if "regression_results" in st.session_state and st.session_state.get("regression_target") == target_col:
        res = st.session_state["regression_results"]
        display_regression_results(res, df, target_col, feature_cols)

def train_regression_models(df, target_col, feature_cols, test_size=0.2, rf_estimators=100, random_state=42):
    """
    Executa a preparação e treinamento dos modelos de regressão estritamente
    obedecendo ao princípio de não vazamento de dados (Strict Featurization Ordering).
    """
    data = df.dropna(subset=[target_col]).copy()
    X = data[feature_cols]
    y = data[target_col]

    # Identificar tipos das features
    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # 1. DIVISÃO TREINO / TESTE ANTES DE QUALQUER AJUSTE DE PIPELINE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 2. Pipelines de pré-processamento
    transformers = []
    if num_features:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_pipeline, num_features))

    if cat_features:
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipeline, cat_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    # 3. Modelos a comparar
    models = {
        "Regressão Linear (Baseline)": LinearRegression(),
        "Ridge Regression (L2)": Ridge(alpha=1.0, random_state=random_state),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=rf_estimators, random_state=random_state, max_depth=12),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=rf_estimators, random_state=random_state, learning_rate=0.08)
    }

    metrics_list = []
    fitted_pipelines = {}
    predictions_test = {}
    predictions_train = {}

    for name, model in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Ajuste no conjunto de treino apenas
        pipe.fit(X_train, y_train)
        
        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)

        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = root_mean_squared_error(y_test, y_pred_test)
        
        # Tratar MAPE para valores próximos de 0
        try:
            non_zero_mask = y_test != 0
            if non_zero_mask.sum() > 0:
                mape_test = mean_absolute_percentage_error(y_test[non_zero_mask], y_pred_test[non_zero_mask]) * 100
            else:
                mape_test = np.nan
        except Exception:
            mape_test = np.nan

        metrics_list.append({
            "Modelo": name,
            "R² (Teste)": r2_test,
            "MAE (Erro Médio Absoluto)": mae_test,
            "RMSE (Raiz do Erro Quadrático)": rmse_test,
            "MAPE (%)": mape_test
        })

        fitted_pipelines[name] = pipe
        predictions_test[name] = y_pred_test
        predictions_train[name] = y_pred_train

    # Identificar o melhor modelo com base no R² do teste
    metrics_df = pd.DataFrame(metrics_list).sort_values(by="R² (Teste)", ascending=False).reset_index(drop=True)
    best_model_name = metrics_df.iloc[0]["Modelo"]
    best_pipeline = fitted_pipelines[best_model_name]

    # Extrair importância de features se disponível
    feature_names = []
    if hasattr(best_pipeline.named_steps['preprocessor'], 'get_feature_names_out'):
        try:
            feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
            feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]
        except Exception:
            feature_names = [f"feat_{i}" for i in range(len(num_features) + len(cat_features))]

    importances = None
    reg_step = best_pipeline.named_steps['regressor']
    if hasattr(reg_step, 'feature_importances_') and len(feature_names) == len(reg_step.feature_importances_):
        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importancia': reg_step.feature_importances_
        }).sort_values(by='Importancia', ascending=False)
    elif hasattr(reg_step, 'coef_') and len(feature_names) == len(reg_step.coef_):
        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importancia': np.abs(reg_step.coef_)
        }).sort_values(by='Importancia', ascending=False)

    return {
        "metrics_df": metrics_df,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "fitted_pipelines": fitted_pipelines,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions_test": predictions_test,
        "feature_importances": importances,
        "num_features": num_features,
        "cat_features": cat_features
    }

def display_regression_results(res, df, target_col, feature_cols):
    """Exibe tabelas de métricas, gráficos interativos e simulador de negócios."""
    metrics_df = res["metrics_df"]
    best_name = res["best_model_name"]
    best_pipe = res["best_pipeline"]
    y_test = res["y_test"]
    y_pred_best = res["predictions_test"][best_name]

    st.markdown("---")
    st.markdown("### 2. Comparativo de Desempenho dos Modelos")
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    best_r2 = metrics_df.iloc[0]["R² (Teste)"]
    best_mae = metrics_df.iloc[0]["MAE (Erro Médio Absoluto)"]
    best_rmse = metrics_df.iloc[0]["RMSE (Raiz do Erro Quadrático)"]

    with col_kpi1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Melhor Modelo</div>
                <div class="metric-value" style="font-size: 1.3rem; color: #818CF8;">{best_name}</div>
                <div class="metric-sub">Maior capacidade explicativa</div>
            </div>
        """, unsafe_allow_html=True)
    with col_kpi2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">R² Score (Teste)</div>
                <div class="metric-value">{best_r2:.3f}</div>
                <div class="metric-sub">{(best_r2 * 100):.1f}% da variância explicada</div>
            </div>
        """, unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Erro Médio Absoluto (MAE)</div>
                <div class="metric-value">{best_mae:,.2f}</div>
                <div class="metric-sub">RMSE: {best_rmse:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    # Tabela comparativa formatada
    styled_df = metrics_df.copy()
    styled_df["R² (Teste)"] = styled_df["R² (Teste)"].map(lambda x: f"{x:.4f}")
    styled_df["MAE (Erro Médio Absoluto)"] = styled_df["MAE (Erro Médio Absoluto)"].map(lambda x: f"{x:,.2f}")
    styled_df["RMSE (Raiz do Erro Quadrático)"] = styled_df["RMSE (Raiz do Erro Quadrático)"].map(lambda x: f"{x:,.2f}")
    styled_df["MAPE (%)"] = styled_df["MAPE (%)"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")
    st.dataframe(styled_df, use_container_width=True)

    # 3. Gráficos de Avaliação (Real vs Previsto e Resíduos)
    st.markdown("### 3. Diagnóstico Visual & Análise de Resíduos")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        # Gráfico Real vs Previsto
        fig_real_pred = go.Figure()
        fig_real_pred.add_trace(go.Scatter(
            x=y_test,
            y=y_pred_best,
            mode='markers',
            marker=dict(color='#6366F1', opacity=0.75, size=8),
            name='Predições (Teste)'
        ))
        
        # Linha ideal de 45 graus
        min_val = min(y_test.min(), y_pred_best.min())
        max_val = max(y_test.max(), y_pred_best.max())
        fig_real_pred.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='#EF4444', dash='dash', width=2),
            name='Previsão Perfeita (y = x)'
        ))
        
        fig_real_pred.update_layout(
            title=f"Valores Reais vs Previstos ({best_name})",
            xaxis_title=f"Real ({target_col})",
            yaxis_title=f"Previsto ({target_col})",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_real_pred, use_container_width=True)

    with col_g2:
        # Gráfico de Resíduos (Resíduos vs Previsto)
        residuals = y_test - y_pred_best
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=y_pred_best,
            y=residuals,
            mode='markers',
            marker=dict(color='#34D399', opacity=0.75, size=8),
            name='Resíduo (Real - Previsto)'
        ))
        fig_res.add_hline(y=0, line_color="#EF4444", line_dash="dash", line_width=2)
        fig_res.update_layout(
            title="Dispersão de Resíduos vs Previsão",
            xaxis_title="Valor Previsto",
            yaxis_title="Resíduo",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_res, use_container_width=True)

    # 4. Importância de Variáveis
    if res["feature_importances"] is not None and not res["feature_importances"].empty:
        st.markdown("### 4. Importância das Variáveis no Desempenho")
        top_feats = res["feature_importances"].head(10)
        fig_imp = px.bar(
            top_feats,
            x='Importancia',
            y='Feature',
            orientation='h',
            title=f"Top 10 Variáveis com Maior Impacto ({best_name})",
            template="plotly_dark",
            color='Importancia',
            color_continuous_scale="Purples"
        )
        fig_imp.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_imp, use_container_width=True)

    # 5. Simulador Interativo "What-If" para Tomada de Decisão de Mídia
    st.markdown("---")
    st.markdown("### 5. 🧮 Simulador de Investimento em Mídia (What-If)")
    st.markdown("Simule um cenário de alocação de orçamento e veja a previsão de resultado calculada pelo modelo campeão.")

    sim_inputs = {}
    col_s1, col_s2, col_s3 = st.columns(3)

    for i, col in enumerate(feature_cols):
        col_container = [col_s1, col_s2, col_s3][i % 3]
        with col_container:
            if col in res["num_features"]:
                min_v = float(df[col].min())
                max_v = float(df[col].max())
                median_v = float(df[col].median())
                sim_inputs[col] = st.slider(
                    f"{col}:",
                    min_value=min_v,
                    max_value=max_v * 1.5,
                    value=median_v,
                    key=f"sim_reg_{col}"
                )
            else:
                unique_opts = df[col].dropna().unique().tolist()
                sim_inputs[col] = st.selectbox(
                    f"{col}:",
                    options=unique_opts,
                    key=f"sim_reg_{col}"
                )

    sim_df = pd.DataFrame([sim_inputs])
    predicted_val = best_pipe.predict(sim_df)[0]

    st.markdown(f"""
        <div class="simulator-container">
            <h4 style="color: #818CF8; margin-top: 0;">🎯 Previsão do Cenário Simulado</h4>
            <div style="display: flex; gap: 30px; align-items: baseline; flex-wrap: wrap;">
                <div>
                    <span style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase;">Métrica Estimada ({target_col})</span>
                    <div style="font-size: 2.2rem; font-weight: 800; color: #34D399;">
                        {format_currency_brl(predicted_val) if 'brl' in target_col.lower() or 'receita' in target_col.lower() else f"{predicted_val:,.2f}"}
                    </div>
                </div>
                <div>
                    <span style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase;">Modelo Responsável</span>
                    <div style="font-size: 1.1rem; color: #F8FAFC; font-weight: 600;">{best_name}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # 6. Conclusões e Recomendações Táticas de Marketing
    st.markdown("""
        <div class="insight-box">
            <h4>💡 Interpretação de Negócio & Decisão de Mídia</h4>
            <p>
                <b>Alocação Eficiente:</b> Se a variável mais importante for o Investimento ou Cliques, 
                verifique o ponto em que a curva de retorno começa a estabilizar (retornos decrescentes). 
                Recomenda-se distribuir o budget entre os canais de maior elasticidade para maximizar o ROAS global.
            </p>
        </div>
    """, unsafe_allow_html=True)
