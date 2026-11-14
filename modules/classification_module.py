"""
Módulo de Machine Learning: CLASSIFICAÇÃO
Foco em Marketing Digital: Predição de Conversão de Leads, Qualificação e Sucesso de Campanhas.
Segue estritamente as diretrizes de 'ml-best-practices':
- Divisão Treino/Teste ANTES do pré-processamento (Strict Featurization Ordering)
- Tratamento de nulos com imputadores
- Codificação One-Hot para categóricos e StandardScaler para numéricos
- Treinamento e benchmark de múltiplos modelos (Logistic Regression, Random Forest, Gradient Boosting)
- Matriz de Confusão interativa, Curva ROC com AUC, Precision-Recall e Importância de Features
- Simulador Interativo de Lead Scoring com classificação Quente/Morno/Frio
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from utils.metrics import format_percentage, format_number

def render_classification_tab(df):
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
            <span class="badge badge-clf">Supervisionado</span>
            <span style="color: #94A3B8; font-size: 0.9rem;">Modelagem Categórica / Propensão</span>
        </div>
        <h2>🎯 Classificação: Propensão de Conversão & Qualificação de Leads</h2>
        <p style="color: #CBD5E1;">
            Identifique com precisão quais leads ou visitantes têm maior probabilidade de se tornarem clientes pagantes,
            otimizando o trabalho do time comercial e de automação de marketing.
        </p>
    """, unsafe_allow_html=True)

    if df is None or df.empty:
        st.warning("Carregue uma base de dados para iniciar a modelagem.")
        return

    # 1. Configuração de Variáveis
    st.markdown("### 1. Configuração do Modelo & Variáveis")
    all_cols = df.columns.tolist()

    # Detectar candidatas a alvo categórico / binário
    candidate_targets = []
    for col in all_cols:
        n_unique = df[col].nunique()
        if 2 <= n_unique <= 5:
            candidate_targets.append(col)

    if not candidate_targets:
        candidate_targets = [c for c in all_cols if df[c].nunique() <= 10]

    if not candidate_targets:
        st.error("Não foram encontradas colunas categóricas ou binárias adequadas para classificação (entre 2 e 5 classes).")
        return

    # Seleção de alvo padrão
    default_targets = ['Converteu_Venda', 'converteu', 'conversao', 'Campanha_Lucrativa', 'target', 'classe', 'status']
    target_idx = 0
    for dt in default_targets:
        for idx, c in enumerate(candidate_targets):
            if dt.lower() in c.lower():
                target_idx = idx
                break

    col_cfg1, col_cfg2 = st.columns([1, 2])
    with col_cfg1:
        target_col = st.selectbox(
            "🎯 Variável Alvo Categórica / Binária (Target Y):",
            options=candidate_targets,
            index=target_idx,
            help="Variável indicando o resultado a ser previsto (ex: Converteu 0/1, Lead Qualificado Sim/Não)."
        )

    # Filtrar IDs da lista padrão de features
    potential_features = [c for c in all_cols if c != target_col]
    default_features = [
        c for c in potential_features
        if not any(id_word in c.lower() for id_word in ['_id', 'id_', 'lead_id', 'id_registro', 'cod_'])
    ]

    with col_cfg2:
        feature_cols = st.multiselect(
            "🧩 Variáveis Preditoras (Features X):",
            options=potential_features,
            default=default_features if default_features else potential_features[:5],
            help="Atributos comportamentais e demográficos do lead (ex: Tempo no site, Páginas visitadas, Canal de origem)."
        )

    if not feature_cols:
        st.warning("Selecione pelo menos uma variável explicativa.")
        return

    with st.expander("⚙️ Hiperparâmetros de Treinamento", expanded=False):
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            test_size = st.slider("Proporção do Teste (%):", min_value=10, max_value=40, value=25, step=5, key="clf_test_size") / 100
        with c_p2:
            rf_estimators = st.slider("Árvores no Ensemble:", min_value=50, max_value=250, value=100, step=25, key="clf_estimators")
        with c_p3:
            random_state = st.number_input("Random Seed:", value=42, step=1, key="clf_seed")

    train_button = st.button("🚀 Treinar & Avaliar Classificadores", type="primary", use_container_width=True)

    if "clf_results" not in st.session_state or train_button:
        if train_button or "clf_results" not in st.session_state:
            with st.spinner("Preparando variáveis, treinando modelos e gerando curvas de calibração..."):
                results = train_classification_models(
                    df=df,
                    target_col=target_col,
                    feature_cols=feature_cols,
                    test_size=test_size,
                    rf_estimators=rf_estimators,
                    random_state=random_state
                )
                st.session_state["clf_results"] = results
                st.session_state["clf_target"] = target_col
                st.session_state["clf_features"] = feature_cols

    if "clf_results" in st.session_state and st.session_state.get("clf_target") == target_col:
        res = st.session_state["clf_results"]
        display_classification_results(res, df, target_col, feature_cols)

def train_classification_models(df, target_col, feature_cols, test_size=0.25, rf_estimators=100, random_state=42):
    """Treina e compara modelos de classificação com validação sem vazamento de dados."""
    data = df.dropna(subset=[target_col]).copy()
    X = data[feature_cols]
    y_raw = data[target_col]

    # Codificar alvo se for string
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = [str(c) for c in label_encoder.classes_]
    is_binary = len(class_names) == 2

    num_features = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Divisão de Treino e Teste antes dos transformadores
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Pré-processamento
    transformers = []
    if num_features:
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', num_pipe, num_features))

    if cat_features:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat', cat_pipe, cat_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    models = {
        "Regressão Logística (Baseline)": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=rf_estimators, max_depth=10, random_state=random_state),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=rf_estimators, random_state=random_state, learning_rate=0.08)
    }

    metrics_list = []
    fitted_pipelines = {}
    preds_test = {}
    probas_test = {}

    for name, model in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        preds_test[name] = y_pred

        y_proba = None
        auc_val = np.nan
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)
            probas_test[name] = y_proba
            if is_binary:
                try:
                    auc_val = roc_auc_score(y_test, y_proba[:, 1])
                except Exception:
                    pass
            else:
                try:
                    auc_val = roc_auc_score(y_test, y_proba, multi_class='ovr')
                except Exception:
                    pass

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        metrics_list.append({
            "Modelo": name,
            "Acurácia": acc,
            "F1-Score": f1,
            "Precisão": prec,
            "Recall (Revocação)": rec,
            "ROC-AUC": auc_val
        })
        fitted_pipelines[name] = pipe

    metrics_df = pd.DataFrame(metrics_list).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
    best_name = metrics_df.iloc[0]["Modelo"]
    best_pipe = fitted_pipelines[best_name]

    # Importância de Features
    feature_names = []
    try:
        feature_names = best_pipe.named_steps['preprocessor'].get_feature_names_out()
        feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]
    except Exception:
        feature_names = [f"feat_{i}" for i in range(len(num_features) + len(cat_features))]

    importances = None
    clf_step = best_pipe.named_steps['classifier']
    if hasattr(clf_step, 'feature_importances_') and len(feature_names) == len(clf_step.feature_importances_):
        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importancia': clf_step.feature_importances_
        }).sort_values(by='Importancia', ascending=False)
    elif hasattr(clf_step, 'coef_') and len(feature_names) == len(clf_step.coef_[0]):
        importances = pd.DataFrame({
            'Feature': feature_names,
            'Importancia': np.abs(clf_step.coef_[0])
        }).sort_values(by='Importancia', ascending=False)

    return {
        "metrics_df": metrics_df,
        "best_model_name": best_name,
        "best_pipeline": best_pipe,
        "fitted_pipelines": fitted_pipelines,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "preds_test": preds_test,
        "probas_test": probas_test,
        "class_names": class_names,
        "is_binary": is_binary,
        "feature_importances": importances,
        "num_features": num_features,
        "cat_features": cat_features,
        "label_encoder": label_encoder
    }

def display_classification_results(res, df, target_col, feature_cols):
    """Exibe resultados com matriz de confusão interativa, curva ROC e Simulador de Lead Scoring."""
    metrics_df = res["metrics_df"]
    best_name = res["best_model_name"]
    best_pipe = res["best_pipeline"]
    y_test = res["y_test"]
    y_pred_best = res["preds_test"][best_name]
    class_names = res["class_names"]
    is_binary = res["is_binary"]

    st.markdown("---")
    st.markdown("### 2. Comparativo de Desempenho dos Classificadores")

    best_row = metrics_df.iloc[0]
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)

    with col_k1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Melhor Modelo</div>
                <div class="metric-value" style="font-size: 1.25rem; color: #34D399;">{best_name}</div>
                <div class="metric-sub">Maior equilíbrio global</div>
            </div>
        """, unsafe_allow_html=True)
    with col_k2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Acurácia Global</div>
                <div class="metric-value">{format_percentage(best_row['Acurácia'] * 100)}</div>
                <div class="metric-sub">Acertos totais no teste</div>
            </div>
        """, unsafe_allow_html=True)
    with col_k3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">F1-Score Ponderado</div>
                <div class="metric-value">{best_row['F1-Score']:.3f}</div>
                <div class="metric-sub">Harmonia Precisão/Recall</div>
            </div>
        """, unsafe_allow_html=True)
    with col_k4:
        val_auc = f"{best_row['ROC-AUC']:.3f}" if pd.notnull(best_row['ROC-AUC']) else "N/A"
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ROC-AUC</div>
                <div class="metric-value">{val_auc}</div>
                <div class="metric-sub">Discriminação das classes</div>
            </div>
        """, unsafe_allow_html=True)

    # Tabela comparativa
    disp_df = metrics_df.copy()
    for col in ["Acurácia", "F1-Score", "Precisão", "Recall (Revocação)"]:
        disp_df[col] = disp_df[col].map(lambda x: f"{x * 100:.2f}%")
    disp_df["ROC-AUC"] = disp_df["ROC-AUC"].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "-")
    st.dataframe(disp_df, use_container_width=True)

    # 3. Matriz de Confusão e Curva ROC
    st.markdown("### 3. Matriz de Confusão & Curvas de Desempenho")
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        # Matriz de Confusão com Plotly
        cm = confusion_matrix(y_test, y_pred_best)
        cm_norm = confusion_matrix(y_test, y_pred_best, normalize='true') * 100

        text_matrix = [[f"{cm[i][j]}<br>({cm_norm[i][j]:.1f}%)" for j in range(len(class_names))] for i in range(len(class_names))]

        fig_cm = px.imshow(
            cm,
            text_auto=False,
            x=class_names,
            y=class_names,
            labels=dict(x="Classe Prevista", y="Classe Real", color="Contagem"),
            title=f"Matriz de Confusão ({best_name})",
            color_continuous_scale="Blues"
        )
        # Adicionar texto customizado com percentuais
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                fig_cm.add_annotation(
                    x=j, y=i,
                    text=text_matrix[i][j],
                    showarrow=False,
                    font=dict(color="white" if cm[i][j] > cm.max()/2 else "#CBD5E1", size=14)
                )

        fig_cm.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_c2:
        # Curva ROC para problemas binários
        if is_binary and best_name in res["probas_test"]:
            y_proba_1 = res["probas_test"][best_name][:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba_1)
            auc_score = roc_auc_score(y_test, y_proba_1)

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC ({best_name}) [AUC={auc_score:.3f}]', line=dict(color='#10B981', width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aleatório (AUC = 0.50)', line=dict(color='#EF4444', dash='dash')))

            fig_roc.update_layout(
                title="Curva ROC (Receiver Operating Characteristic)",
                xaxis_title="Taxa de Falsos Positivos (FPR)",
                yaxis_title="Taxa de Verdadeiros Positivos (TPR)",
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            # Importância de Features alternativa
            if res["feature_importances"] is not None:
                top_feats = res["feature_importances"].head(10)
                fig_imp = px.bar(
                    top_feats,
                    x='Importancia',
                    y='Feature',
                    orientation='h',
                    title=f"Importância das Variáveis ({best_name})",
                    template="plotly_dark",
                    color='Importancia',
                    color_continuous_scale="Greens"
                )
                fig_imp.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_imp, use_container_width=True)

    # 4. Simulador de Lead Scoring e Qualificação
    st.markdown("---")
    st.markdown("### 4. 🎯 Simulador de Lead Scoring & Propensão em Tempo Real")
    st.markdown("Insira os dados de navegação ou perfil de um novo lead para estimar instantaneamente a probabilidade de conversão.")

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
                    max_value=max_v,
                    value=median_v,
                    key=f"sim_clf_{col}"
                )
            else:
                unique_opts = df[col].dropna().unique().tolist()
                sim_inputs[col] = st.selectbox(
                    f"{col}:",
                    options=unique_opts,
                    key=f"sim_clf_{col}"
                )

    sim_df = pd.DataFrame([sim_inputs])
    pred_class = best_pipe.predict(sim_df)[0]
    class_label = res["class_names"][pred_class]

    prob_pct = None
    if hasattr(best_pipe, "predict_proba"):
        probas = best_pipe.predict_proba(sim_df)[0]
        prob_pct = (probas[1] * 100) if is_binary else (probas[pred_class] * 100)

    # Classificação de Lead (Temperatura)
    if is_binary and prob_pct is not None:
        if prob_pct >= 65:
            tier_badge = "🔥 Lead Quente (Hot Lead)"
            tier_color = "#10B981"
            action_text = "Acione imediatamente o time de vendas (SDR/Inside Sales) para contato em menos de 15 minutos via WhatsApp ou Ligação."
        elif prob_pct >= 35:
            tier_badge = "⚡ Lead Morno (Warm Lead)"
            tier_color = "#F59E0B"
            action_text = "Insira em fluxo de nutrição de e-mails com cases de sucesso, webinars e ofertas de demonstração guiada."
        else:
            tier_badge = "❄️ Lead Frio (Cold Lead)"
            tier_color = "#64748B"
            action_text = "Mantenha em campanhas de topo de funil com conteúdos educativos e remarketing de baixa frequência."
    else:
        tier_badge = f"Classe Prevista: {class_label}"
        tier_color = "#818CF8"
        action_text = "Adote a estratégia específica conforme a taxonomia definida no seu funil de marketing."

    st.markdown(f"""
        <div class="simulator-container">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; margin-bottom: 1rem;">
                <h4 style="color: #F8FAFC; margin: 0;">Diagnóstico de Propensão</h4>
                <span style="background: {tier_color}22; color: {tier_color}; border: 1px solid {tier_color}55; padding: 4px 14px; border-radius: 999px; font-weight: 700; font-size: 0.85rem;">
                    {tier_badge}
                </span>
            </div>
            <div style="display: flex; gap: 40px; align-items: baseline; flex-wrap: wrap;">
                <div>
                    <span style="color: #94A3B8; font-size: 0.82rem; text-transform: uppercase;">Probabilidade de Conversão</span>
                    <div style="font-size: 2.2rem; font-weight: 800; color: {tier_color};">
                        {f'{prob_pct:.1f}%' if prob_pct is not None else class_label}
                    </div>
                </div>
                <div>
                    <span style="color: #94A3B8; font-size: 0.82rem; text-transform: uppercase;">Recomendação Tática Imediata</span>
                    <div style="font-size: 0.95rem; color: #E2E8F0; margin-top: 4px; max-width: 600px;">
                        {action_text}
                    </div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
