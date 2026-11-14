"""
Script de teste de fumaça (Smoke Test) para validar a integridade dos módulos ML.
"""
import sys
import pandas as pd
from modules.data_loader import load_sample_dataset
from modules.regression_module import train_regression_models
from modules.classification_module import train_classification_models
from modules.clustering_module import evaluate_clusters_range, run_kmeans_clustering
from utils.metrics import calculate_marketing_kpis

print("==> 1. Testando carregamento de datasets...")
df_reg = load_sample_dataset("regressao")
assert df_reg is not None and len(df_reg) > 0, "Falha ao carregar dataset de regressão"
print(f"Dataset Regressão: {df_reg.shape}")

df_clf = load_sample_dataset("classificacao")
assert df_clf is not None and len(df_clf) > 0, "Falha ao carregar dataset de classificação"
print(f"Dataset Classificação: {df_clf.shape}")

df_clu = load_sample_dataset("clusterizacao")
assert df_clu is not None and len(df_clu) > 0, "Falha ao carregar dataset de clusterização"
print(f"Dataset Clusterização: {df_clu.shape}")

print("==> 2. Testando métricas de marketing...")
kpis = calculate_marketing_kpis(df_reg)
print(f"Investimento: {kpis['total_investimento']}, Receita: {kpis['total_receita']}, ROAS: {kpis['roas_global']:.2f}")

print("==> 3. Testando Módulo de Regressão...")
reg_res = train_regression_models(
    df=df_reg,
    target_col="Receita_BRL",
    feature_cols=["Investimento_BRL", "Cliques", "CTR_Pct", "CPC_BRL", "Canal"],
    test_size=0.2,
    rf_estimators=30,
    random_state=42
)
print("Regressão treinada com sucesso! Melhor modelo:", reg_res['best_model_name'])
print(reg_res['metrics_df'])

print("==> 4. Testando Módulo de Classificação...")
clf_res = train_classification_models(
    df=df_clf,
    target_col="Converteu_Venda",
    feature_cols=["Origem_Trafego", "Dispositivo", "Tempo_No_Site_Min", "Paginas_Visitadas", "Score_Engajamento"],
    test_size=0.25,
    rf_estimators=30,
    random_state=42
)
print("Classificação treinada com sucesso! Melhor modelo:", clf_res['best_model_name'])
print(clf_res['metrics_df'])

print("==> 5. Testando Módulo de Clusterização...")
features_clu = ["Recencia_Dias", "Frequencia_Compras", "Valor_Total_Gasto_BRL", "Taxa_Abandono_Carrinho_Pct"]
eval_res = evaluate_clusters_range(df_clu[features_clu], max_k=5)
print("Melhor k sugerido:", eval_res['best_k'])
clu_res = run_kmeans_clustering(df_clu, features_clu, k=eval_res['best_k'], random_state=42)
print("Silhouette Score:", clu_res['silhouette'])
print("PCA Variância:", clu_res['pca_var_explained'])
print("\n[SUCESSO] TODOS OS TESTES PASSARAM COM SUCESSO!")
