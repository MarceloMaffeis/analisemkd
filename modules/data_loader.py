"""
Carregador e Processador de Dados para Marketing Digital.
Suporta upload de arquivos CSV com detecção inteligente de delimitador e encoding,
além de disponibilizar datasets de demonstração prontos para uso.
"""

import os
import io
import pandas as pd
import numpy as np
import streamlit as st

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data")

SAMPLE_DATASETS = {
    "regressao": {
        "nome": "Previsão de Receita & ROAS (Regressão)",
        "arquivo": "marketing_regressao_roas.csv",
        "descricao": "500 campanhas com Investimento, Impressões, Cliques, CTR, CPC, CPA, Receita e ROAS.",
        "tipo": "Regressão"
    },
    "classificacao": {
        "nome": "Propensão de Conversão de Leads (Classificação)",
        "arquivo": "marketing_classificacao_leads.csv",
        "descricao": "800 leads com canais, dispositivo, tempo no site, páginas vistas e status de conversão (0/1).",
        "tipo": "Classificação"
    },
    "clusterizacao": {
        "nome": "Segmentação RFM & Comportamental (Clusterização)",
        "arquivo": "marketing_clusterizacao_clientes.csv",
        "descricao": "600 clientes com Recência, Frequência, Gasto Total, Visitas e Taxa de Abandono.",
        "tipo": "Clusterização"
    },
    "completo": {
        "nome": "Base Completa de Marketing Digital (Unificada)",
        "arquivo": "marketing_digital_completo.csv",
        "descricao": "700 registros com métricas de mídia, conversão e comportamento para todos os modelos.",
        "tipo": "Multi-Modelo"
    }
}

def load_sample_dataset(key):
    """Carrega um dos datasets de exemplo inclusos no projeto."""
    if key not in SAMPLE_DATASETS:
        return None
    file_path = os.path.join(SAMPLE_DIR, SAMPLE_DATASETS[key]["arquivo"])
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def parse_uploaded_csv(uploaded_file, sep=None, decimal=None, encoding=None):
    """
    Lê um arquivo CSV enviado pelo usuário tentando detectar delimitadores e codificações comuns.
    """
    bytes_data = uploaded_file.getvalue()
    
    # Tentativas de encodings comuns
    encodings_to_try = [encoding] if encoding else ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    delimiters_to_try = [sep] if sep else [',', ';', '\t', '|']
    
    df = None
    last_error = None
    
    for enc in encodings_to_try:
        if not enc:
            continue
        for d in delimiters_to_try:
            if not d:
                continue
            try:
                dec = decimal if decimal else (',' if d == ';' else '.')
                df_candidate = pd.read_csv(
                    io.BytesIO(bytes_data),
                    sep=d,
                    decimal=dec,
                    encoding=enc,
                    engine='c'
                )
                # Verifica se gerou mais de 1 coluna
                if df_candidate.shape[1] > 1:
                    df = df_candidate
                    break
            except Exception as e:
                last_error = e
                continue
        if df is not None:
            break
            
    if df is None:
        # Fallback usando Python engine com sniffer
        try:
            df = pd.read_csv(io.BytesIO(bytes_data), sep=None, engine='python', encoding='utf-8')
        except Exception:
            try:
                df = pd.read_csv(io.BytesIO(bytes_data), sep=None, engine='python', encoding='latin-1')
            except Exception as e:
                raise ValueError(f"Não foi possível processar o arquivo CSV. Erro: {last_error or e}")
                
    return clean_column_names(df)

def clean_column_names(df):
    """Normaliza nomes de colunas para evitar caracteres invisíveis ou quebras."""
    df = df.copy()
    cleaned = []
    for col in df.columns:
        c = str(col).strip()
        c = c.replace('"', '').replace("'", "")
        cleaned.append(c)
    df.columns = cleaned
    return df

def analyze_columns(df):
    """Classifica as colunas em numéricas, categóricas, booleanas e possíveis IDs."""
    numeric_cols = []
    categorical_cols = []
    id_cols = []
    datetime_cols = []
    
    for col in df.columns:
        col_lower = str(col).lower()
        # Verificar se é coluna de identificação
        if ('id' in col_lower or 'cod' in col_lower) and df[col].nunique() > len(df) * 0.8:
            id_cols.append(col)
            continue
            
        # Detectar datetime
        if 'data' in col_lower or 'date' in col_lower:
            try:
                pd.to_datetime(df[col], errors='raise')
                datetime_cols.append(col)
                continue
            except Exception:
                pass
                
        # Detectar numéricos
        if pd.api.types.is_numeric_dtype(df[col]):
            # Se for 0 e 1 apenas, pode ser binário
            if set(df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        else:
            categorical_cols.append(col)
            
    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "id": id_cols,
        "datetime": datetime_cols
    }

def get_csv_download_link(df, filename="template_marketing.csv"):
    """Gera dados CSV para download no formato padrão."""
    return df.to_csv(index=False).encode('utf-8')
