"""
Script para geração de bases de dados realistas de Marketing Digital
utilizadas para demonstração e testes imediatos nos 3 módulos de Machine Learning:
- Regressão (Previsão de Receita e ROAS)
- Classificação (Propensão de Conversão de Leads)
- Clusterização (Segmentação Comportamental RFM de Clientes)
- Base Unificada Completa
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

os.makedirs("sample_data", exist_ok=True)

# -------------------------------------------------------------
# 1. Dataset para REGRESSÃO: Previsão de Receita e ROAS
# -------------------------------------------------------------
n_reg = 500
canais = ['Google Ads', 'Meta Ads', 'TikTok Ads', 'LinkedIn Ads', 'YouTube Ads']
tipos_campanha = ['Search', 'Display', 'Video', 'Feed', 'Retargeting']
publicos = ['Lookalike 1%', 'Remarketing 30d', 'Interesses Gerais', 'B2B Decisores']

canal_col = np.random.choice(canais, size=n_reg, p=[0.35, 0.30, 0.15, 0.10, 0.10])
tipo_col = np.random.choice(tipos_campanha, size=n_reg)
publico_col = np.random.choice(publicos, size=n_reg)

# Investimento de R$ 500 a R$ 25.000
investimento = np.random.uniform(500, 25000, size=n_reg).round(2)

# Multiplicadores de canal para CTR e CPC
cpc_base = {'Google Ads': 2.8, 'Meta Ads': 1.6, 'TikTok Ads': 0.9, 'LinkedIn Ads': 7.5, 'YouTube Ads': 1.2}
cpc = np.array([cpc_base[c] * np.random.uniform(0.75, 1.35) for c in canal_col]).round(2)

cliques = (investimento / cpc).astype(int) + np.random.randint(10, 100, size=n_reg)
ctr = np.random.uniform(1.2, 5.8, size=n_reg).round(2)
impressoes = ((cliques / (ctr / 100)) + np.random.randint(100, 2000, size=n_reg)).astype(int)

# Taxa de conversão influenciada pelo tipo de campanha (Retargeting converte mais)
conv_rate_base = {'Retargeting': 0.055, 'Search': 0.040, 'Feed': 0.025, 'Video': 0.015, 'Display': 0.010}
conv_rates = np.array([conv_rate_base[t] * np.random.uniform(0.7, 1.3) for t in tipo_col])
conversoes = np.maximum(1, (cliques * conv_rates).astype(int))

ticket_medio = np.random.uniform(140, 650, size=n_reg).round(2)
# Receita: Conversoes * Ticket Médio com retorno decrescente para investimentos muito altos
receita = (conversoes * ticket_medio * np.random.uniform(0.9, 1.15)).round(2)
roas = (receita / investimento).round(2)
cpa = (investimento / conversoes).round(2)

df_reg = pd.DataFrame({
    'Campanha_ID': [f'CAMP_{i+1:04d}' for i in range(n_reg)],
    'Canal': canal_col,
    'Tipo_Campanha': tipo_col,
    'Publico_Alvo': publico_col,
    'Investimento_BRL': investimento,
    'Impressoes': impressoes,
    'Cliques': cliques,
    'CTR_Pct': ctr,
    'CPC_BRL': cpc,
    'Conversoes': conversoes,
    'CPA_BRL': cpa,
    'Ticket_Medio_BRL': ticket_medio,
    'Receita_BRL': receita,
    'ROAS': roas
})
df_reg.to_csv("sample_data/marketing_regressao_roas.csv", index=False)
print("1. marketing_regressao_roas.csv gerado com sucesso!")

# -------------------------------------------------------------
# 2. Dataset para CLASSIFICAÇÃO: Conversão de Leads
# -------------------------------------------------------------
n_clf = 800
origens = ['Google Ads', 'Meta Ads', 'Organico', 'Email Marketing', 'LinkedIn Ads', 'Direto']
dispositivos = ['Mobile', 'Desktop', 'Tablet']

origem_col = np.random.choice(origens, size=n_clf, p=[0.30, 0.25, 0.18, 0.12, 0.08, 0.07])
disp_col = np.random.choice(dispositivos, size=n_clf, p=[0.60, 0.35, 0.05])
idade = np.random.randint(18, 68, size=n_clf)
tempo_site = np.random.exponential(scale=4.5, size=n_clf).round(1) # minutos
paginas = np.maximum(1, np.random.poisson(lam=4.2, size=n_clf))
view_precos = np.random.binomial(1, p=0.42, size=n_clf)
baixou_material = np.random.binomial(1, p=0.35, size=n_clf)
form_preenchido = np.random.binomial(1, p=0.28, size=n_clf)
abriu_email = np.random.binomial(1, p=0.48, size=n_clf)
clicou_email = np.where(abriu_email == 1, np.random.binomial(1, p=0.55, size=n_clf), 0)
chat_interacoes = np.random.choice([0, 1, 2, 3], size=n_clf, p=[0.65, 0.22, 0.10, 0.03])
score_engajamento = (tempo_site * 3 + paginas * 4 + view_precos * 20 + form_preenchido * 25 + clicou_email * 15 + chat_interacoes * 10).clip(0, 100).round(1)

# Probabilidade logística de conversão
logit = -3.8 + 0.12 * tempo_site + 0.18 * paginas + 1.2 * view_precos + 1.5 * form_preenchido + 0.8 * clicou_email + 0.6 * chat_interacoes
prob_conv = 1 / (1 + np.exp(-logit))
converteu = np.random.binomial(1, p=prob_conv)

df_clf = pd.DataFrame({
    'Lead_ID': [f'LEAD_{i+1:05d}' for i in range(n_clf)],
    'Origem_Trafego': origem_col,
    'Dispositivo': disp_col,
    'Idade': idade,
    'Tempo_No_Site_Min': tempo_site,
    'Paginas_Visitadas': paginas,
    'Visualizou_Precos': view_precos,
    'Baixou_Ebook': baixou_material,
    'Preencheu_Formulario': form_preenchido,
    'Abriu_Email': abriu_email,
    'Clicou_Email': clicou_email,
    'Interacoes_Chat': chat_interacoes,
    'Score_Engajamento': score_engajamento,
    'Converteu_Venda': converteu
})
df_clf.to_csv("sample_data/marketing_classificacao_leads.csv", index=False)
print("2. marketing_classificacao_leads.csv gerado com sucesso!")

# -------------------------------------------------------------
# 3. Dataset para CLUSTERIZAÇÃO: Segmentação RFM de Clientes
# -------------------------------------------------------------
n_clu = 600

# Vamos criar 4 perfis/personas reais de clientes:
# 1: Campeões / VIPs (Alta frequência, alto valor, baixa recência)
# 2: Clientes Fieis Recorrentes (Frequência média-alta, valor moderado, recência baixa)
# 3: Clientes em Risco / Inativos (Frequência moderada, recência alta)
# 4: Compradores Ocasionais / Pechinchadores (Baixa frequência, baixo valor, alto abandono)
n_perfil = n_clu // 4

# Perfil 1: VIPs
rec_1 = np.random.randint(1, 45, size=n_perfil)
freq_1 = np.random.randint(12, 35, size=n_perfil)
gasto_1 = np.random.uniform(4500, 18000, size=n_perfil).round(2)
visitas_1 = np.random.randint(15, 45, size=n_perfil)
email_1 = np.random.uniform(60, 95, size=n_perfil).round(1)
abandono_1 = np.random.uniform(10, 30, size=n_perfil).round(1)

# Perfil 2: Clientes Fieis
rec_2 = np.random.randint(15, 90, size=n_perfil)
freq_2 = np.random.randint(5, 14, size=n_perfil)
gasto_2 = np.random.uniform(1500, 4800, size=n_perfil).round(2)
visitas_2 = np.random.randint(8, 25, size=n_perfil)
email_2 = np.random.uniform(40, 75, size=n_perfil).round(1)
abandono_2 = np.random.uniform(25, 50, size=n_perfil).round(1)

# Perfil 3: Em Risco / Churn Potencial
rec_3 = np.random.randint(120, 360, size=n_perfil)
freq_3 = np.random.randint(3, 10, size=n_perfil)
gasto_3 = np.random.uniform(800, 3000, size=n_perfil).round(2)
visitas_3 = np.random.randint(1, 8, size=n_perfil)
email_3 = np.random.uniform(10, 40, size=n_perfil).round(1)
abandono_3 = np.random.uniform(50, 85, size=n_perfil).round(1)

# Perfil 4: Compradores Ocasionais
rec_4 = np.random.randint(60, 240, size=n_perfil)
freq_4 = np.random.randint(1, 3, size=n_perfil)
gasto_4 = np.random.uniform(80, 750, size=n_perfil).round(2)
visitas_4 = np.random.randint(2, 12, size=n_perfil)
email_4 = np.random.uniform(5, 30, size=n_perfil).round(1)
abandono_4 = np.random.uniform(60, 95, size=n_perfil).round(1)

df_clu = pd.DataFrame({
    'Cliente_ID': [f'CLI_{i+1:05d}' for i in range(n_clu)],
    'Recencia_Dias': np.concatenate([rec_1, rec_2, rec_3, rec_4]),
    'Frequencia_Compras': np.concatenate([freq_1, freq_2, freq_3, freq_4]),
    'Valor_Total_Gasto_BRL': np.concatenate([gasto_1, gasto_2, gasto_3, gasto_4]),
    'Visitas_Mensais_Site': np.concatenate([visitas_1, visitas_2, visitas_3, visitas_4]),
    'Taxa_Abertura_Email_Pct': np.concatenate([email_1, email_2, email_3, email_4]),
    'Taxa_Abandono_Carrinho_Pct': np.concatenate([abandono_1, abandono_2, abandono_3, abandono_4])
})
df_clu['Ticket_Medio_BRL'] = (df_clu['Valor_Total_Gasto_BRL'] / df_clu['Frequencia_Compras']).round(2)
# Embaralhar para não ficar ordenado por perfil
df_clu = df_clu.sample(frac=1, random_state=42).reset_index(drop=True)
df_clu.to_csv("sample_data/marketing_clusterizacao_clientes.csv", index=False)
print("3. marketing_clusterizacao_clientes.csv gerado com sucesso!")

# -------------------------------------------------------------
# 4. Dataset Completo Unificado
# -------------------------------------------------------------
n_full = 700
df_full = pd.DataFrame({
    'ID_Registro': [f'REG_{i+1:05d}' for i in range(n_full)],
    'Canal_Marketing': np.random.choice(['Google Search', 'Meta Instagram', 'TikTok Ads', 'Meta Facebook', 'LinkedIn B2B', 'Email CRM'], size=n_full),
    'Segmento_Publico': np.random.choice(['B2B Enterprise', 'PME / Pessoas Juridicas', 'Consumidor Jovem', 'Publico Alta Renda'], size=n_full),
    'Dispositivo': np.random.choice(['Mobile', 'Desktop'], size=n_full, p=[0.65, 0.35]),
    'Investimento_BRL': np.random.uniform(400, 15000, size=n_full).round(2),
    'Cliques': np.random.randint(150, 8500, size=n_full),
    'Impressoes': np.random.randint(5000, 250000, size=n_full),
    'Tempo_Medio_Sessao_Seg': np.random.uniform(25, 450, size=n_full).round(1),
    'Paginas_Por_Sessao': np.random.uniform(1.2, 9.5, size=n_full).round(1),
    'Taxa_Rejeicao_Pct': np.random.uniform(20.0, 78.0, size=n_full).round(2),
    'Conversoes': np.random.randint(5, 320, size=n_full),
    'Receita_Total_BRL': np.random.uniform(1000, 65000, size=n_full).round(2),
    'Campanha_Lucrativa': np.random.choice([0, 1], size=n_full, p=[0.38, 0.62]),
    'Nivel_Engajamento': np.random.choice(['Baixo', 'Medio', 'Alto'], size=n_full, p=[0.25, 0.45, 0.30])
})
df_full['CTR_Pct'] = ((df_full['Cliques'] / df_full['Impressoes']) * 100).round(2)
df_full['CPC_BRL'] = (df_full['Investimento_BRL'] / df_full['Cliques']).round(2)
df_full['ROAS'] = (df_full['Receita_Total_BRL'] / df_full['Investimento_BRL']).round(2)
df_full.to_csv("sample_data/marketing_digital_completo.csv", index=False)
print("4. marketing_digital_completo.csv gerado com sucesso!")
