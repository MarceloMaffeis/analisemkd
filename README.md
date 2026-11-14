# 🚀 Marketing AI Lab — Machine Learning Aplicado a Resultados de Marketing Digital

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Plataforma inteligente completa desenvolvida em **Python** e **Streamlit** para análise de dados e modelagem preditiva com foco em resultados de **Marketing Digital** (Google Ads, Meta Ads, TikTok Ads, CRM e Inbound Marketing).

A aplicação contempla os três pilares essenciais de Machine Learning:
1. **Regressão**: Previsão de métricas contínuas de retorno financeiro (**Receita**, **ROAS**, **Conversões**).
2. **Classificação**: Predição de propensão e qualificação de leads (**Conversão Sim/Não**, **Lead Scoring Quente/Morno/Frio**).
3. **Clusterização**: Segmentação não supervisionada de clientes e campanhas (**Análise RFM**, **Personas**, **PCA 2D/3D**).

---

## 🎯 Funcionalidades Principais

### 1. Ingestão Flexível & Datasets de Demonstração
- **Upload de CSV pelo Cliente**: Suporte a qualquer tabela com detecção automática de delimitadores (`,`, `;`, `\t`), decimais e encodings (`utf-8`, `latin-1`).
- **4 Datasets Realistas Inclusos**:
  - `sample_data/marketing_regressao_roas.csv`: 500 campanhas com Investimento, Impressões, Cliques, CTR, CPC, CPA, Receita e ROAS.
  - `sample_data/marketing_classificacao_leads.csv`: 800 leads com canais, dispositivo, tempo no site, páginas navegadas e conversão (0/1).
  - `sample_data/marketing_clusterizacao_clientes.csv`: 600 clientes com Recência, Frequência, Valor Gasto e Taxa de Abandono.
  - `sample_data/marketing_digital_completo.csv`: 700 registros integrados para testes simultâneos de todos os módulos.
- **Botões de Download de Templates**: O cliente pode baixar os modelos em formato CSV com 1 clique para preencher com seus dados reais.

### 2. Análise Exploratória de Dados (EDA)
- Cálculo automático de KPIs globais de marketing: **CTR Global**, **CPC Médio**, **CPA Médio** e **ROAS Global**.
- Matriz de Correlação Linear interativa (Plotly Heatmap).
- Gráficos de dispersão customizáveis com linha de tendência OLS.
- Histogramas de distribuição com boxplots integrados para detecção de anomalias/outliers.

### 3. Módulo 1 — Regressão (Previsão de Receita & ROAS)
- **Modelos Treinados e Comparados**:
  - Regressão Linear (Baseline)
  - Ridge Regression (Regularização L2)
  - Random Forest Regressor
  - Gradient Boosting Regressor
- **Métricas Calculadas**: $R^2$, MAE (Erro Médio Absoluto), RMSE (Raiz do Erro Quadrático) e MAPE (%).
- **Visualizações**:
  - Gráfico de Dispersão Real vs Previsto com linha ideal de 45°.
  - Gráfico de Análise de Resíduos (Resíduos vs Previsto).
  - Ranking de Importância das Variáveis (Feature Importance).
- **Simulador What-If de Orçamento**: Permite alterar dinamicamente investimentos e canais para simular a receita e o ROAS previsto em tempo real.

### 4. Módulo 2 — Classificação (Propensão de Conversão de Leads)
- **Modelos Treinados e Comparados**:
  - Regressão Logística (Baseline)
  - Random Forest Classifier
  - Gradient Boosting Classifier
- **Métricas Calculadas**: Acurácia, F1-Score Ponderado, Precisão, Recall (Revocação) e ROC-AUC.
- **Visualizações**:
  - Matriz de Confusão Interativa com valores absolutos e percentuais.
  - Curva ROC (Receiver Operating Characteristic) com área sob a curva (AUC).
  - Importância dos atributos preditores de conversão.
- **Simulador de Lead Scoring**: Avalia um lead em tempo real e emite o diagnóstico de temperatura:
  - 🔥 **Lead Quente (Hot)**: Probabilidade $\ge 65\%$ (Contato imediato pelo time comercial).
  - ⚡ **Lead Morno (Warm)**: Probabilidade $35\% - 64\%$ (Nutrição por automação de e-mails e cases).
  - ❄️ **Lead Frio (Cold)**: Probabilidade $< 35\%$ (Campanhas de topo de funil e remarketing).

### 5. Módulo 3 — Clusterização (Segmentação RFM & Personas)
- **Modelagem**: Algoritmo K-Means com padronização rigorosa (`StandardScaler`).
- **Diagnóstico Algorítmico do Número Ideal de Grupos**:
  - Curva do Método do Cotovelo (Elbow Curve - Inércia WCSS).
  - Curva do Coeficiente de Silhueta para $k \in [2, 8]$.
- **Redução de Dimensionalidade (PCA)**:
  - Gráfico de Dispersão 2D com coloração por cluster.
  - Gráfico de Dispersão 3D Interativo e rotacionável.
- **Geração de Personas & Playbook Tático**:
  - Caracterização das personas (Ex: *Clientes VIP*, *Compradores Ocasionais*, *Clientes em Risco*).
  - Estratégias recomendadas de marketing para cada cluster.
  - **Exportação dos Resultados**: Download da base completa com as colunas de clusters atribuídos em CSV.

---

## 🛡️ Rigor Técnico & Boas Práticas de Machine Learning

Este projeto segue estritamente as diretrizes corporativas de engenharia de Machine Learning:
- **Strict Featurization Ordering (Sem Data Leakage)**: O conjunto de dados é dividido em treino e teste **antes** de qualquer cálculo de pré-processamento (imputadores, One-Hot Encoders e Scalers são ajustados exclusivamente no conjunto de treino).
- **Tratamento de Nulos**: Imputação por mediana para valores numéricos contínuos e moda para categóricos.
- **Pipelines Reutilizáveis**: Uso de `Pipeline` e `ColumnTransformer` do `scikit-learn` garantindo reprodutibilidade total.

---

## 📂 Estrutura do Repositório

```text
analisemkd/
├── app.py                      # Ponto de entrada da aplicação Streamlit
├── generate_sample_data.py     # Script gerador dos dados de exemplo
├── requirements.txt            # Dependências do projeto
├── .gitignore                  # Arquivos ignorados pelo controle de versão
├── README.md                   # Documentação completa
├── .streamlit/
│   └── config.toml             # Tema visual e configurações do servidor Streamlit
├── modules/
│   ├── __init__.py
│   ├── styles.py               # Estilização CSS e componentes de design
│   ├── data_loader.py          # Leitura de CSV e datasets de demonstração
│   ├── eda.py                  # Análise exploratória e correlações
│   ├── regression_module.py    # Módulo de Regressão e Simulador What-If
│   ├── classification_module.py# Módulo de Classificação e Lead Scoring
│   └── clustering_module.py    # Módulo de Clusterização e Personas
├── utils/
│   ├── __init__.py
│   └── metrics.py              # Cálculo e formatação de KPIs de Marketing
└── sample_data/
    ├── marketing_regressao_roas.csv
    ├── marketing_classificacao_leads.csv
    ├── marketing_clusterizacao_clientes.csv
    └── marketing_digital_completo.csv
```

---

## ⚙️ Como Executar Localmente

### 1. Clonar o Repositório
```bash
git clone https://github.com/MarceloMaffeis/analisemkd.git
cd analisemkd
```

### 2. Criar e Ativar um Ambiente Virtual (Opcional, mas Recomendado)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar as Dependências
```bash
pip install -r requirements.txt
```

### 4. Executar a Aplicação Streamlit
```bash
streamlit run app.py
```

A aplicação abrirá automaticamente no seu navegador no endereço `http://localhost:8501`.

---

## 🌐 Deploy no Streamlit Community Cloud

Para disponibilizar esta aplicação online gratuitamente:
1. Acesse [share.streamlit.io](https://share.streamlit.io/) e faça login com sua conta do GitHub.
2. Clique em **"New app"**.
3. Selecione o repositório `MarceloMaffeis/analisemkd`.
4. Defina o arquivo principal como `app.py`.
5. Clique em **"Deploy"**.

A aplicação estará acessível publicamente com HTTPS em poucos minutos!

---

## 📊 Glossário de Métricas de Marketing Digital

| Métrica | Nome | Significado | Fórmula |
| :--- | :--- | :--- | :--- |
| **CTR** | Click-Through Rate | Taxa de atratividade dos anúncios | $\frac{\text{Cliques}}{\text{Impressões}} \times 100$ |
| **CPC** | Cost Per Click | Custo médio por clique | $\frac{\text{Investimento}}{\text{Cliques}}$ |
| **CPA** | Cost Per Acquisition | Custo para gerar uma conversão | $\frac{\text{Investimento}}{\text{Conversões}}$ |
| **ROAS** | Return On Ad Spend | Retorno financeiro por real investido | $\frac{\text{Receita}}{\text{Investimento}}$ |
| **CR** | Conversion Rate | Taxa de conversão de cliques em vendas | $\frac{\text{Conversões}}{\text{Cliques}} \times 100$ |

---

## 📄 Licença

Distribuído sob a licença **MIT**. Consulte o arquivo `LICENSE` para mais detalhes.
