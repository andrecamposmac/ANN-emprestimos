# Previsão de Pagamento de Empréstimos (LendingClub)

Este projeto consiste em um notebook Jupyter que desenvolve um modelo de Deep Learning para prever se um mutuário pagará ou não o seu empréstimo. A base de dados utilizada é do LendingClub, uma plataforma de empréstimos peer-to-peer.

## 📋 Descrição

O objetivo principal é realizar uma classificação binária para prever a variável alvo `loan_repaid` (1 = Pagou, 0 = Não Pagou / Charged Off), utilizando dados históricos de empréstimos. O projeto segue um fluxo completo de ciência de dados:

1.  **Análise Exploratória de Dados (EDA):**
    *   Visualização da distribuição da variável alvo (balanceamento de classes).
    *   Análise de correlação entre variáveis numéricas (mapa de calor).
    *   Investigação da relação entre *grade* (nota de crédito), *sub_grade* e o status do empréstimo.
2.  **Pré-processamento e Limpeza de Dados:**
    *   Tratamento de valores ausentes (missing data) em colunas como `emp_length`, `mort_acc`, `pub_rec_bankruptcies`, etc.
    *   Imputação de dados (ex: preencher `mort_acc` baseando-se em `total_acc`).
    *   Engenharia de atributos (Feature Engineering):
        *   Extração do código postal (zip code) do endereço.
        *   Extração do ano de emissão do empréstimo (`issue_d`).
        *   Conversão da coluna `term` (prazo) para numérico.
    *   Codificação de variáveis categóricas (One-Hot Encoding/Dummy variables).
    *   Normalização dos dados com `MinMaxScaler`.
3.  **Modelagem:**
    *   Construção de uma Rede Neural Artificial (ANN) com Keras/TensorFlow.
    *   Arquitetura Sequencial com camadas densas e Dropout para regularização.
4.  **Avaliação:**
    *   Análise das curvas de perda (loss) de treino e validação.
    *   Matriz de Confusão e Relatório de Classificação (Precision, Recall, F1-Score).

## 📂 Dataset

Os dados contêm informações sobre empréstimos concedidos pelo LendingClub.
*   **Variável Alvo:** `loan_status` (transformada em `loan_repaid`).
*   **Principais Features:**
    *   `loan_amnt`: Valor do empréstimo.
    *   `int_rate`: Taxa de juros.
    *   `installment`: Parcela mensal.
    *   `annual_inc`: Renda anual.
    *   `dti`: Debt-to-income ratio.
    *   `open_acc`, `total_acc`, `mort_acc`: Histórico de contas de crédito.
    *   `grade`, `sub_grade`: Classificação de risco do LendingClub.
    *   `address`: Endereço do mutuário.

## 🛠 Tecnologias e Bibliotecas

*   **Python**
*   **Pandas & NumPy:** Manipulação e limpeza de dados.
*   **Matplotlib & Seaborn:** Visualização de dados (histrogramas, boxplots, heatmaps).
*   **Scikit-learn:** Pré-processamento e métricas de avaliação.
*   **TensorFlow / Keras:** Construção e treinamento da rede neural.

## 🧠 Arquitetura do Modelo (Exemplo)

O modelo final sugerido no notebook possui uma estrutura similar a:
*   **Camada de Entrada:** Conectada às features processadas (após dummies e scaling).
*   **Camadas Ocultas:** Várias camadas densas (ex: 78 -> 39 -> 19 neurônios) com função de ativação **ReLU**.
*   **Dropout:** Camadas de dropout (ex: 20% a 50%) para evitar overfitting.
*   **Camada de Saída:** 1 neurônio com ativação **Sigmoid** (para classificação binária).
*   **Otimizador:** Adam.
*   **Loss Function:** Binary Crossentropy.

## 📊 Resultados e Métricas

O notebook avalia o modelo comparando as classes `Fully Paid` vs `Charged Off`.
*   É dada atenção especial às métricas de **f1-score**, **precision** e **recall**, dado o desbalanceamento natural das classes (muito mais pagadores do que inadimplentes).
*   O modelo é testado simulando a aprovação de um novo cliente aleatório para verificar a predição na prática.

## 🚀 Como Executar

1.  Certifique-se de ter o arquivo de dados (ex: `lending_club_loan_two.csv`) disponível.
2.  Abra o notebook `housing_emprestimos.ipynb` no Jupyter ou Google Colab.
3.  Ajuste o caminho de carregamento do CSV se necessário.
4.  Execute as células sequencialmente para realizar a limpeza, treino e avaliação.

---
**Autor:** André Campos Machado
