# Projeto de Inteligência Artificial

Este repositório é destinado ao desenvolvimento e consolidação de modelos preditivos e de classificação aplicados a problemas práticos de Machine Learning e Inteligência Artificial. O projeto está dividido em duas frentes principais: a estimativa de valores de mercado de veículos (Regressão) e a identificação de transações financeiras suspeitas (Classificação).

O objetivo principal é aplicar técnicas avançadas de pré-processamento de dados, engenharia de atributos e análise estatística para treinar modelos com alta capacidade de generalização e performance.

---

## 🚗 1. Previsão de Preço de Carros (Car Price Prediction)

### 📋 Descrição do Problema

O objetivo desta frente é estimar o valor comercial de veículos automotores com base em suas características técnicas e dados históricos (como marca, ano de fabricação, quilometragem, tipo de combustível, potência do motor, entre outros). Trata-se de um problema de **Regressão**.

### 🛠️ Pré-processamento e Engenharia de Atributos

* Limpeza de dados e tratamento de valores ausentes ou inconsistentes.
* Codificação de variáveis categóricas (*Categorical Encoding*).
* Seleção de atributos (*Feature Selection*) para otimização do tempo de processamento e acurácia do modelo.

### 🤖 Algoritmos Utilizados

* **Árvore de Decisão (Decision Tree Regressor):** Utilizado para segmentar as características dos veículos de forma hierárquica e capturar relações não-lineares complexas.
* **KNN Regressão (K-Nearest Neighbors):** Aplicado para estimar os preços com base na proximidade e semelhança de veículos parecidos (vizinhos) dentro do espaço de atributos.
* **Regressão Linear Múltipla:** Modelo matemático estrutural de base para avaliar as relações lineares diretas entre os atributos preditores e o preço final do veículo.
* **SVM (Support Vector Regression - SVR):** Utilizado para mapear as variáveis do veículo em dimensões superiores, buscando definir uma margem ótima de tolerância para os erros de previsão.

### 📈 Resultados Obtidos

* **Árvore de Decisão:**
* **R² (Coeficiente de Determinação):** 95.81%
* **MAE (Erro Médio Absoluto):** 2066.42
--
* **KNN Regressão:**
* **MAE:** 2823
--
* **Regressão Linear Múltipla:**
* **MAE:** 3249
--
* **SVM:**
* **MAE:** 10152

---

## 💳 2. Detecção de Fraude de Cartão de Crédito (Credit Card Fraud Detection)

### 📋 Descrição do Problema

Esta frente visa identificar transações financeiras fraudulentas em operações de cartão de crédito de forma automatizada. Por envolver a separação entre transações legítimas e suspeitas, o problema é abordado como uma **Classificação**, enfrentando o desafio técnico de lidar com conjuntos de dados altamente desbalanceados.

### 🛠️ Pré-processamento e Engenharia de Atributos

* **Limpeza de Dados:** Filtragem e tratamento de ruídos nos registros de transações.
* **Engenharia de Atributos Temporais:** Extração de padrões comportamentais complexos a partir de campos de data e hora, convertendo variáveis de timestamp em atributos numéricos específicos, como *Hora do Dia* e *Dia da Semana*, permitindo mapear janelas temporais críticas de incidência de fraudes.
* **Tratamento de Desbalanceamento:** Ajustes de pesos ou técnicas de amostragem para lidar com a disparidade entre transações comuns e fraudes.

### 🤖 Algoritmos Utilizados

* **Redes Neurais (Neural Networks):** Implementadas para capturar relações complexas e padrões não-lineares ocultos no comportamento de consumo e uso do cartão.
* **Naive Bayes:** Utilizado como um classificador probabilístico rápido e eficiente baseado no Teorema de Bayes, servindo como uma excelente base de comparação estatística.

### 📈 Resultados Obtidos

* **Redes Neurais:**
* **Precisão (Classe Fraude):** 99,3%
--
* **Naive Bayes:**
* **Precisão (Classe Fraude):** 94,8%

---

## 🛠️ Tecnologias e Ferramentas Utilizadas

* **Python**
* **Pandas & NumPy:** Manipulação, limpeza e análise exploratória de dados.
* **Scikit-Learn:** Criação de pipelines de Machine Learning, engenharia de atributos e avaliação de métricas.
* **Keras / TensorFlow:** Construção e treinamento da arquitetura da Rede Neural.
* **Git & GitHub:** Controle de versão e documentação do projeto.
