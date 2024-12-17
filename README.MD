# 🚀 Projeto Acadêmico: Construindo uma aplicação de Perguntas e Respostas sobre RH com RAG (Retrieval Augmented Generation)
Esta aplicação foi desenvolvida como exemplo prático do curso: Amazon Bedrock, Amazon Q & AWS Generative AI HANDS-ON, ministrado pelo instrutor Rahul Trisal, na plataforma Udemy.

## 📋 Descrição do Projeto
Consiste numa aplicação que atraves de Inteligência Artificial, responde perguntas sobre um documento de domínio público contendo políticas de licenças de empregados disponível em: https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf.


Ela usa as informações disponíveis no PDF, "quebra" essas informações em chunks, transforma em embbedings, armazena em um banco de dados vetorial (FAISS) para fornecer um contexto para a LLM (Llama 3) conseguir responder a pergunta do usuário, utilizando o mecanismo de Recuperação e Geração aumentada (RAG - Retrieval Augmented Generation).

## 🛠️ Tecnologias e conceitos utilizados

- Python 3.12
- Langchain
- AWS Bedrock (IA = Amazon Titan para os embeddings)
- AWS Bedrock (Llama 3 = Para recuperação e geração aumentada)
- FAISS
- Streamlit
- Anaconda Navigator
- VS Code
- Boto3

## ⚙️ Como Configurar e Executar o Projeto
Considera-se que o usuário já possua o Python (preferencialmente o 3.12) e o pip instalado em sua máquina, além de ter conta na AWS e AWS SSO configurado. 

- ## 1 - Instale o Anaconda Navigator
  - Abra o VS Code pelo Anaconda Navigator

- ## 2 - Clone o repositório
   - `git clone <URL do repositório>`
   - `cd nome-do-repositorio`

- ## 3 - Instale as dependências via pip
   - langchain                   Versão: 0.3.12
   - langchain-aws               Versão: 0.2.9
   - langchain-community         Versão: 0.3.12
   - boto3                       Versão: 1.35.81
   - streamlit                   Versão: 1.37.1

- ## 4 - Inicie a authenticação aws sso. 
   1 - Substituir as strings nos campos: `profile_name` e `credentials_profile_name` pelo seu perfil SSO

   2 - Fazer a autenticação pelo terminal utilizando o comando:
   ```
      aws sso login --profile seu_perfil_aws_aqui
   ```
   
   - OBS.: Não esqueça que o sso tem limite de 24hrs, depois deste prazo, terá que gerar um novo token
   - Certifique-se que o seu console está bem configurado

- ## 5 - Rodar a aplicação
   - Executar o comando abaixo no terminal:

   ```bash
      streamlit run rag_frontend.py
   ```