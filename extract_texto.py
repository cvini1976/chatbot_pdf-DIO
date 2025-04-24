## Instalar a biblioteca.
pip install unstructured[local-file]

import os
from unstructured.partition.auto import partition

def extrair_texto_de_pdfs(pasta_pdfs):
    textos = {}
    for nome_arquivo in os.listdir(pasta_pdfs):
        if nome_arquivo.endswith(".pdf"):
            caminho_arquivo = os.path.join(pasta_pdfs, nome_arquivo)
            try:
                elementos = partition(filename=caminho_arquivo)
                texto_completo = "\n\n".join([str(el) for el in elementos])
                textos[nome_arquivo] = texto_completo
                print(f"Texto extraído de: {nome_arquivo}")
            except Exception as e:
                print(f"Erro ao processar {nome_arquivo}: {e}")
    return textos

# Substitua pelo caminho da sua pasta de PDFs
pasta_dos_pdfs = "/caminho/para/seus/pdfs"
textos_extraidos = extrair_texto_de_pdfs(pasta_dos_pdfs)

# O dicionário 'textos_extraidos' agora contém o texto de cada PDF,
# com o nome do arquivo como chave.
# Você pode imprimir para verificar o conteúdo (com cuidado, pode ser longo).
# print(textos_extraidos)

from azure.core.credentials import AzureKeyCredential
from azure.openai import OpenAIClient

# Substitua pelos seus valores de chave e endpoint do Azure OpenAI
chave_openai = os.environ.get("AZURE_OPENAI_KEY")
endpoint_openai = os.environ.get("AZURE_OPENAI_ENDPOINT")

if not chave_openai or not endpoint_openai:
    raise ValueError("As variáveis de ambiente AZURE_OPENAI_KEY e AZURE_OPENAI_ENDPOINT não estão definidas.")

cliente = OpenAIClient(endpoint=endpoint_openai, credential=AzureKeyCredential(chave_openai))
modelo_embedding = "text-embedding-ada-002"

def gerar_embeddings(textos: dict, cliente: OpenAIClient, modelo: str):
    embeddings = {}
    for nome_arquivo, texto in textos.items():
        try:
            resposta = cliente.embeddings.create(input=texto, model=modelo)
            # Geralmente, a resposta contém uma lista de embeddings. Assumimos que para cada texto, teremos um embedding.
            # Se você dividiu o texto em chunks menores, precisará ajustar isso.
            embeddings[nome_arquivo] = resposta.data[0].embedding
            print(f"Embedding gerado para: {nome_arquivo}")
        except Exception as e:
            print(f"Erro ao gerar embedding para {nome_arquivo}: {e}")
    return embeddings

embeddings_dos_pdfs = gerar_embeddings(textos_extraidos, cliente, modelo_embedding)

# O dicionário 'embeddings_dos_pdfs' agora contém os embeddings para cada PDF,
# com o nome do arquivo como chave.
# print(embeddings_dos_pdfs)

## INstalar a biblioteca do Azure AI Search.
pip install azure-search-documents

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector

# Substitua pelos seus valores do Azure AI Search
nome_servico_pesquisa = os.environ.get("AZURE_SEARCH_SERVICE_NAME")
chave_admin_pesquisa = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
nome_indice = "meu-indice-de-pdfs"  # Escolha um nome para o seu índice

if not nome_servico_pesquisa or not chave_admin_pesquisa:
    raise ValueError("As variáveis de ambiente AZURE_SEARCH_SERVICE_NAME e AZURE_SEARCH_ADMIN_KEY não estão definidas.")

endpoint_pesquisa = f"https://{nome_servico_pesquisa}.search.windows.net/"
credencial = AzureKeyCredential(chave_admin_pesquisa)
cliente_pesquisa = SearchClient(endpoint=endpoint_pesquisa, index_name=nome_indice, credential=credencial)

from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField,
    SearchableField,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
    VectorSearchVectorizer,
    HnswAlgorithmConfiguration,
    AzureOpenAIEmbeddingVectorizer,
)

cliente_indice = SearchIndexClient(endpoint=endpoint_pesquisa, credential=credencial)
nome_vetorizador = "vetorizador-openai"

campos = [
    SimpleField(name="id", type="Edm.String", key=True),
    SearchableField(name="conteudo", type="Edm.String"),
    SimpleField(name="nome_arquivo", type="Edm.String"),
    SearchField(
        name="embedding",
        type="Edm.Single[Collection]",
        vector_search_dimensions=1536,  # Dimensão do embedding do modelo ada-002
        vector_search_configuration=nome_vetorizador,
    ),
]

configuracao_vetor_hnsw = HnswAlgorithmConfiguration(name="config-hnsw")
configuracao_vetorizador_openai = AzureOpenAIEmbeddingVectorizer(
    name=nome_vetorizador,
    azure_open_ai_service_endpoint=endpoint_openai,
    azure_open_ai_service_key=chave_openai,
    azure_open_ai_deployment_id=modelo_embedding, # Usamos o mesmo deployment para embedding
)
busca_vetorial = VectorSearch(
    algorithms=[configuracao_vetor_hnsw],
    vectorizers=[configuracao_vetorizador_openai],
)

definicao_indice = {
    "name": nome_indice,
    "fields": campos,
    "vector_search": busca_vetorial,
}

try:
    result = cliente_indice.create_index(definicao_indice)
    print(f"Índice '{nome_indice}' criado com sucesso.")
except Exception as e:
    print(f"Erro ao criar o índice '{nome_indice}': {e}")

def adicionar_embeddings_ao_indice(embeddings: dict, textos: dict, cliente_pesquisa: SearchClient):
    documentos = []
    for nome_arquivo, embedding in embeddings.items():
        # Assumindo que 'textos' também tem o texto completo por nome de arquivo
        # Se você chunkificou, precisará adaptar isso para cada chunk
        documento = {
            "id": nome_arquivo,  # Um identificador único para o documento
            "conteudo": textos[nome_arquivo],
            "nome_arquivo": nome_arquivo,
            "embedding": embedding,
        }
        documentos.append(documento)

    try:
        result = cliente_pesquisa.upload_documents(documents=documentos)
        print(f"Documentos indexados com sucesso: {result[0].succeeded_count}")
    except Exception as e:
        print(f"Erro ao indexar documentos: {e}")

adicionar_embeddings_ao_indice(embeddings_dos_pdfs, textos_extraidos, cliente_pesquisa)


def gerar_embedding_pergunta(pergunta: str, cliente: OpenAIClient, modelo: str):
    try:
        resposta = cliente.embeddings.create(input=pergunta, model=modelo)
        return resposta.data[0].embedding
    except Exception as e:
        print(f"Erro ao gerar embedding da pergunta: {e}")
        return None


def buscar_documentos_relevantes(cliente_pesquisa: SearchClient, embedding_pergunta: list[float], top_n: int = 3):
    try:
        resultados = cliente_pesquisa.search(
            search_vector=Vector(value=embedding_pergunta, k=top_n, fields="embedding"),
            select=["conteudo", "nome_arquivo"],  # Quais campos queremos retornar
        )
        documentos_encontrados = [doc for doc in resultados]
        return documentos_encontrados
    except Exception as e:
        print(f"Erro ao buscar documentos: {e}")
        return []


def gerar_resposta(pergunta: str, documentos_relevantes: list[dict], cliente: OpenAIClient, modelo_llm: str = "gpt-35-turbo"):
    contexto = "\n\n".join([f"Conteúdo do documento '{doc['nome_arquivo']}':\n{doc['conteudo']}" for doc in documentos_relevantes])
    prompt = f"""Você é um assistente virtual que responde perguntas com base no contexto fornecido.
    Use apenas as informações contidas nos documentos a seguir para responder à pergunta.
    Se a resposta não puder ser encontrada nos documentos, responda que você não sabe.

    Contexto:
    {contexto}

    Pergunta: {pergunta}

    Resposta:"""

    try:
        resposta = cliente.chat.completions.create(
            model=modelo_llm,
            messages=[
                {"role": "system", "content": "Você é um assistente virtual útil baseado em documentos."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,  # Ajuste conforme necessário
            temperature=0.2, # Ajuste para controlar a aleatoriedade da resposta
        )
        return resposta.choices[0].message.content
    except Exception as e:
        print(f"Erro ao gerar resposta: {e}")
        return "Ocorreu um erro ao gerar a resposta."


pergunta_teste = "Qual foi a principal metodologia utilizada no artigo X?"
embedding_pergunta = gerar_embedding_pergunta(pergunta_teste, cliente, modelo_embedding)

if embedding_pergunta:
    documentos_encontrados = buscar_documentos_relevantes(cliente_pesquisa, embedding_pergunta)
    if documentos_encontrados:
        resposta = gerar_resposta(pergunta_teste, documentos_encontrados, cliente)
        print(f"Pergunta: {pergunta_teste}")
        print(f"Resposta: {resposta}")
    else:
        print("Nenhum documento relevante encontrado.")
else:
    print("Não foi possível gerar o embedding da pergunta.")

## INstalar a Biblioteca:
pip install streamlit

import streamlit as st
import os
from azure.core.credentials import AzureKeyCredential
from azure.openai import OpenAIClient
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector

# --- Configurações do Azure ---
chave_openai = os.environ.get("AZURE_OPENAI_KEY")
endpoint_openai = os.environ.get("AZURE_OPENAI_ENDPOINT")
nome_servico_pesquisa = os.environ.get("AZURE_SEARCH_SERVICE_NAME")
chave_admin_pesquisa = os.environ.get("AZURE_SEARCH_ADMIN_KEY")
nome_indice = "meu-indice-de-pdfs"  # Certifique-se de usar o mesmo nome do seu índice

if not chave_openai or not endpoint_openai or not nome_servico_pesquisa or not chave_admin_pesquisa:
    st.error("Variáveis de ambiente do Azure não configuradas.")
    st.stop()

cliente_openai = OpenAIClient(endpoint=endpoint_openai, credential=AzureKeyCredential(chave_openai))
endpoint_pesquisa = f"https://{nome_servico_pesquisa}.search.windows.net/"
credencial_pesquisa = AzureKeyCredential(chave_admin_pesquisa)
cliente_pesquisa = SearchClient(endpoint=endpoint_pesquisa, index_name=nome_indice, credential=credencial_pesquisa)
modelo_embedding = "text-embedding-ada-002"
modelo_llm = "gpt-35-turbo"  # Ou outro modelo de sua preferência

# --- Funções (as mesmas que criamos antes) ---
def gerar_embedding_pergunta(pergunta: str, cliente: OpenAIClient, modelo: str):
    try:
        resposta = cliente.embeddings.create(input=pergunta, model=modelo)
        return resposta.data[0].embedding
    except Exception as e:
        st.error(f"Erro ao gerar embedding da pergunta: {e}")
        return None

def buscar_documentos_relevantes(cliente_pesquisa: SearchClient, embedding_pergunta: list[float], top_n: int = 3):
    try:
        resultados = cliente_pesquisa.search(
            search_vector=Vector(value=embedding_pergunta, k=top_n, fields="embedding"),
            select=["conteudo", "nome_arquivo"],
        )
        documentos_encontrados = [doc for doc in resultados]
        return documentos_encontrados
    except Exception as e:
        st.error(f"Erro ao buscar documentos: {e}")
        return []

def gerar_resposta(pergunta: str, documentos_relevantes: list[dict], cliente: OpenAIClient, modelo_llm: str = "gpt-35-turbo"):
    contexto = "\n\n".join([f"Conteúdo do documento '{doc['nome_arquivo']}':\n{doc['conteudo']}" for doc in documentos_relevantes])
    prompt = f"""Você é um assistente virtual que responde perguntas com base no contexto fornecido.
    Use apenas as informações contidas nos documentos a seguir para responder à pergunta.
    Se a resposta não puder ser encontrada nos documentos, responda que você não sabe.

    Contexto:
    {contexto}

    Pergunta: {pergunta}

    Resposta:"""

    try:
        resposta = cliente.chat.completions.create(
            model=modelo_llm,
            messages=[
                {"role": "system", "content": "Você é um assistente virtual útil baseado em documentos."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.2,
        )
        return resposta.choices[0].message.content
    except Exception as e:
        st.error(f"Erro ao gerar resposta: {e}")
        return "Ocorreu um erro ao gerar a resposta."

# --- Interface do Streamlit ---
st.title("Chat Interativo com seus PDFs")
pergunta = st.text_input("Faça sua pergunta:")

if pergunta:
    with st.spinner("Processando sua pergunta..."):
        embedding_pergunta = gerar_embedding_pergunta(pergunta, cliente_openai, modelo_embedding)
        if embedding_pergunta:
            documentos_relevantes = buscar_documentos_relevantes(cliente_pesquisa, embedding_pergunta)
            if documentos_relevantes:
                resposta = gerar_resposta(pergunta, documentos_relevantes, cliente_openai, modelo_llm)
                st.subheader("Resposta:")
                st.markdown(resposta)
                with st.expander("Documentos Relevantes"):
                    for doc in documentos_relevantes:
                        st.markdown(f"**{doc['nome_arquivo']}**: {doc['conteudo'][:500]}...") # Exibe os primeiros 500 caracteres
            else:
                st.warning("Nenhum documento relevante encontrado para sua pergunta.")
        else:
            st.error("Não foi possível gerar o embedding da pergunta.")


## Salve o arquivo app_chat.py e execute-o a partir do seu terminal usando o seguinte comando.
streamlit run app_chat.py
