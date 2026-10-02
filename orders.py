# ============================================
# IMPORTACOES
# ============================================

# Importar o Ollama (substitui OpenAI)
# Tenta langchain-ollama (mais novo), se nao existir usa langchain_community (deprecated)
import re

try:
    from langchain_ollama import OllamaLLM as Ollama
except ImportError:
    from langchain_community.llms import Ollama

# Importar a ferramenta decoradora (transforma funcoes em ferramentas)
from langchain_core.tools import tool

# ============================================
# CONFIGURACAO DO MODELO LLM
# ============================================

# Criar instancia do Ollama/Mistral
# base_url: endereco onde o Ollama esta rodando localmente
# model: qual modelo usar (mistral neste caso)
# verbose: mostra logs detalhados da execucao
llm = Ollama(
    base_url='http://localhost:11434',
    model="mistral",
    verbose=True
)

# ============================================
# DEFINICAO DAS FERRAMENTAS
# ============================================

# CORRECAO: Usar o decorator @tool do LangChain
# O @tool transforma a funcao Python em uma ferramenta que o agente consegue chamar
# A ferramenta fica "registrada" e disponivel para o LLM usar


@tool
def order_lookup(order_id: str) -> str:
    """
    Busca o status de um pedido pelo ID.

    Quando o agente quiser saber o status de um pedido,
    esta e a ferramenta que deve ser chamada.

    Args:
        order_id: O ID do pedido a buscar (ex: "123")

    Returns:
        Uma string com o status do pedido
    """
    # Simula uma busca em banco de dados
    # Em uma aplicacao real, aqui haveria uma conexao com BD
    return f"Pedido {order_id} esta em transito."


# Coloca a ferramenta em uma lista para o agente usar
tools = [order_lookup]

# ============================================
# EXECUCAO SIMPLIFICADA
# ============================================

# VERSAO SIMPLES (compativel com LangChain 1.4.3)
#
# Nota: Versoes antigas do LangChain tinham create_react_agent e AgentExecutor
# que facilitavam a criacao de agentes. Na versao 1.4.3, foi refatorado.
#
# Aqui fazemos de forma manual:
# 1. Chamamos o LLM com informacao sobre as ferramentas
# 2. O LLM retorna o ID do pedido
# 3. Chamamos a ferramenta com esse ID
# 4. Retornamos o resultado

print("=" * 50)
print("AGENTE COM OLLAMA/MISTRAL")
print("=" * 50 + "\n")

# Pergunta do usuario
user_input = "Qual o status do pedido 123?"
print(f"Usuario: {user_input}\n")

# Criar prompt que instrua o LLM sobre as ferramentas disponiveis
# Esta e a tecnica de "function calling" - informar ao LLM quais funcoes tem
system_prompt = f"""Voce e um assistente inteligente que ajuda com pedidos.

FERRAMENTAS DISPONIVEIS:
- order_lookup(order_id): Busca o status de um pedido

INSTRUCOES:
- Responda com APENAS o numero do pedido extraido da pergunta
- Nada de explicacoes extras, APENAS o numero

Pergunta do usuario: {user_input}"""

# Chamar o LLM
print("Processando com Ollama/Mistral...")
llm_response = llm.invoke(system_prompt)
print(f"Resposta do LLM: {llm_response}\n")

# Extrair o ID do pedido da resposta
# Procura por qualquer numero que aparecer na resposta
# Pode ser "123", "ID: 123", "Pedido 123", "ID_DO_PEDIDO: 123", etc
# Os parenteses ( ) criam um "grupo de captura" que permite extrair o numero
match = re.search(r'(\d+)', llm_response)

if match:
    order_id = match.group(1)
    print(f"ID do pedido encontrado: {order_id}\n")

    # Chamar a ferramenta com o ID extraido
    print(f"Buscando informacoes do pedido {order_id}...")
    result = order_lookup.invoke({"order_id": order_id})

    print(f"Resultado: {result}\n")
else:
    print(f"Nao consegui extrair o ID do pedido da resposta\n")
    result = "Desculpe, nao consegui processar sua pergunta sobre um pedido."

print("=" * 50)
print("FIM DA EXECUCAO")
print("=" * 50)
