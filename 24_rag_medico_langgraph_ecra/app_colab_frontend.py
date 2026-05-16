import streamlit as st
import os
import json
import chromadb
from sqlalchemy import create_engine, text
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

# --- Configuração ---
st.set_page_config(page_title="Sistema Médico IA", page_icon="🩺", layout="wide")

os.environ['GOOGLE_API_KEY'] = 'AIzaSyCkN96sXtUuIvYvxPbwLA0ZuGC5k_IhjSU'

# Conectar ao Postgres que já está rodando via Docker do exercício 24
DB_URL = os.environ.get('DATABASE_URL', 'postgresql://medico:medico@localhost:5432/clinicadb')
engine = create_engine(DB_URL)
chroma_client = chromadb.PersistentClient(path="./chroma_medico")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# --- Pydantic Models ---
class Diagnostico(BaseModel):
    hipoteses: List[str] = Field(description="Lista de possíveis patologias baseadas nos sintomas e exames.")
    justificativa: str = Field(description="Justificativa clínica detalhada.")

class AvaliacaoGravidade(BaseModel):
    gravidade: str = Field(description="Nível de gravidade: 'Baixa', 'Média', 'Alta' ou 'Crítica'.")
    riscos: List[str] = Field(description="Riscos clínicos e complicações identificadas.")

# --- State ---
class MedicalState(TypedDict):
    paciente_id: int
    dados_paciente: dict
    historico_conversas: str
    documentos_rag: str
    
    avaliacao: dict
    diagnostico: dict
    
    status_validacao: str
    motivo_falha: str
    tentativas_diagnostico: int
    horas_consulta_simuladas: float
    
    custo_operacao: float
    preco_cliente: float
    relatorio_final: str

# --- Nodes ---
def extrator_node(state: MedicalState) -> MedicalState:
    pid = state['paciente_id']
    with engine.connect() as conn:
        res = conn.execute(text("SELECT nome, idade, genero FROM pacientes WHERE id = :pid"), {"pid": pid}).fetchone()
        dados = {"nome": res[0], "idade": res[1], "genero": res[2]}
        
        res_conv = conn.execute(text("SELECT papel, mensagem FROM conversas WHERE paciente_id = :pid ORDER BY id"), {"pid": pid}).fetchall()
        historico = "\n".join([f"{r[0].capitalize()}: {r[1]}" for r in res_conv])
        
    try:
        col = chroma_client.get_collection("historico_clinico")
        docs = col.query(query_texts=[historico], n_results=1, where={"paciente_id": str(pid)})
        docs_rag = docs['documents'][0][0] if docs['documents'] else ""
    except:
        docs_rag = ""
        
    return {
        "dados_paciente": dados, 
        "historico_conversas": historico, 
        "documentos_rag": docs_rag,
        "tentativas_diagnostico": 1,
        "horas_consulta_simuladas": 0.5
    }

def avaliador_node(state: MedicalState) -> MedicalState:
    llm_structured = llm.with_structured_output(AvaliacaoGravidade)
    prompt = f"Atue como triador. Analise o histórico: {state['historico_conversas']} do paciente de {state['dados_paciente']['idade']} anos. Forneça gravidade e riscos."
    resultado = llm_structured.invoke(prompt)
    return {"avaliacao": resultado.model_dump(), "horas_consulta_simuladas": state.get("horas_consulta_simuladas", 0) + 0.5}

def diagnostico_node(state: MedicalState) -> MedicalState:
    llm_structured = llm.with_structured_output(Diagnostico)
    prompt = f"Paciente: {state['dados_paciente']}. Histórico: {state['historico_conversas']}. Histórico Médico Anterior RAG: {state['documentos_rag']}. Gere hipóteses e justificativas."
    if state.get("motivo_falha"):
        prompt += f"\nCORREÇÃO NECESSÁRIA. A validação anterior falhou pelo motivo: {state['motivo_falha']}. Melhore seu diagnóstico."
    
    resultado = llm_structured.invoke(prompt)
    return {"diagnostico": resultado.model_dump(), "horas_consulta_simuladas": state.get("horas_consulta_simuladas", 0) + 1.0}

def validador_node(state: MedicalState) -> MedicalState:
    prompt = f"O nível de gravidade é {state['avaliacao']['gravidade']}. As hipóteses foram {state['diagnostico']['hipoteses']}. Este diagnóstico está coerente com os sintomas graves ou de risco {state['avaliacao']['riscos']} e com o contexto {state['historico_conversas']}? Responda APENAS 'PASS' se estiver totalmente correto, ou 'FAIL: <motivo>' se houver inconsistências graves."
    res = llm.invoke(prompt).content.strip()
    
    if res.startswith("FAIL") and state["tentativas_diagnostico"] < 3:
        return {"status_validacao": "FAIL", "motivo_falha": res}
    return {"status_validacao": "PASS"}

def corretor_node(state: MedicalState) -> MedicalState:
    return {"tentativas_diagnostico": state["tentativas_diagnostico"] + 1, "horas_consulta_simuladas": state["horas_consulta_simuladas"] + 0.5}

def contabilista_node(state: MedicalState) -> MedicalState:
    horas = state["horas_consulta_simuladas"]
    custo_hora = 25.0
    custo_operacao = horas * custo_hora
    preco_cliente = (custo_operacao * 2) + 150.0
    return {"custo_operacao": custo_operacao, "preco_cliente": preco_cliente}

def consolidador_node(state: MedicalState) -> MedicalState:
    return {"relatorio_final": "Concluído"}

def route_validacao(state: MedicalState) -> str:
    if state["status_validacao"] == "FAIL":
        return "corretor"
    return "contabilista"

# --- Build Graph ---
builder = StateGraph(MedicalState)
builder.add_node("extrator", extrator_node)
builder.add_node("avaliador", avaliador_node)
builder.add_node("diagnostico", diagnostico_node)
builder.add_node("validador", validador_node)
builder.add_node("corretor", corretor_node)
builder.add_node("contabilista", contabilista_node)
builder.add_node("consolidador", consolidador_node)

builder.add_edge(START, "extrator")
builder.add_edge("extrator", "avaliador")
builder.add_edge("avaliador", "diagnostico")
builder.add_edge("diagnostico", "validador")
builder.add_conditional_edges("validador", route_validacao, {"corretor": "corretor", "contabilista": "contabilista"})
builder.add_edge("corretor", "diagnostico")
builder.add_edge("contabilista", "consolidador")
builder.add_edge("consolidador", END)
graph = builder.compile()

# --- STREAMLIT UI ---
st.title("🩺 Plataforma de Diagnóstico IA e Operações")
st.markdown("Interface executável baseada no notebook `sistema_medico_colab.ipynb`")

# Obter lista de pacientes
try:
    with engine.connect() as conn:
        pacientes = conn.execute(text("SELECT id, nome FROM pacientes ORDER BY id")).fetchall()
except Exception as e:
    st.error(f"Erro ao ligar à base de dados. Certifique-se que executou a célula de setup no notebook ou que o docker DB está rodando. Erro: {e}")
    st.stop()

if not pacientes:
    st.warning("Nenhum paciente encontrado. Verifique se os dados sintéticos foram gerados na base de dados.")
    st.stop()

opcoes = {f"{p[1]} (ID: {p[0]})": p[0] for p in pacientes}
paciente_selecionado = st.selectbox("Selecione o Paciente para Análise:", options=list(opcoes.keys()))

if st.button("Gerar Diagnóstico IA", type="primary"):
    pid = opcoes[paciente_selecionado]
    
    with st.spinner(f"A analisar o paciente ID {pid}..."):
        initial_state = {"paciente_id": pid}
        final_state = graph.invoke(initial_state)
        
    st.success("Análise concluída com sucesso!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Dados Clínicos e Riscos")
        st.write(f"**Nome:** {final_state['dados_paciente']['nome']}")
        st.write(f"**Gravidade:** {final_state['avaliacao']['gravidade']}")
        st.write(f"**Riscos Identificados:** {', '.join(final_state['avaliacao']['riscos'])}")
        
        st.subheader("🏥 Diagnóstico")
        for h in final_state['diagnostico']['hipoteses']:
            st.markdown(f"- {h}")
        st.info(f"**Justificativa Clínica:** {final_state['diagnostico']['justificativa']}")

    with col2:
        st.subheader("💰 Análise Financeira e Custos")
        st.metric("Horas Simuladas da Equipa IA", f"{final_state['horas_consulta_simuladas']} h")
        st.metric("Tentativas de Refinamento (Correção)", final_state['tentativas_diagnostico'])
        st.metric("Custo Operacional Interno", f"R$ {final_state['custo_operacao']:.2f}", help="Cálculo a R$ 25,00/h")
        st.metric("Preço Sugerido ao Cliente", f"R$ {final_state['preco_cliente']:.2f}")

