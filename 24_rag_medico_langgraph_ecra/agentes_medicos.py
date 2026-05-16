import os
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
import chromadb
from sqlalchemy import create_engine, text
import json

# Environment
DB_URL = os.environ.get('DATABASE_URL', 'postgresql://medico:medico@localhost:5432/clinicadb')
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
chroma_client = chromadb.PersistentClient(path="./chroma_medico")

# --- Pydantic Models ---
class Diagnostico(BaseModel):
    hipoteses: List[str] = Field(description="Lista de possíveis patologias baseadas nos sintomas.")
    justificativa: str = Field(description="Justificativa clínica para as hipóteses.")

class AvaliacaoGravidade(BaseModel):
    gravidade: str = Field(description="Nível de gravidade: 'Baixa', 'Média', 'Alta' ou 'Crítica'.")
    riscos: List[str] = Field(description="Riscos clínicos identificados no paciente.")

# --- State ---
class MedicalState(TypedDict):
    paciente_id: int
    dados_paciente: dict
    historico_conversas: str
    documentos_rag: str
    diagnostico: dict # from Pydantic
    avaliacao: dict   # from Pydantic
    relatorio_final: str

# --- Nodes ---
def extrator_node(state: MedicalState) -> MedicalState:
    pid = state['paciente_id']
    
    # 1. Obter dados estruturados do Postgres
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        res_paciente = conn.execute(text("SELECT nome, idade, genero FROM pacientes WHERE id = :pid"), {"pid": pid}).fetchone()
        dados = {"nome": res_paciente[0], "idade": res_paciente[1], "genero": res_paciente[2]}
        
        res_conv = conn.execute(text("SELECT papel, mensagem FROM conversas WHERE paciente_id = :pid ORDER BY id"), {"pid": pid}).fetchall()
        historico = "\n".join([f"{r[0].capitalize()}: {r[1]}" for r in res_conv])
        
    # 2. Obter contexto RAG do ChromaDB (mesmo que seja redundante com as conversas, demonstra RAG vetorial)
    try:
        collection = chroma_client.get_collection("historico_clinico")
        results = collection.query(
            query_texts=[historico],
            n_results=1,
            where={"paciente_id": str(pid)}
        )
        docs_rag = results['documents'][0][0] if results['documents'] else ""
    except:
        docs_rag = ""
        
    return {"dados_paciente": dados, "historico_conversas": historico, "documentos_rag": docs_rag}

def gerador_hipoteses_node(state: MedicalState) -> MedicalState:
    # Usar output estruturado (Pydantic)
    llm_structured = llm.with_structured_output(Diagnostico)
    
    prompt = f"""
    Atue como um médico diagnosticador.
    Paciente: {state['dados_paciente']['nome']}, {state['dados_paciente']['idade']} anos, Gênero {state['dados_paciente']['genero']}.
    Histórico de Conversas e Queixas:
    {state['historico_conversas']}
    
    Gere as hipóteses diagnósticas para este quadro clínico.
    """
    
    resultado: Diagnostico = llm_structured.invoke(prompt)
    
    return {"diagnostico": resultado.model_dump()}

def avaliador_gravidade_node(state: MedicalState) -> MedicalState:
    llm_structured = llm.with_structured_output(AvaliacaoGravidade)
    
    prompt = f"""
    Atue como um médico de triagem clínica.
    Avaliando as queixas: {state['historico_conversas']}
    E as hipóteses: {state['diagnostico']['hipoteses']}
    
    Determine o nível de gravidade e os principais riscos associados a este paciente.
    """
    
    resultado: AvaliacaoGravidade = llm_structured.invoke(prompt)
    
    return {"avaliacao": resultado.model_dump()}

def consolidador_node(state: MedicalState) -> MedicalState:
    pid = state['paciente_id']
    diag = state['diagnostico']
    aval = state['avaliacao']
    
    relatorio = f"""
    ### Parecer Médico Consolidado
    **Paciente:** {state['dados_paciente']['nome']}
    **Gravidade:** {aval['gravidade']}
    
    **Hipóteses Diagnósticas:**
    {chr(10).join(['- ' + h for h in diag['hipoteses']])}
    
    **Riscos Identificados:**
    {chr(10).join(['- ' + r for r in aval['riscos']])}
    
    **Justificativa Clínica:**
    {diag['justificativa']}
    """
    
    # Guardar no banco de dados Postgres
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO pareceres (paciente_id, hipoteses_json, gravidade, riscos_json)
            VALUES (:pid, :hipoteses, :gravidade, :riscos)
        """), {
            "pid": pid,
            "hipoteses": json.dumps(diag['hipoteses']),
            "gravidade": aval['gravidade'],
            "riscos": json.dumps(aval['riscos'])
        })
        conn.commit()
        
    return {"relatorio_final": relatorio}

# --- Build Graph ---
builder = StateGraph(MedicalState)
builder.add_node("extrator", extrator_node)
builder.add_node("diagnostico", gerador_hipoteses_node)
builder.add_node("gravidade", avaliador_gravidade_node)
builder.add_node("consolidador", consolidador_node)

builder.add_edge(START, "extrator")
builder.add_edge("extrator", "diagnostico")
builder.add_edge("diagnostico", "gravidade")
builder.add_edge("gravidade", "consolidador")
builder.add_edge("consolidador", END)

graph = builder.compile()
