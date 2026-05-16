import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""# Sistema Médico de Diagnóstico IA - LangGraph

Neste notebook, implementamos um sistema de diagnóstico médico usando **LangGraph**, **LangChain**, e **Pydantic**.
O sistema usa um banco **PostgreSQL** para armazenar os dados clínicos dos pacientes e **ChromaDB** para busca de histórico via RAG.
A orquestração conta com 6 agentes: Orquestrador, Avaliador de Gravidade, Gerador Diagnóstico, Validador, Corretor e Contabilista."""))

cells.append(nbf.v4.new_markdown_cell("## 1. Instalação e Configuração de Dependências e Banco de Dados (Postgres)"))

cells.append(nbf.v4.new_code_cell("""# Instalando bibliotecas python
!pip install -q langchain langgraph langchain-google-genai pydantic chromadb sqlalchemy psycopg2-binary nest_asyncio ipywidgets

# Instalando PostgreSQL no ambiente Colab
!apt-get update > /dev/null
!apt-get install -y postgresql postgresql-contrib > /dev/null
!service postgresql start
!sudo -u postgres psql -c "CREATE USER medico WITH PASSWORD 'medico';"
!sudo -u postgres psql -c "CREATE DATABASE clinicadb OWNER medico;"

import nest_asyncio
nest_asyncio.apply()"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Configuração de Ambiente e Imports"))

cells.append(nbf.v4.new_code_cell("""import os
import getpass
import json
import chromadb
from sqlalchemy import create_engine, text
from typing import TypedDict, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
import ipywidgets as widgets
from IPython.display import display, clear_output

if 'GOOGLE_API_KEY' not in os.environ:
    print("Insira a sua GOOGLE_API_KEY (Gemini):")
    os.environ['GOOGLE_API_KEY'] = getpass.getpass()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

DB_URL = 'postgresql://medico:medico@localhost:5432/clinicadb'
engine = create_engine(DB_URL)
chroma_client = chromadb.PersistentClient(path="./chroma_medico")
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Inicialização e Geração de Pacientes Sintéticos (10 Pacientes)"))

cells.append(nbf.v4.new_code_cell("""def init_db():
    print("A inicializar base de dados PostgreSQL e ChromaDB...")
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS pareceres;"))
        conn.execute(text("DROP TABLE IF EXISTS conversas;"))
        conn.execute(text("DROP TABLE IF EXISTS pacientes;"))
        
        conn.execute(text('''
            CREATE TABLE pacientes (
                id SERIAL PRIMARY KEY,
                nome VARCHAR(100),
                idade INT,
                genero VARCHAR(10)
            )
        '''))
        conn.execute(text('''
            CREATE TABLE conversas (
                id SERIAL PRIMARY KEY,
                paciente_id INT REFERENCES pacientes(id),
                papel VARCHAR(50),
                mensagem TEXT
            )
        '''))
        conn.execute(text('''
            CREATE TABLE pareceres (
                id SERIAL PRIMARY KEY,
                paciente_id INT REFERENCES pacientes(id),
                hipoteses_json TEXT,
                gravidade VARCHAR(50),
                riscos_json TEXT,
                custo_operacao FLOAT,
                preco_cliente FLOAT,
                horas_simuladas FLOAT
            )
        '''))
        conn.commit()
    
    try:
        chroma_client.delete_collection("historico_clinico")
    except Exception:
        pass
    collection = chroma_client.create_collection("historico_clinico")
    
    pacientes_data = [
        {"nome": "Maria Silva", "idade": 65, "genero": "F", "dialogo": [
            ("medico", "Olá Maria, como se tem sentido?"),
            ("paciente", "Doutor, tenho tido muitas dores de cabeça e sinto o coração muito acelerado ultimamente."),
            ("medico", "Quando ocorrem essas palpitações?"),
            ("paciente", "Principalmente quando me deito. E também sinto um pouco de falta de ar e vi no meu laudo anterior de ECG algumas anomalias.")
        ]},
        {"nome": "João Fernandes", "idade": 45, "genero": "M", "dialogo": [
            ("medico", "João, os seus exames mostram glicose elevada."),
            ("paciente", "Eu tenho sentido muita sede e vou à casa de banho várias vezes à noite."),
            ("medico", "Alguma perda de peso não intencional?"),
            ("paciente", "Sim, perdi uns 4kg este mês sem mudar a dieta.")
        ]},
        {"nome": "Ana Pereira", "idade": 28, "genero": "F", "dialogo": [
            ("medico", "Ana, que queixas a trazem cá hoje?"),
            ("paciente", "Tenho sentido uma tristeza profunda e não tenho vontade de fazer nada há meses."),
            ("medico", "Como está o seu sono?"),
            ("paciente", "Acordo às 3 da manhã e não consigo voltar a dormir.")
        ]},
        {"nome": "Carlos Mendes", "idade": 55, "genero": "M", "dialogo": [
            ("medico", "Carlos, a sua pressão arterial está 160/100."),
            ("paciente", "Eu não sinto nada, doutor. Mas o meu pai faleceu de enfarte aos 60."),
            ("medico", "Tem tomado a medicação?"),
            ("paciente", "Esqueço-me algumas vezes, para ser sincero.")
        ]},
        {"nome": "Sofia Costa", "idade": 35, "genero": "F", "dialogo": [
            ("medico", "Diga-me, Sofia, o que se passa?"),
            ("paciente", "Tenho dores fortes nas articulações das mãos, principalmente de manhã. Ficam rígidas."),
            ("medico", "Há quanto tempo tem estes sintomas?"),
            ("paciente", "Há cerca de 6 meses, mas está a piorar. Os exames reumatológicos indicam fator reumatoide positivo.")
        ]},
        {"nome": "Rui Santos", "idade": 70, "genero": "M", "dialogo": [
            ("medico", "Sr. Rui, a sua tosse continua?"),
            ("paciente", "Sim doutor, tosse com expetoração escura. Sou fumador há 50 anos."),
            ("medico", "Notou sangue na expetoração?"),
            ("paciente", "Sim, ontem notei umas manchas de sangue no lenço.")
        ]},
        {"nome": "Inês Almeida", "idade": 19, "genero": "F", "dialogo": [
            ("medico", "Inês, referiu dor abdominal. Onde se localiza?"),
            ("paciente", "Começou no meio e agora dói muito aqui no lado direito em baixo."),
            ("medico", "Teve febre ou vómitos?"),
            ("paciente", "Tive vómitos hoje de manhã e estou com 38 de febre. O hemograma aponta leucocitose.")
        ]},
        {"nome": "Pedro Oliveira", "idade": 82, "genero": "M", "dialogo": [
            ("medico", "Como está a sua memória, Sr. Pedro?"),
            ("paciente", "O meu filho diz que me esqueço de apagar o fogão. Eu acho que é da idade."),
            ("medico", "Teve algum episódio de se perder num local conhecido?"),
            ("paciente", "Sim, na semana passada perdi-me a ir para o café.")
        ]},
        {"nome": "Catarina Lima", "idade": 41, "genero": "F", "dialogo": [
            ("medico", "Catarina, os nódulos na tiroide aumentaram."),
            ("paciente", "Tenho sentido muito cansaço, queda de cabelo e frio constante."),
            ("medico", "O seu peso alterou-se?"),
            ("paciente", "Ganhei 10kg no último ano.")
        ]},
        {"nome": "Tiago Rocha", "idade": 33, "genero": "M", "dialogo": [
            ("medico", "Tiago, o que o trouxe às urgências?"),
            ("paciente", "Senti uma dor no peito muito forte a jogar futebol, que irradiou para o braço esquerdo."),
            ("medico", "Ainda tem a dor agora?"),
            ("paciente", "Aliviou um pouco, mas ainda sinto um peso enorme. O exame de troponina veio alterado.")
        ]}
    ]
    
    with engine.connect() as conn:
        for p in pacientes_data:
            result = conn.execute(
                text("INSERT INTO pacientes (nome, idade, genero) VALUES (:nome, :idade, :genero) RETURNING id"),
                {"nome": p["nome"], "idade": p["idade"], "genero": p["genero"]}
            )
            paciente_id = result.fetchone()[0]
            
            full_dialogue = ""
            for papel, msg in p["dialogo"]:
                conn.execute(
                    text("INSERT INTO conversas (paciente_id, papel, mensagem) VALUES (:pid, :papel, :msg)"),
                    {"pid": paciente_id, "papel": papel, "msg": msg}
                )
                full_dialogue += f"{papel.capitalize()}: {msg}\\n"
            
            collection.add(
                documents=[full_dialogue],
                metadatas=[{"paciente_id": str(paciente_id), "nome": p["nome"]}],
                ids=[f"paciente_{paciente_id}"]
            )
        conn.commit()
    print("Base de dados populada com 10 pacientes!")

init_db()"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Definição do Grafo (State & Models)"))

cells.append(nbf.v4.new_code_cell("""class Diagnostico(BaseModel):
    hipoteses: List[str] = Field(description="Lista de possíveis patologias baseadas nos sintomas e exames.")
    justificativa: str = Field(description="Justificativa clínica detalhada.")

class AvaliacaoGravidade(BaseModel):
    gravidade: str = Field(description="Nível de gravidade: 'Baixa', 'Média', 'Alta' ou 'Crítica'.")
    riscos: List[str] = Field(description="Riscos clínicos e complicações identificadas.")

class MedicalState(TypedDict):
    paciente_id: int
    dados_paciente: dict
    historico_conversas: str
    documentos_rag: str
    
    avaliacao: dict
    diagnostico: dict
    
    status_validacao: str # 'PASS' ou 'FAIL'
    motivo_falha: str
    tentativas_diagnostico: int
    horas_consulta_simuladas: float
    
    custo_operacao: float
    preco_cliente: float
    relatorio_final: str
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Agentes Médicos e Financeiros"))

cells.append(nbf.v4.new_code_cell("""# 1. Orquestrador / Extrator
def extrator_node(state: MedicalState) -> MedicalState:
    pid = state['paciente_id']
    with engine.connect() as conn:
        res = conn.execute(text("SELECT nome, idade, genero FROM pacientes WHERE id = :pid"), {"pid": pid}).fetchone()
        dados = {"nome": res[0], "idade": res[1], "genero": res[2]}
        
        res_conv = conn.execute(text("SELECT papel, mensagem FROM conversas WHERE paciente_id = :pid ORDER BY id"), {"pid": pid}).fetchall()
        historico = "\\n".join([f"{r[0].capitalize()}: {r[1]}" for r in res_conv])
        
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
        "horas_consulta_simuladas": 0.5 # Tempo inicial mínimo
    }

# 2. Avaliador de Gravidade
def avaliador_node(state: MedicalState) -> MedicalState:
    llm_structured = llm.with_structured_output(AvaliacaoGravidade)
    prompt = f"Atue como triador. Analise o histórico: {state['historico_conversas']} do paciente de {state['dados_paciente']['idade']} anos. Forneça gravidade e riscos."
    resultado = llm_structured.invoke(prompt)
    return {"avaliacao": resultado.model_dump(), "horas_consulta_simuladas": state.get("horas_consulta_simuladas", 0) + 0.5}

# 3. Gerador Diagnóstico
def diagnostico_node(state: MedicalState) -> MedicalState:
    llm_structured = llm.with_structured_output(Diagnostico)
    prompt = f"Paciente: {state['dados_paciente']}. Histórico: {state['historico_conversas']}. Histórico Médico Anterior RAG: {state['documentos_rag']}. Gere hipóteses e justificativas."
    if state.get("motivo_falha"):
        prompt += f"\\nCORREÇÃO NECESSÁRIA. A validação anterior falhou pelo motivo: {state['motivo_falha']}. Melhore seu diagnóstico."
    
    resultado = llm_structured.invoke(prompt)
    return {"diagnostico": resultado.model_dump(), "horas_consulta_simuladas": state.get("horas_consulta_simuladas", 0) + 1.0}

# 4. Validador Diagnóstico
def validador_node(state: MedicalState) -> MedicalState:
    prompt = f"O nível de gravidade é {state['avaliacao']['gravidade']}. As hipóteses foram {state['diagnostico']['hipoteses']}. Este diagnóstico está coerente com os sintomas graves ou de risco {state['avaliacao']['riscos']} e com o contexto {state['historico_conversas']}? Responda APENAS 'PASS' se estiver totalmente correto, ou 'FAIL: <motivo>' se houver inconsistências graves, omissões de doenças mortais (como IAM, Sepse, AVC se sintomas presentes)."
    res = llm.invoke(prompt).content.strip()
    
    if res.startswith("FAIL") and state["tentativas_diagnostico"] < 3:
        return {"status_validacao": "FAIL", "motivo_falha": res}
    return {"status_validacao": "PASS"}

# 5. Corretor
def corretor_node(state: MedicalState) -> MedicalState:
    # Apenas incrementa as tentativas e horas, o roteamento o mandará de volta pro diagnóstico
    return {"tentativas_diagnostico": state["tentativas_diagnostico"] + 1, "horas_consulta_simuladas": state["horas_consulta_simuladas"] + 0.5}

# 6. Contabilista / Custos
def contabilista_node(state: MedicalState) -> MedicalState:
    horas = state["horas_consulta_simuladas"]
    custo_hora = 25.0
    custo_operacao = horas * custo_hora
    
    # Preço ao cliente: margem de 100% sobre o custo + taxa base de tecnologia
    preco_cliente = (custo_operacao * 2) + 150.0
    
    return {"custo_operacao": custo_operacao, "preco_cliente": preco_cliente}

# 7. Consolidador
def consolidador_node(state: MedicalState) -> MedicalState:
    pid = state['paciente_id']
    diag = state['diagnostico']
    aval = state['avaliacao']
    
    relatorio = f"**Relatório Clínico e Financeiro**\\n"
    relatorio += f"- **Paciente:** {state['dados_paciente']['nome']}\\n"
    relatorio += f"- **Gravidade:** {aval['gravidade']}\\n"
    relatorio += f"- **Hipóteses:** {', '.join(diag['hipoteses'])}\\n"
    relatorio += f"- **Horas Empregadas (IA):** {state['horas_consulta_simuladas']}h\\n"
    relatorio += f"- **Custo Empresa:** R$ {state['custo_operacao']:.2f}\\n"
    relatorio += f"- **Valor Cobrado:** R$ {state['preco_cliente']:.2f}\\n"
    
    with engine.connect() as conn:
        conn.execute(text('''
            INSERT INTO pareceres (paciente_id, hipoteses_json, gravidade, riscos_json, custo_operacao, preco_cliente, horas_simuladas)
            VALUES (:pid, :hipoteses, :gravidade, :riscos, :custo, :preco, :horas)
        '''), {
            "pid": pid,
            "hipoteses": json.dumps(diag['hipoteses']),
            "gravidade": aval['gravidade'],
            "riscos": json.dumps(aval['riscos']),
            "custo": state["custo_operacao"],
            "preco": state["preco_cliente"],
            "horas": state["horas_consulta_simuladas"]
        })
        conn.commit()
    
    return {"relatorio_final": relatorio}
"""))

cells.append(nbf.v4.new_markdown_cell("## 6. Construção do Grafo"))

cells.append(nbf.v4.new_code_cell("""def route_validacao(state: MedicalState) -> str:
    if state["status_validacao"] == "FAIL":
        return "corretor"
    return "contabilista"

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

# Corretor envia de volta ao diagnostico para refazer
builder.add_edge("corretor", "diagnostico")
builder.add_edge("contabilista", "consolidador")
builder.add_edge("consolidador", END)

graph = builder.compile()
"""))

cells.append(nbf.v4.new_markdown_cell("## 7. Frontend e Interação (ipywidgets)"))

cells.append(nbf.v4.new_code_cell("""# Obter pacientes
with engine.connect() as conn:
    pacientes = conn.execute(text("SELECT id, nome FROM pacientes ORDER BY id")).fetchall()

dropdown_pacientes = widgets.Dropdown(
    options=[(f"{p[1]} (ID: {p[0]})", p[0]) for p in pacientes],
    description='Paciente:',
    style={'description_width': 'initial'}
)

btn_executar = widgets.Button(
    description='Gerar Diagnóstico IA',
    button_style='success',
    icon='user-md'
)

out_resultados = widgets.Output()

def on_button_clicked(b):
    with out_resultados:
        clear_output()
        pid = dropdown_pacientes.value
        print(f"🔄 A analisar Paciente ID: {pid} ... Por favor aguarde.\\n")
        
        initial_state = {"paciente_id": pid}
        final_state = graph.invoke(initial_state)
        
        print("✅ ANÁLISE CONCLUÍDA\\n")
        print("====== DADOS CLÍNICOS E RISCOS ======")
        print(f"Nome: {final_state['dados_paciente']['nome']}")
        print(f"Gravidade: {final_state['avaliacao']['gravidade']}")
        print(f"Riscos: {', '.join(final_state['avaliacao']['riscos'])}")
        print("\\n====== DIAGNÓSTICO ======")
        print(f"Hipóteses: {', '.join(final_state['diagnostico']['hipoteses'])}")
        print(f"Justificativa: {final_state['diagnostico']['justificativa']}")
        print("\\n====== ANÁLISE FINANCEIRA E DE CUSTOS ======")
        print(f"Horas Simuladas da Equipa IA: {final_state['horas_consulta_simuladas']}h")
        print(f"Tentativas de Refinamento: {final_state['tentativas_diagnostico']}")
        print(f"Custo Operacional Interno (R$ 25/h): R$ {final_state['custo_operacao']:.2f}")
        print(f"Preço Sugerido ao Cliente: R$ {final_state['preco_cliente']:.2f}")

btn_executar.on_click(on_button_clicked)

display(widgets.VBox([
    widgets.HTML("<h2>🩺 Plataforma de Diagnóstico IA</h2>"),
    dropdown_pacientes, 
    btn_executar, 
    out_resultados
]))
"""))

nb['cells'] = cells

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sistema_medico_colab.ipynb')
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook '{output_path}' gerado com sucesso!")
