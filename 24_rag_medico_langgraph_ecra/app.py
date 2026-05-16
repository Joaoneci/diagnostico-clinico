import streamlit as st
import os
from sqlalchemy import create_engine, text
from agentes_medicos import graph

st.set_page_config(page_title="RAG Médico - Pareceres", page_icon="🏥", layout="wide")

DB_URL = os.environ.get('DATABASE_URL', 'postgresql://medico:medico@localhost:5432/clinicadb')
engine = create_engine(DB_URL)

st.title("🏥 RAG Médico - Análise de Pacientes")

# --- Obter Lista de Pacientes ---
pacientes = []
try:
    with engine.connect() as conn:
        res = conn.execute(text("SELECT id, nome FROM pacientes ORDER BY id")).fetchall()
        pacientes = [{'id': r[0], 'nome': r[1]} for r in res]
except Exception as e:
    st.warning("Aguardando a base de dados iniciar ou ser preenchida...")

if not pacientes:
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("Selecione o Paciente")
    paciente_sel = st.selectbox("Pacientes", pacientes, format_func=lambda x: f"#{x['id']} - {x['nome']}")
    
    st.info("""
    **Equipa de Agentes (LangGraph)**:
    1. **Extrator**: Obtém histórico do Postgres e ChromaDB.
    2. **Diagnóstico**: Formula hipóteses via Pydantic.
    3. **Triagem**: Avalia gravidade e riscos.
    4. **Consolidador**: Grava o relatório final.
    """)

pid = paciente_sel['id']

# --- Obter Dados do Paciente Selecionado ---
with engine.connect() as conn:
    dados_paciente = conn.execute(text("SELECT nome, idade, genero FROM pacientes WHERE id = :pid"), {"pid": pid}).fetchone()
    conversas = conn.execute(text("SELECT papel, mensagem, data_hora FROM conversas WHERE paciente_id = :pid ORDER BY id"), {"pid": pid}).fetchall()

tabs = st.tabs(["💬 Diálogo do Paciente", "📋 Relatório Médico IA"])

with tabs[0]:
    st.subheader(f"Conversa Registada: {dados_paciente[0]}")
    for c in conversas:
        if c[0] == "medico":
            st.chat_message("assistant", avatar="🧑‍⚕️").write(c[1])
        else:
            st.chat_message("user", avatar="👤").write(c[1])
            
with tabs[1]:
    st.subheader("Análise de Risco e Diagnóstico")
    if st.button("Gerar Parecer Médico (LangGraph)", type="primary"):
        with st.spinner("A equipa de Agentes está a analisar o histórico..."):
            try:
                final_state = graph.invoke({"paciente_id": pid})
                
                # Extrair os outputs estruturados que estavam no state
                gravidade = final_state['avaliacao']['gravidade']
                
                # Mostrar Gravidade com cor
                if gravidade == "Alta" or gravidade == "Crítica":
                    st.error(f"🚨 **Gravidade Avaliada:** {gravidade}")
                elif gravidade == "Média":
                    st.warning(f"⚠️ **Gravidade Avaliada:** {gravidade}")
                else:
                    st.success(f"✅ **Gravidade Avaliada:** {gravidade}")
                
                # Mostrar Hipóteses
                st.markdown("### 🩺 Hipóteses Diagnósticas")
                for h in final_state['diagnostico']['hipoteses']:
                    st.markdown(f"- {h}")
                    
                st.markdown("**Justificativa:**")
                st.info(final_state['diagnostico']['justificativa'])
                
                # Mostrar Riscos
                st.markdown("### ⚠️ Riscos Clínicos")
                for r in final_state['avaliacao']['riscos']:
                    st.markdown(f"- {r}")
                    
                st.success("Relatório gravado na base de dados com sucesso!")
            except Exception as e:
                st.error(f"Erro na execução do LangGraph: {e}")
    
    # Mostrar os relatórios já guardados no DB para este paciente
    st.divider()
    st.markdown("### Pareceres Anteriores")
    with engine.connect() as conn:
        pareceres = conn.execute(text("SELECT gravidade, hipoteses_json, riscos_json, data_hora FROM pareceres WHERE paciente_id = :pid ORDER BY id DESC"), {"pid": pid}).fetchall()
        
        if not pareceres:
            st.write("Nenhum parecer guardado ainda.")
        else:
            import json
            for p in pareceres:
                with st.expander(f"Parecer de {p[3].strftime('%Y-%m-%d %H:%M:%S')} - Gravidade: {p[0]}"):
                    st.write("**Hipóteses:**", json.loads(p[1]))
                    st.write("**Riscos:**", json.loads(p[2]))
