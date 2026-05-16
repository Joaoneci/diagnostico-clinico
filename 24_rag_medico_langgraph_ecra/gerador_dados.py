import os
from sqlalchemy import create_engine, text
import chromadb
from chromadb.config import Settings
import time
import getpass

# Fallback for API Key if needed for embeddings (assuming we use Gemini or just basic Chroma default)
if 'GOOGLE_API_KEY' not in os.environ:
    os.environ['GOOGLE_API_KEY'] = 'fake-key-for-local' # Not strictly needed if using sentence-transformers, but good for gemini embeddings

DB_URL = os.environ.get('DATABASE_URL', 'postgresql://medico:medico@localhost:5432/clinicadb')

pacientes_data = [
    {"nome": "Maria Silva", "idade": 65, "genero": "F", "dialogo": [
        ("medico", "Olá Maria, como se tem sentido?"),
        ("paciente", "Doutor, tenho tido muitas dores de cabeça e sinto o coração muito acelerado ultimamente."),
        ("medico", "Quando ocorrem essas palpitações?"),
        ("paciente", "Principalmente quando me deito. E também sinto um pouco de falta de ar.")
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
        ("paciente", "Há cerca de 6 meses, mas está a piorar.")
    ]},
    {"nome": "Rui Santos", "idade": 70, "genero": "M", "dialogo": [
        ("medico", "Sr. Rui, a sua tosse continua?"),
        ("paciente", "Sim doutor, tosse com expetoração escura. Sou fumador há 50 anos."),
        ("medico", "Notou sangue na expetoração?"),
        ("paciente", "Sim, ontem notei umas manchas de sangue.")
    ]},
    {"nome": "Inês Almeida", "idade": 19, "genero": "F", "dialogo": [
        ("medico", "Inês, referiu dor abdominal. Onde se localiza?"),
        ("paciente", "Começou no meio e agora dói muito aqui no lado direito em baixo."),
        ("medico", "Teve febre ou vómitos?"),
        ("paciente", "Tive vómitos hoje de manhã e estou com 38 de febre.")
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
        ("paciente", "Aliviou um pouco, mas ainda sinto um peso enorme.")
    ]}
]

def init_db():
    print("A ligar ao PostgreSQL...")
    engine = create_engine(DB_URL)
    
    # Esperar pelo DB se necessário
    for _ in range(5):
        try:
            with engine.connect() as conn:
                break
        except:
            print("Aguardando DB...")
            time.sleep(2)

    print("A inicializar dados no PostgreSQL...")
    with engine.connect() as conn:
        # Limpar tabelas
        conn.execute(text("DELETE FROM pareceres;"))
        conn.execute(text("DELETE FROM conversas;"))
        conn.execute(text("DELETE FROM pacientes;"))
        
        chroma_client = chromadb.PersistentClient(path="./chroma_medico")
        try:
            chroma_client.delete_collection("historico_clinico")
        except:
            pass
        
        collection = chroma_client.create_collection("historico_clinico")
        
        for p in pacientes_data:
            # Inserir Paciente
            result = conn.execute(
                text("INSERT INTO pacientes (nome, idade, genero) VALUES (:nome, :idade, :genero) RETURNING id"),
                {"nome": p["nome"], "idade": p["idade"], "genero": p["genero"]}
            )
            paciente_id = result.fetchone()[0]
            
            # Inserir Conversas
            full_dialogue = ""
            for papel, msg in p["dialogo"]:
                conn.execute(
                    text("INSERT INTO conversas (paciente_id, papel, mensagem) VALUES (:pid, :papel, :msg)"),
                    {"pid": paciente_id, "papel": papel, "msg": msg}
                )
                full_dialogue += f"{papel.capitalize()}: {msg}\n"
            
            # Inserir no ChromaDB para RAG (Trecho unificado por paciente para simplificar)
            collection.add(
                documents=[full_dialogue],
                metadatas=[{"paciente_id": str(paciente_id), "nome": p["nome"]}],
                ids=[f"paciente_{paciente_id}"]
            )
            
        conn.commit()
    print("Dados gerados com sucesso!")

if __name__ == "__main__":
    init_db()
