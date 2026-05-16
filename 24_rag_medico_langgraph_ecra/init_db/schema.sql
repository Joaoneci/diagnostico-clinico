CREATE TABLE pacientes (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    idade INT,
    genero VARCHAR(20)
);

CREATE TABLE conversas (
    id SERIAL PRIMARY KEY,
    paciente_id INT REFERENCES pacientes(id),
    papel VARCHAR(50) NOT NULL, -- 'medico', 'paciente'
    mensagem TEXT NOT NULL,
    data_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE pareceres (
    id SERIAL PRIMARY KEY,
    paciente_id INT REFERENCES pacientes(id),
    hipoteses_json TEXT,
    gravidade VARCHAR(50),
    riscos_json TEXT,
    data_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
