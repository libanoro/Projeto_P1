import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
from pathlib import Path
from datetime import date, time, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="PROJETO INTEGRADOR 4",
    page_icon="💀",
    layout="wide"
)

# ==========================================
# INJEÇÃO DA IMAGEM DE FUNDO (OFFLINE / BASE64)
# ==========================================
def aplicar_fundo_logo(caminho_imagem="logo.png"):
    """Carrega o logo local, converte para Base64 e aplica como marca d'água no fundo."""
    if Path(caminho_imagem).exists():
        with open(caminho_imagem, "rb") as f:
            dados_base64 = base64.b64encode(f.read()).decode()
        
        css = f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(rgba(255, 255, 255, 0.90), rgba(255, 255, 255, 0.90)), 
                        url("data:image/png;base64,{dados_base64}");
            background-size: 550px;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{
            background-color: rgba(0,0,0,0);
        }}
        [data-testid="stSidebar"] {{
            background-color: rgba(245, 250, 252, 0.95);
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

aplicar_fundo_logo("logo.png")

# ==========================================
# GERENCIAMENTO DE SESSÃO & USUÁRIOS (ADMINS)
# ==========================================
if "usuarios" not in st.session_state:
    st.session_state.usuarios = {
        "admin": {"senha": "123456", "nome": "Administrador Principal"}
    }

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if "usuario_ativo" not in st.session_state:
    st.session_state.usuario_ativo = None

def tela_login():
    col1, col2, col3 = st.columns([1, 1.8, 1])
    with col2:
        st.markdown("## 🎻 Studio Duo - Acesso")
        tab_entrar, tab_cadastrar_admin = st.tabs(["🔐 Entrar", "➕ Cadastrar Novo Administrador"])

        with tab_entrar:
            with st.form("form_login"):
                usuario = st.text_input("Usuário", placeholder="Ex: admin").strip().lower()
                senha = st.text_input("Senha", type="password", placeholder="Digite sua senha")
                botao_entrar = st.form_submit_button("Acessar Sistema", use_container_width=True)

                if botao_entrar:
                    if usuario in st.session_state.usuarios and st.session_state.usuarios[usuario]["senha"] == senha:
                        st.session_state.autenticado = True
                        st.session_state.usuario_ativo = usuario
                        st.session_state.nome_ativo = st.session_state.usuarios[usuario]["nome"]
                        st.success("Login realizado com sucesso!")
                        st.rerun()
                    else:
                        st.error("Usuário ou senha incorretos.")

        with tab_cadastrar_admin:
            with st.form("form_novo_admin_login"):
                nome_novo = st.text_input("Nome Completo *").strip()
                usuario_novo = st.text_input("Nome de Usuário (Login) *").strip().lower()
                senha_nova = st.text_input("Senha *", type="password")
                senha_confirma = st.text_input("Confirmar Senha *", type="password")
                botao_cadastrar = st.form_submit_button("Criar Conta de Administrador", use_container_width=True)

                if botao_cadastrar:
                    if not nome_novo or not usuario_novo or not senha_nova:
                        st.error("Preencha todos os campos obrigatórios.")
                    elif usuario_novo in st.session_state.usuarios:
                        st.error("Este nome de usuário já está cadastrado.")
                    elif senha_nova != senha_confirma:
                        st.error("As senhas digitadas não coincidem.")
                    elif len(senha_nova) < 4:
                        st.error("A senha deve conter no mínimo 4 caracteres.")
                    else:
                        st.session_state.usuarios[usuario_novo] = {
                            "senha": senha_nova,
                            "nome": nome_novo
                        }
                        st.success(f"Administrador '{usuario_novo}' cadastrado com sucesso! Acesse na aba 'Entrar'.")

if not st.session_state.autenticado:
    tela_login()
    st.stop()

# ==========================================
# BARRA LATERAL (ADMINISTRAÇÃO & LOGOUT)
# ==========================================
with st.sidebar:
    st.markdown("### 👤 Administrador Logado")
    st.write(f"**{st.session_state.get('nome_ativo', 'Administrador')}** (`@{st.session_state.usuario_ativo}`)")
    
    if st.button("🚪 Sair (Logout)", use_container_width=True):
        st.session_state.autenticado = False
        st.session_state.usuario_ativo = None
        st.rerun()

    st.divider()
    with st.expander("👥 Cadastrar Novo Administrador"):
        with st.form("form_novo_admin_sidebar", clear_on_submit=True):
            nome_sb = st.text_input("Nome Completo *").strip()
            user_sb = st.text_input("Usuário (Login) *").strip().lower()
            senha_sb = st.text_input("Senha *", type="password")
            senha_sb_conf = st.text_input("Confirmar Senha *", type="password")
            btn_sb_admin = st.form_submit_button("Salvar Novo Admin", use_container_width=True)

            if btn_sb_admin:
                if not nome_sb or not user_sb or not senha_sb:
                    st.error("Preencha todos os campos.")
                elif user_sb in st.session_state.usuarios:
                    st.error("Usuário já existente.")
                elif senha_sb != senha_sb_conf:
                    st.error("Senhas incompatíveis.")
                else:
                    st.session_state.usuarios[user_sb] = {
                        "senha": senha_sb,
                        "nome": nome_sb
                    }
                    st.success(f"Administrador @{user_sb} registrado!")

    with st.expander("📋 Administradores Cadastrados"):
        for user_key, info in st.session_state.usuarios.items():
            st.markdown(f"- **{info['nome']}** (`@{user_key}`)")

# ==========================================
# INICIALIZAÇÃO DE DADOS OPERACIONAIS
# ==========================================
hoje = date.today()

if "alunos" not in st.session_state:
    st.session_state.alunos = [
        {
            "id": 1,
            "nome": "Mariana Souza",
            "endereco": "Rua das Flores, 120 - Centro",
            "instrumento": "Violino",
            "faixa_etaria": "Adulto (18-59)",
            "valor_mensalidade": 350.00,
            "status_pagamento": "Pago",
            "aulas_totais": 16,
            "aulas_feitas": 12,
            "historico_aulas": [
                {"id": i + 1, "data": str(hoje - timedelta(days=(12 - i) * 7))}
                for i in range(12)
            ],
            "tempo_matricula": 8,
            "frequencia": 75.0,
            "horas_estudo": 4.5,
            "evolucao": 8.0
        },
        {
            "id": 2,
            "nome": "Lucas Silveira",
            "endereco": "Av. Brasil, 450 - Jd. Paulista",
            "instrumento": "Piano",
            "faixa_etaria": "Jovem (13-17)",
            "valor_mensalidade": 400.00,
            "status_pagamento": "Inadimplente",
            "aulas_totais": 16,
            "aulas_feitas": 4,
            "historico_aulas": [
                {"id": i + 1, "data": str(hoje - timedelta(days=(4 - i) * 7))}
                for i in range(4)
            ],
            "tempo_matricula": 3,
            "frequencia": 25.0,
            "horas_estudo": 1.0,
            "evolucao": 4.0
        },
        {
            "id": 3,
            "nome": "Beatriz Lima",
            "endereco": "Rua do Sol, 88 - Vila Nova",
            "instrumento": "Piano",
            "faixa_etaria": "Infantil (6-12)",
            "valor_mensalidade": 380.00,
            "status_pagamento": "Pago",
            "aulas_totais": 20,
            "aulas_feitas": 18,
            "historico_aulas": [
                {"id": i + 1, "data": str(hoje - timedelta(days=(18 - i) * 7))}
                for i in range(18)
            ],
            "tempo_matricula": 14,
            "frequencia": 90.0,
            "horas_estudo": 6.0,
            "evolucao": 9.2
        },
        {
            "id": 4,
            "nome": "Carlos Mendes",
            "endereco": "Rua XV de Novembro, 1020",
            "instrumento": "Violino",
            "faixa_etaria": "Sênior (60+)",
            "valor_mensalidade": 350.00,
            "status_pagamento": "Pendente",
            "aulas_totais": 12,
            "aulas_feitas": 5,
            "historico_aulas": [
                {"id": i + 1, "data": str(hoje - timedelta(days=(5 - i) * 7))}
                for i in range(5)
            ],
            "tempo_matricula": 2,
            "frequencia": 41.7,
            "horas_estudo": 2.0,
            "evolucao": 5.5
        }
    ]

if "agenda" not in st.session_state:
    st.session_state.agenda = [
        {"id": 1, "aluno": "Mariana Souza", "instrumento": "Violino", "data": str(hoje), "horario": "14:00", "status": "Agendada"},
        {"id": 2, "aluno": "Lucas Silveira", "instrumento": "Piano", "data": str(hoje), "horario": "15:30", "status": "Agendada"},
        {"id": 3, "aluno": "Beatriz Lima", "instrumento": "Piano", "data": str(hoje), "horario": "17:00", "status": "Realizada"}
    ]

# ==========================================
# MOTOR DE DADOS & MACHINE LEARNING (BACKEND)
# ==========================================
@st.cache_resource
def inicializar_modelo_ia():
    np.random.seed(42)
    n_samples = 1200

    instrumentos = np.random.choice(["Violino", "Piano"], size=n_samples, p=[0.48, 0.52])
    faixas_etarias = np.random.choice(["Infantil (6-12)", "Jovem (13-17)", "Adulto (18-59)", "Sênior (60+)"], size=n_samples)
    tempo_matricula = np.random.randint(1, 48, size=n_samples)
    taxa_frequencia = np.random.uniform(40, 100, size=n_samples)
    aulas_canceladas_3m = np.random.poisson(lam=1.8, size=n_samples)
    atrasos_pagamento = np.random.poisson(lam=1.2, size=n_samples)
    evolucao_tecnica = np.random.uniform(1.0, 10.0, size=n_samples)
    horas_estudo_casa = np.random.uniform(0.5, 12.0, size=n_samples)

    score_risco = (
        (100 - taxa_frequencia) * 0.35 +
        (aulas_canceladas_3m * 6.5) +
        (atrasos_pagamento * 8.0) +
        (10 - evolucao_tecnica) * 3.0 -
        (horas_estudo_casa * 2.5) -
        (tempo_matricula * 0.4)
    )
    probabilidade_evasao = 1 / (1 + np.exp(-(score_risco - 25) / 10))
    evasao = (np.random.rand(n_samples) < probabilidade_evasao).astype(int)

    df_treino = pd.DataFrame({
        "Tempo_Matricula_Meses": tempo_matricula,
        "Taxa_Frequencia_Pct": np.round(taxa_frequencia, 1),
        "Aulas_Canceladas_3M": aulas_canceladas_3m,
        "Atrasos_Pagamento": atrasos_pagamento,
        "Evolucao_Tecnica": np.round(evolucao_tecnica, 1),
        "Horas_Estudo_Semana": np.round(horas_estudo_casa, 1),
        "Instrumento": instrumentos,
        "Faixa_Etaria": faixas_etarias,
        "Evasao": evasao
    })

    df_encoded = pd.get_dummies(df_treino.drop(columns=["Evasao"]), columns=["Instrumento", "Faixa_Etaria"], dtype=int)
    X = df_encoded
    y = df_treino["Evasao"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    modelo = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    modelo.fit(X_train, y_train)
    acuracia = accuracy_score(y_test, modelo.predict(X_test))

    return modelo, list(X.columns), acuracia, df_treino

modelo_ia, colunas_modelo, acuracia_ia, df_historico = inicializar_modelo_ia()

def predizer_risco_aluno(dados_aluno):
    df_single = pd.DataFrame([dados_aluno])
    df_enc = pd.get_dummies(df_single, columns=["Instrumento", "Faixa_Etaria"], dtype=int)
    df_final = df_enc.reindex(columns=colunas_modelo, fill_value=0)
    prob = modelo_ia.predict_proba(df_final)[0][1] * 100
    return float(prob)

# ==========================================
# PAINEL PRINCIPAL
# ==========================================
st.title("💀 PROJETO INTEGRADOR 4 💀")

tab_financas, tab_alunos, tab_agenda, tab_predicao = st.tabs([
    "💰 Financeiro",
    "👥 Alunos & Presença",
    "📅 Agendamentos",
    "🧠 Diagnóstico IA"
])

# ----------------------------------------------------
# ABA FINANCEIRO
# ----------------------------------------------------
with tab_financas:
    st.subheader("Controle de Caixa e Inadimplência")
    df_cadastrados = pd.DataFrame(st.session_state.alunos)

    if not df_cadastrados.empty:
        total_previsto = df_cadastrados["valor_mensalidade"].sum()
        total_pago = df_cadastrados[df_cadastrados["status_pagamento"] == "Pago"]["valor_mensalidade"].sum()
        total_pendente = df_cadastrados[df_cadastrados["status_pagamento"] == "Pendente"]["valor_mensalidade"].sum()
        total_inadimplente = df_cadastrados[df_cadastrados["status_pagamento"] == "Inadimplente"]["valor_mensalidade"].sum()
        total_falta = total_pendente + total_inadimplente

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Faturamento Previsto", f"R$ {total_previsto:,.2f}")
        c2.metric("Valor Entrado (Pago)", f"R$ {total_pago:,.2f}", delta=f"{(total_pago / total_previsto) * 100:.1f}% recebido" if total_previsto > 0 else "0%")
        c3.metric("Valor em Aberto (Falta)", f"R$ {total_falta:,.2f}", delta=f"-{(total_falta / total_previsto) * 100:.1f}%" if total_previsto > 0 else "0%", delta_color="inverse")
        c4.metric("Inadimplência Crítica", f"R$ {total_inadimplente:,.2f}", delta_color="inverse")

        st.divider()
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_status = px.pie(
                df_cadastrados,
                names="status_pagamento",
                values="valor_mensalidade",
                title="Composição dos Recebimentos",
                color="status_pagamento",
                color_discrete_map={"Pago": "#2ecc71", "Pendente": "#f39c12", "Inadimplente": "#e74c3c"}
            )
            st.plotly_chart(fig_status, use_container_width=True)

        with col_g2:
            fig_inst = px.bar(
                df_cadastrados,
                x="instrumento",
                y="valor_mensalidade",
                color="status_pagamento",
                title="Mensalidades por Instrumento",
                barmode="group",
                color_discrete_map={"Pago": "#2ecc71", "Pendente": "#f39c12", "Inadimplente": "#e74c3c"}
            )
            st.plotly_chart(fig_inst, use_container_width=True)

# ----------------------------------------------------
# ABA ALUNOS & PRESENÇA COM HISTÓRICO COMPLETO
# ----------------------------------------------------
with tab_alunos:
    st.subheader("Cadastro e Registro de Presença")

    with st.expander("➕ Cadastrar Novo Aluno", expanded=False):
        with st.form("form_cadastro_aluno", clear_on_submit=True):
            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                nome = st.text_input("Nome do Aluno *")
                endereco = st.text_input("Endereço *")
                instrumento = st.selectbox("Instrumento", ["Violino", "Piano"])
                faixa_etaria = st.selectbox("Faixa Etária", ["Infantil (6-12)", "Jovem (13-17)", "Adulto (18-59)", "Sênior (60+)"])

            with col_a2:
                mensalidade = st.number_input("Valor da Mensalidade (R$)", min_value=100.0, max_value=2000.0, value=350.0, step=50.0)
                status_pag = st.selectbox("Status do Pagamento", ["Pago", "Pendente", "Inadimplente"])
                aulas_totais = st.number_input("Total de Aulas do Pacote", min_value=1, max_value=100, value=16)

            with col_a3:
                aulas_feitas = st.number_input("Aulas Concluídas", min_value=0, max_value=100, value=0)
                tempo_mat = st.number_input("Tempo de Matrícula (Meses)", min_value=1, max_value=60, value=1)
                horas_estudo = st.number_input("Estudo Semanal em Casa (Horas)", min_value=0.0, max_value=20.0, value=2.5, step=0.5)

            cadastrar = st.form_submit_button("Salvar Registro")

            if cadastrar:
                if nome.strip() and endereco.strip():
                    novo_id = max([a["id"] for a in st.session_state.alunos], default=0) + 1
                    freq_calc = 100.0 if aulas_totais == 0 else round(min(100.0, (int(aulas_feitas) / int(aulas_totais)) * 100), 1)
                    hist_inicial = [{"id": i + 1, "data": str(hoje)} for i in range(int(aulas_feitas))]

                    st.session_state.alunos.append({
                        "id": novo_id,
                        "nome": nome.strip(),
                        "endereco": endereco.strip(),
                        "instrumento": instrumento,
                        "faixa_etaria": faixa_etaria,
                        "valor_mensalidade": float(mensalidade),
                        "status_pagamento": status_pag,
                        "aulas_totais": int(aulas_totais),
                        "aulas_feitas": int(aulas_feitas),
                        "historico_aulas": hist_inicial,
                        "tempo_matricula": int(tempo_mat),
                        "frequencia": float(freq_calc),
                        "horas_estudo": float(horas_estudo),
                        "evolucao": 7.0
                    })
                    st.success(f"Aluno {nome} adicionado com sucesso!")
                    st.rerun()
                else:
                    st.error("Preencha ao menos o Nome e o Endereço.")

    st.markdown("### 📋 Ficha de Alunos e Progresso das Aulas")
    for aluno in st.session_state.alunos:
        if "historico_aulas" not in aluno:
            aluno["historico_aulas"] = [{"id": i + 1, "data": str(hoje)} for i in range(aluno.get("aulas_feitas", 0))]

        aluno["aulas_feitas"] = len(aluno["historico_aulas"])
        aulas_a_fazer = max(0, aluno["aulas_totais"] - aluno["aulas_feitas"])
        pct_concluido = min(1.0, aluno["aulas_feitas"] / aluno["aulas_totais"]) if aluno["aulas_totais"] > 0 else 0.0
        aluno["frequencia"] = round(min(100.0, (aluno["aulas_feitas"] / aluno["aulas_totais"]) * 100), 1) if aluno["aulas_totais"] > 0 else 100.0

        atraso_num = 3 if aluno["status_pagamento"] == "Inadimplente" else (1 if aluno["status_pagamento"] == "Pendente" else 0)
        dados_inferencia = {
            "Tempo_Matricula_Meses": aluno["tempo_matricula"],
            "Taxa_Frequencia_Pct": aluno["frequencia"],
            "Aulas_Canceladas_3M": max(0, aulas_a_fazer - 2),
            "Atrasos_Pagamento": atraso_num,
            "Evolucao_Tecnica": aluno.get("evolucao", 7.0),
            "Horas_Estudo_Semana": aluno["horas_estudo"],
            "Instrumento": aluno["instrumento"],
            "Faixa_Etaria": aluno["faixa_etaria"]
        }
        risco_aluno = predizer_risco_aluno(dados_inferencia)

        with st.container():
            c_info1, c_info2, c_info3, c_info4 = st.columns([3, 2, 2.5, 2])
            with c_info1:
                st.markdown(f"**{aluno['nome']}** (`{aluno['instrumento']}`)")
                st.caption(f"📍 {aluno['endereco']} | {aluno['faixa_etaria']}")
                if risco_aluno < 35:
                    st.caption(f"🟢 Risco de Evasão: **{risco_aluno:.1f}% (Baixo)**")
                elif risco_aluno < 65:
                    st.caption(f"🟡 Risco de Evasão: **{risco_aluno:.1f}% (Moderado)**")
                else:
                    st.caption(f"🔴 Risco de Evasão: **{risco_aluno:.1f}% (Crítico)**")

            with c_info2:
                st.write(f"**Mensalidade:** R$ {aluno['valor_mensalidade']:.2f}")
                status_atual = aluno["status_pagamento"]
                novo_st = st.selectbox(
                    "Status Financeiro",
                    ["Pago", "Pendente", "Inadimplente"],
                    index=["Pago", "Pendente", "Inadimplente"].index(status_atual),
                    key=f"status_sel_{aluno['id']}"
                )
                if novo_st != status_atual:
                    aluno["status_pagamento"] = novo_st
                    st.rerun()

            with c_info3:
                st.write(f"**Aulas:** {aluno['aulas_feitas']} feitas / {aluno['aulas_totais']} total | **{aulas_a_fazer} a fazer**")
                st.progress(pct_concluido)

            with c_info4:
                if st.button("➕ Presença Hoje", key=f"pres_hoje_{aluno['id']}"):
                    if aluno["aulas_feitas"] < aluno["aulas_totais"]:
                        novo_id_h = max([h["id"] for h in aluno["historico_aulas"]], default=0) + 1
                        aluno["historico_aulas"].append({"id": novo_id_h, "data": str(hoje)})
                        st.success("Presença de hoje confirmada!")
                        st.rerun()
                    else:
                        st.info("Pacote de aulas concluído.")

            with st.expander(f"📅 Ver e Gerenciar Histórico de Aulas ({len(aluno['historico_aulas'])} registradas)"):
                st.markdown("##### 📌 Marcar Aula em Outro Dia")
                col_n1, col_n2 = st.columns([3, 1])
                with col_n1:
                    data_marcar = st.date_input(
                        "Selecione a data da aula realizada",
                        value=hoje,
                        key=f"data_marcar_{aluno['id']}"
                    )
                with col_n2:
                    st.write("")
                    st.write("")
                    if st.button("➕ Registrar Aula na Data", key=f"btn_marcar_data_{aluno['id']}"):
                        if aluno["aulas_feitas"] < aluno["aulas_totais"]:
                            novo_id_h = max([h["id"] for h in aluno["historico_aulas"]], default=0) + 1
                            aluno["historico_aulas"].append({"id": novo_id_h, "data": str(data_marcar)})
                            st.success(f"Aula em {data_marcar.strftime('%d/%m/%Y')} registrada!")
                            st.rerun()
                        else:
                            st.warning("Limite total de aulas já atingido.")

                st.divider()
                st.markdown("##### 📋 Dias Realizados (Editar ou Excluir)")

                if aluno["historico_aulas"]:
                    for idx, aula_reg in enumerate(aluno["historico_aulas"]):
                        col_h1, col_h2, col_h3, col_h4 = st.columns([1, 2, 1.2, 1])
                        with col_h1:
                            st.write(f"**Aula #{idx+1}**")
                        with col_h2:
                            try:
                                dt_atual = date.fromisoformat(aula_reg["data"])
                            except Exception:
                                dt_atual = hoje
                            nova_data_edit = st.date_input(
                                f"Data Aula {aula_reg['id']}",
                                value=dt_atual,
                                key=f"dt_edit_{aluno['id']}_{aula_reg['id']}",
                                label_visibility="collapsed"
                            )
                        with col_h3:
                            if st.button("💾 Alterar", key=f"btn_alt_{aluno['id']}_{aula_reg['id']}"):
                                aula_reg["data"] = str(nova_data_edit)
                                st.success("Data alterada com sucesso!")
                                st.rerun()
                        with col_h4:
                            if st.button("🗑️ Excluir", key=f"btn_del_{aluno['id']}_{aula_reg['id']}"):
                                aluno["historico_aulas"].remove(aula_reg)
                                st.warning("Aula removida.")
                                st.rerun()
                else:
                    st.info("Nenhuma aula realizada até o momento.")

            st.divider()

# ----------------------------------------------------
# ABA AGENDAMENTOS
# ----------------------------------------------------
with tab_agenda:
    st.subheader("Agendamento de Aulas")
    col_ag1, col_ag2 = st.columns([1, 2])

    with col_ag1:
        st.markdown("#### Marcar Horário")
        lista_nomes = [a["nome"] for a in st.session_state.alunos]
        if lista_nomes:
            aluno_agenda = st.selectbox("Aluno", lista_nomes, key="ag_aluno")
            data_sel = st.date_input("Data da Aula", min_value=hoje, key="ag_data")
            hora_sel = st.time_input("Horário", value=time(14, 0), key="ag_hora")

            if st.button("Confirmar Agendamento", key="btn_confirmar_ag"):
                aluno_obj = next(a for a in st.session_state.alunos if a["nome"] == aluno_agenda)
                st.session_state.agenda.append({
                    "id": len(st.session_state.agenda) + 1,
                    "aluno": aluno_agenda,
                    "instrumento": aluno_obj["instrumento"],
                    "data": str(data_sel),
                    "horario": hora_sel.strftime("%H:%M"),
                    "status": "Agendada"
                })
                st.success("Aula agendada!")
                st.rerun()
        else:
            st.warning("Cadastre alunos primeiro.")

    with col_ag2:
        st.markdown("#### Grade de Aulas")
        for ag in st.session_state.agenda:
            ca1, ca2, ca3 = st.columns([3, 2, 2])
            ca1.write(f"🎵 **{ag['aluno']}** ({ag['instrumento']})")
            ca2.write(f"🗓️ {ag['data']} às {ag['horario']}")
            if ag["status"] == "Agendada":
                if ca3.button("Marcar Realizada", key=f"btn_done_{ag['id']}"):
                    ag["status"] = "Realizada"
                    for al in st.session_state.alunos:
                        if al["nome"] == ag["aluno"] and al["aulas_feitas"] < al["aulas_totais"]:
                            if "historico_aulas" not in al:
                                al["historico_aulas"] = []
                            novo_id_h = max([h["id"] for h in al["historico_aulas"]], default=0) + 1
                            al["historico_aulas"].append({"id": novo_id_h, "data": str(ag["data"])})
                    st.rerun()
            else:
                ca3.success(" Concluída")
            st.divider()

# ----------------------------------------------------
# ABA PREDIÇÃO IA
# ----------------------------------------------------
with tab_predicao:
    st.subheader("Análise Preditiva de Risco de Evasão")
    origem = st.radio("Selecione o modo de análise:", ["Carregar Aluno Cadastrado", "Simulação Manual"], horizontal=True)

    if origem == "Carregar Aluno Cadastrado" and st.session_state.alunos:
        aluno_nome = st.selectbox("Escolha o Aluno Cadastrado", [a["nome"] for a in st.session_state.alunos], key="sel_aluno_pred")
        aluno_selecionado = next(a for a in st.session_state.alunos if a["nome"] == aluno_nome)

        val_inst = aluno_selecionado["instrumento"]
        val_faixa = aluno_selecionado["faixa_etaria"]
        val_tempo = aluno_selecionado["tempo_matricula"]
        val_freq = aluno_selecionado["frequencia"]
        val_atrasos = 3 if aluno_selecionado["status_pagamento"] == "Inadimplente" else (1 if aluno_selecionado["status_pagamento"] == "Pendente" else 0)
        val_canceladas = max(0, aluno_selecionado["aulas_totais"] - aluno_selecionado["aulas_feitas"] - 2)
        val_evolucao = aluno_selecionado.get("evolucao", 7.0)
        val_estudo = aluno_selecionado["horas_estudo"]

        st.info(f"Parâmetros de **{aluno_nome}** carregados: Pagamento `{aluno_selecionado['status_pagamento']}` | Frequência `{val_freq}%` | Estudo `{val_estudo}h/sem`")
    else:
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            val_inst = st.selectbox("Instrumento", ["Violino", "Piano"], key="man_inst")
            val_faixa = st.selectbox("Faixa Etária", ["Infantil (6-12)", "Jovem (13-17)", "Adulto (18-59)", "Sênior (60+)"], key="man_faixa")
            val_tempo = st.slider("Tempo de Matrícula (meses)", 1, 48, 6, key="man_tempo")
        with col_s2:
            val_freq = st.slider("Taxa de Frequência (%)", 0.0, 100.0, 75.0, key="man_freq")
            val_canceladas = st.number_input("Aulas Canceladas (últimos 3 meses)", min_value=0, max_value=12, value=1, key="man_canc")
            val_atrasos = st.number_input("Histórico de Atrasos / Falta de Pagamento", min_value=0, max_value=10, value=0, key="man_atra")
        with col_s3:
            val_evolucao = st.slider("Nota de Evolução Técnica (1 a 10)", 1.0, 10.0, 6.5, step=0.5, key="man_evol")
            val_estudo = st.slider("Horas de Estudo em Casa / Semana", 0.0, 15.0, 3.0, step=0.5, key="man_est")

    if st.button("🔍 Executar Diagnóstico de Risco com IA", key="btn_rodar_ia"):
        input_dict = {
            "Tempo_Matricula_Meses": val_tempo,
            "Taxa_Frequencia_Pct": val_freq,
            "Aulas_Canceladas_3M": val_canceladas,
            "Atrasos_Pagamento": val_atrasos,
            "Evolucao_Tecnica": val_evolucao,
            "Horas_Estudo_Semana": val_estudo,
            "Instrumento": val_inst,
            "Faixa_Etaria": val_faixa
        }

        probabilidade = predizer_risco_aluno(input_dict)

        st.divider()
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            if probabilidade < 35:
                st.success(f"🟢 **Risco Baixo de Evasão:** {probabilidade:.1f}%")
            elif probabilidade < 65:
                st.warning(f"🟡 **Risco Moderado de Evasão:** {probabilidade:.1f}%")
            else:
                st.error(f"🔴 **Risco Crítico de Evasão:** {probabilidade:.1f}%")

        with col_res2:
            if probabilidade >= 50:
                st.warning("⚠️ **Recomendação:** Contato ativo pedagógico, revisão da dificuldade do repertório e renegociação preventiva de eventuais parcelas.")
            else:
                st.success("✅ **Recomendação:** Aluno com boa retenção. Manter o cronograma pedagógico atual.")
