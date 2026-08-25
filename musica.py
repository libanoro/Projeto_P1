import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from datetime import date, time, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Gestão Musical - Metro UI",
    page_icon="🎵",
    layout="wide"
)

# ==========================================
# DESIGN SYSTEM: METRO UI (CSS CUSTOMIZADO)
# ==========================================
metro_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Segoe+UI:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif !important;
    }

    .stApp {
        background-color: #f3f3f3;
        color: #1a1a1a;
    }

    /* Títulos Estilo Metro / Modern Windows */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Segoe UI Light', 'Segoe UI', sans-serif !important;
        font-weight: 300 !important;
        letter-spacing: -0.5px;
        color: #111111;
        text-transform: none;
    }

    /* Botões Flat Metro */
    .stButton > button {
        border-radius: 0px !important;
        border: 2px solid transparent !important;
        background-color: #0078d7 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        font-size: 12px !important;
        letter-spacing: 0.5px !important;
        padding: 8px 16px !important;
        transition: background-color 0.2s ease, border-color 0.2s ease !important;
    }

    .stButton > button:hover {
        background-color: #005a9e !important;
        color: #ffffff !important;
        border-color: #004578 !important;
    }

    .stButton > button:focus {
        box-shadow: none !important;
    }

    /* Campos de Entrada Sharp (Sem arredondamento) */
    .stTextInput input, .stSelectbox [data-baseweb="select"], .stNumberInput input, .stDateInput input, .stTimeInput input {
        border-radius: 0px !important;
        border: 1px solid #767676 !important;
        background-color: #ffffff !important;
        color: #000000 !important;
        font-family: 'Segoe UI', sans-serif !important;
    }

    .stTextInput input:focus, .stNumberInput input:focus, .stDateInput input:focus, .stTimeInput input:focus {
        border-color: #0078d7 !important;
        box-shadow: 0 0 0 1px #0078d7 !important;
    }

    /* Tabs Flat com Indicador Accent */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #ffffff;
        border-bottom: 2px solid #e0e0e0;
        padding-left: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 0px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        color: #555555 !important;
        border: none !important;
        background-color: transparent !important;
    }

    .stTabs [aria-selected="true"] {
        color: #0078d7 !important;
        border-bottom: 3px solid #0078d7 !important;
        background-color: #f9f9f9 !important;
    }

    /* Tiles / Cards Flat Metro */
    .metro-tile {
        padding: 16px 20px;
        margin-bottom: 15px;
        color: #ffffff;
        position: relative;
        overflow: hidden;
    }

    .metro-tile-blue { background-color: #0078d7; }
    .metro-tile-green { background-color: #107c41; }
    .metro-tile-orange { background-color: #d83b01; }
    .metro-tile-red { background-color: #e81123; }
    .metro-tile-purple { background-color: #5c2d91; }
    .metro-tile-teal { background-color: #008272; }

    .metro-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-left: 5px solid #0078d7;
        padding: 16px;
        margin-bottom: 16px;
    }

    /* Expanders Sharp */
    .streamlit-expanderHeader {
        border-radius: 0px !important;
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        font-weight: 600 !important;
    }

    .streamlit-expanderContent {
        border-radius: 0px !important;
        border: 1px solid #e0e0e0 !important;
        border-top: none !important;
        background-color: #ffffff !important;
    }

    /* Alertas e Badges Flat */
    .stAlert {
        border-radius: 0px !important;
        border: none !important;
        border-left: 4px solid !important;
    }
</style>
"""
st.markdown(metro_css, unsafe_allow_html=True)

pio.templates.default = "plotly_white"

# ==========================================
# GERENCIAMENTO DE SESSÃO & AUTENTICAÇÃO
# ==========================================
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

def tela_login():
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.6, 1])
    with c2:
        st.markdown("""
            <div style="background-color: #0078d7; color: white; padding: 25px 30px;">
                <h1 style="color: white !important; margin: 0; font-size: 28px;">INICIAR SESSÃO</h1>
                <p style="margin: 5px 0 0 0; opacity: 0.85; font-size: 14px;">Painel de Controle Metro UI</p>
            </div>
        """, unsafe_allow_html=True)

        with st.container():
            st.markdown('<div style="background-color: #ffffff; padding: 25px; border: 1px solid #e0e0e0; border-top: none;">', unsafe_allow_html=True)
            with st.form("form_login"):
                usuario = st.text_input("NOME DE USUÁRIO", placeholder="admin")
                senha = st.text_input("PALAVRA-PASSE", type="password", placeholder="123456")
                botao_entrar = st.form_submit_button("ENTRAR NO SISTEMA", use_container_width=True)

                if botao_entrar:
                    if usuario == "admin" and senha == "123456":
                        st.session_state.autenticado = True
                        st.success("Autenticação autorizada.")
                        st.rerun()
                    else:
                        st.error("Credenciais inválidas. Tente novamente.")
            st.markdown('</div>', unsafe_allow_html=True)

if not st.session_state.autenticado:
    tela_login()
    st.stop()

# ==========================================
# BARRA LATERAL METRO
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style="background-color: #2b2b2b; color: #ffffff; padding: 15px; margin: -15px -15px 15px -15px;">
            <div style="font-size: 11px; text-transform: uppercase; color: #00a4ef; font-weight: 700;">CONTA ATIVA</div>
            <div style="font-size: 18px; font-weight: 300;">Administrador</div>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 ENCERRAR SESSÃO", use_container_width=True):
        st.session_state.autenticado = False
        st.rerun()

# ==========================================
# INICIALIZAÇÃO DE DADOS
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
            "historico_aulas": [{"id": i + 1, "data": str(hoje - timedelta(days=(12 - i) * 7))} for i in range(12)],
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
            "historico_aulas": [{"id": i + 1, "data": str(hoje - timedelta(days=(4 - i) * 7))} for i in range(4)],
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
            "historico_aulas": [{"id": i + 1, "data": str(hoje - timedelta(days=(18 - i) * 7))} for i in range(18)],
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
            "historico_aulas": [{"id": i + 1, "data": str(hoje - timedelta(days=(5 - i) * 7))} for i in range(5)],
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
# MOTOR IA
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
# CABEÇALHO DO SISTEMA
# ==========================================
st.markdown("""
    <div style="background-color: #0078d7; color: white; padding: 20px 30px; margin-bottom: 25px;">
        <h1 style="color: white !important; margin: 0; font-size: 32px;">SISTEMA DE GESTÃO MUSICAL</h1>
        <div style="font-size: 13px; letter-spacing: 1px; text-transform: uppercase; opacity: 0.9;">Painel Integrado & Diagnóstico Preditivo</div>
    </div>
""", unsafe_allow_html=True)

tab_financas, tab_alunos, tab_agenda, tab_predicao, tab_bi = st.tabs([
    "FINANCEIRO",
    "ALUNOS & PRESENÇA",
    "AGENDAMENTOS",
    "DIAGNÓSTICO IA",
    "BI & DADOS HISTÓRICOS"
])

# ----------------------------------------------------
# TAB 1: FINANCEIRO (METRO LIVE TILES)
# ----------------------------------------------------
with tab_financas:
    st.markdown("### Visão Geral de Caixa")
    df_cadastrados = pd.DataFrame(st.session_state.alunos)

    if not df_cadastrados.empty:
        total_previsto = df_cadastrados["valor_mensalidade"].sum()
        total_pago = df_cadastrados[df_cadastrados["status_pagamento"] == "Pago"]["valor_mensalidade"].sum()
        total_pendente = df_cadastrados[df_cadastrados["status_pagamento"] == "Pendente"]["valor_mensalidade"].sum()
        total_inadimplente = df_cadastrados[df_cadastrados["status_pagamento"] == "Inadimplente"]["valor_mensalidade"].sum()
        total_falta = total_pendente + total_inadimplente

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""
            <div class="metro-tile metro-tile-blue">
                <div style="font-size: 11px; text-transform: uppercase; font-weight: 700;">FATURAMENTO PREVISTO</div>
                <div style="font-size: 26px; font-weight: 300; margin: 8px 0;">R$ {total_previsto:,.2f}</div>
                <div style="font-size: 12px; opacity: 0.85;">Total da base</div>
            </div>
        """, unsafe_allow_html=True)

        c2.markdown(f"""
            <div class="metro-tile metro-tile-green">
                <div style="font-size: 11px; text-transform: uppercase; font-weight: 700;">VALOR RECEBIDO</div>
                <div style="font-size: 26px; font-weight: 300; margin: 8px 0;">R$ {total_pago:,.2f}</div>
                <div style="font-size: 12px; opacity: 0.85;">{(total_pago / total_previsto) * 100:.1f}% liquidado</div>
            </div>
        """, unsafe_allow_html=True)

        c3.markdown(f"""
            <div class="metro-tile metro-tile-orange">
                <div style="font-size: 11px; text-transform: uppercase; font-weight: 700;">VALOR PENDENTE</div>
                <div style="font-size: 26px; font-weight: 300; margin: 8px 0;">R$ {total_falta:,.2f}</div>
                <div style="font-size: 12px; opacity: 0.85;">{(total_falta / total_previsto) * 100:.1f}% em aberto</div>
            </div>
        """, unsafe_allow_html=True)

        c4.markdown(f"""
            <div class="metro-tile metro-tile-red">
                <div style="font-size: 11px; text-transform: uppercase; font-weight: 700;">INADIMPLÊNCIA CRÍTICA</div>
                <div style="font-size: 26px; font-weight: 300; margin: 8px 0;">R$ {total_inadimplente:,.2f}</div>
                <div style="font-size: 12px; opacity: 0.85;">Atrasos confirmados</div>
            </div>
        """, unsafe_allow_html=True)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            fig_status = px.pie(
                df_cadastrados,
                names="status_pagamento",
                values="valor_mensalidade",
                title="Composição dos Recebimentos",
                color="status_pagamento",
                color_discrete_map={"Pago": "#107c41", "Pendente": "#d83b01", "Inadimplente": "#e81123"}
            )
            fig_status.update_layout(font_family="Segoe UI")
            st.plotly_chart(fig_status, use_container_width=True)

        with col_g2:
            fig_inst = px.bar(
                df_cadastrados,
                x="instrumento",
                y="valor_mensalidade",
                color="status_pagamento",
                title="Receita por Categoria de Instrumento",
                barmode="group",
                color_discrete_map={"Pago": "#107c41", "Pendente": "#d83b01", "Inadimplente": "#e81123"}
            )
            fig_inst.update_layout(font_family="Segoe UI")
            st.plotly_chart(fig_inst, use_container_width=True)

# ----------------------------------------------------
# TAB 2: ALUNOS & PRESENÇA (METRO CARDS + GESTÃO)
# ----------------------------------------------------
with tab_alunos:
    st.markdown("### Cadastro e Gestão de Presença")

    with st.expander("➕ NOVO CADASTRO DE ALUNO", expanded=False):
        with st.form("form_cadastro_aluno", clear_on_submit=True):
            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                nome = st.text_input("NOME COMPLETO *")
                endereco = st.text_input("ENDEREÇO *")
                instrumento = st.selectbox("INSTRUMENTO", ["Violino", "Piano"])
                faixa_etaria = st.selectbox("FAIXA ETÁRIA", ["Infantil (6-12)", "Jovem (13-17)", "Adulto (18-59)", "Sênior (60+)"])

            with col_a2:
                mensalidade = st.number_input("VALOR DA MENSALIDADE (R$)", min_value=100.0, max_value=2000.0, value=350.0, step=50.0)
                status_pag = st.selectbox("SITUAÇÃO FINANCEIRA", ["Pago", "Pendente", "Inadimplente"])
                aulas_totais = st.number_input("PACOTE TOTAL DE AULAS", min_value=1, max_value=100, value=16)

            with col_a3:
                aulas_feitas = st.number_input("AULAS REALIZADAS INICIAIS", min_value=0, max_value=100, value=0)
                tempo_mat = st.number_input("TEMPO DE MATRÍCULA (MESES)", min_value=1, max_value=60, value=1)
                horas_estudo = st.number_input("ESTUDO SEMANAL (HORAS)", min_value=0.0, max_value=20.0, value=2.5, step=0.5)

            cadastrar = st.form_submit_button("GRAVAR REGISTRO", use_container_width=True)

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
                    st.success(f"Aluno {nome} registrado com sucesso!")
                    st.rerun()
                else:
                    st.error("Preencha os campos obrigatórios (Nome e Endereço).")

    st.markdown("### Alunos Matriculados")
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

        borda_cor = "#107c41" if risco_aluno < 35 else ("#d83b01" if risco_aluno < 65 else "#e81123")

        st.markdown(f'<div class="metro-card" style="border-left: 6px solid {borda_cor};">', unsafe_allow_html=True)
        c_info1, c_info2, c_info3, c_info4 = st.columns([3, 2, 2.5, 2])
        
        with c_info1:
            st.markdown(f"### {aluno['nome']}")
            st.markdown(f"**Instrumento:** {aluno['instrumento']} | **Grupo:** {aluno['faixa_etaria']}")
            st.caption(f"📍 {aluno['endereco']}")
            if risco_aluno < 35:
                st.markdown(f"<span style='background-color: #107c41; color: white; padding: 2px 8px; font-size: 12px; font-weight: 600;'>RISCO BAIXO: {risco_aluno:.1f}%</span>", unsafe_allow_html=True)
            elif risco_aluno < 65:
                st.markdown(f"<span style='background-color: #d83b01; color: white; padding: 2px 8px; font-size: 12px; font-weight: 600;'>RISCO MODERADO: {risco_aluno:.1f}%</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='background-color: #e81123; color: white; padding: 2px 8px; font-size: 12px; font-weight: 600;'>RISCO CRÍTICO: {risco_aluno:.1f}%</span>", unsafe_allow_html=True)

        with c_info2:
            st.write(f"**Mensalidade:** R$ {aluno['valor_mensalidade']:.2f}")
            status_atual = aluno["status_pagamento"]
            novo_st = st.selectbox(
                "SITUAÇÃO FINANCEIRA",
                ["Pago", "Pendente", "Inadimplente"],
                index=["Pago", "Pendente", "Inadimplente"].index(status_atual),
                key=f"status_sel_{aluno['id']}"
            )
            if novo_st != status_atual:
                aluno["status_pagamento"] = novo_st
                st.rerun()

        with c_info3:
            st.write(f"**Aulas Feitas:** {aluno['aulas_feitas']} / {aluno['aulas_totais']}")
            st.write(f"**Restantes:** {aulas_a_fazer} aulas")
            st.progress(pct_concluido)

        with c_info4:
            st.write("")
            if st.button("✔ PRESENÇA HOJE", key=f"pres_hoje_{aluno['id']}", use_container_width=True):
                if aluno["aulas_feitas"] < aluno["aulas_totais"]:
                    novo_id_h = max([h["id"] for h in aluno["historico_aulas"]], default=0) + 1
                    aluno["historico_aulas"].append({"id": novo_id_h, "data": str(hoje)})
                    st.success("Presença gravada com sucesso.")
                    st.rerun()
                else:
                    st.info("Pacote de aulas preenchido.")

        # HISTÓRICO EXPANSÍVEL
        with st.expander(f"HISTÓRICO E GESTÃO DE DATAS ({len(aluno['historico_aulas'])} aulas)"):
            st.markdown("##### 📌 Agendar/Marcar Data Retroativa ou Nova")
            col_n1, col_n2 = st.columns([3, 1])
            with col_n1:
                data_marcar = st.date_input(
                    "Data da Realização",
                    value=hoje,
                    key=f"data_marcar_{aluno['id']}"
                )
            with col_n2:
                st.write("")
                st.write("")
                if st.button("➕ ADICIONAR DATA", key=f"btn_marcar_data_{aluno['id']}", use_container_width=True):
                    if aluno["aulas_feitas"] < aluno["aulas_totais"]:
                        novo_id_h = max([h["id"] for h in aluno["historico_aulas"]], default=0) + 1
                        aluno["historico_aulas"].append({"id": novo_id_h, "data": str(data_marcar)})
                        st.success("Data incluída no histórico!")
                        st.rerun()
                    else:
                        st.warning("Limite do pacote atingido.")

            st.markdown("##### 📋 Histórico Registrado")
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
                        if st.button("ALTERAR", key=f"btn_alt_{aluno['id']}_{aula_reg['id']}", use_container_width=True):
                            aula_reg["data"] = str(nova_data_edit)
                            st.success("Data modificada!")
                            st.rerun()
                    with col_h4:
                        if st.button("EXCLUIR", key=f"btn_del_{aluno['id']}_{aula_reg['id']}", use_container_width=True):
                            aluno["historico_aulas"].remove(aula_reg)
                            st.rerun()
            else:
                st.info("Nenhum histórico registrado até o momento.")
                
        st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# TAB 3: AGENDAMENTOS (METRO AGENDA)
# ----------------------------------------------------
with tab_agenda:
    st.markdown("### Grade de Horários")
    col_ag1, col_ag2 = st.columns([1.2, 2])

    with col_ag1:
        st.markdown("""
            <div style="background-color: #0078d7; color: white; padding: 10px 15px; font-weight: 600; font-size: 13px;">
                MARCAR NOVO HORÁRIO
            </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="metro-card" style="border-top: none;">', unsafe_allow_html=True)
        lista_nomes = [a["nome"] for a in st.session_state.alunos]
        if lista_nomes:
            aluno_agenda = st.selectbox("ALUNO", lista_nomes, key="ag_aluno")
            data_sel = st.date_input("DATA DA SESSÃO", min_value=hoje, key="ag_data")
            hora_sel = st.time_input("HORÁRIO", value=time(14, 0), key="ag_hora")

            if st.button("CONFIRMAR AGENDAMENTO", key="btn_confirmar_ag", use_container_width=True):
                aluno_obj = next(a for a in st.session_state.alunos if a["nome"] == aluno_agenda)
                st.session_state.agenda.append({
                    "id": len(st.session_state.agenda) + 1,
                    "aluno": aluno_agenda,
                    "instrumento": aluno_obj["instrumento"],
                    "data": str(data_sel),
                    "horario": hora_sel.strftime("%H:%M"),
                    "status": "Agendada"
                })
                st.success("Aula confirmada no calendário!")
                st.rerun()
        else:
            st.warning("Nenhum aluno cadastrado.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_ag2:
        st.markdown("""
            <div style="background-color: #2b2b2b; color: white; padding: 10px 15px; font-weight: 600; font-size: 13px;">
                AULAS PROGRAMADAS
            </div>
        """, unsafe_allow_html=True)
        for ag in st.session_state.agenda:
            st.markdown('<div class="metro-card">', unsafe_allow_html=True)
            ca1, ca2, ca3 = st.columns([3, 2, 2])
            ca1.markdown(f"**{ag['aluno']}** (`{ag['instrumento']}`)")
            ca2.markdown(f"🗓️ {ag['data']} às {ag['horario']}")
            if ag["status"] == "Agendada":
                if ca3.button("CONCLUIR", key=f"btn_done_{ag['id']}", use_container_width=True):
                    ag["status"] = "Realizada"
                    for al in st.session_state.alunos:
                        if al["nome"] == ag["aluno"] and al["aulas_feitas"] < al["aulas_totais"]:
                            if "historico_aulas" not in al:
                                al["historico_aulas"] = []
                            novo_id_h = max([h["id"] for h in al["historico_aulas"]], default=0) + 1
                            al["historico_aulas"].append({"id": novo_id_h, "data": str(ag["data"])})
                    st.rerun()
            else:
                ca3.markdown("<span style='color: #107c41; font-weight: 700;'>✔ CONCLUÍDA</span>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# TAB 4: DIAGNÓSTICO IA
# ----------------------------------------------------
with tab_predicao:
    st.markdown("### Motor Preditivo de Evasão")
    origem = st.radio("SELECIONE O MODO DE OPERAÇÃO:", ["Carregar Aluno Cadastrado", "Simulação Manual"], horizontal=True)

    st.markdown('<div class="metro-card">', unsafe_allow_html=True)
    if origem == "Carregar Aluno Cadastrado" and st.session_state.alunos:
        aluno_nome = st.selectbox("ALUNO REGISTRADO", [a["nome"] for a in st.session_state.alunos], key="sel_aluno_pred")
        aluno_selecionado = next(a for a in st.session_state.alunos if a["nome"] == aluno_nome)

        val_inst = aluno_selecionado["instrumento"]
        val_faixa = aluno_selecionado["faixa_etaria"]
        val_tempo = aluno_selecionado["tempo_matricula"]
        val_freq = aluno_selecionado["frequencia"]
        val_atrasos = 3 if aluno_selecionado["status_pagamento"] == "Inadimplente" else (1 if aluno_selecionado["status_pagamento"] == "Pendente" else 0)
        val_canceladas = max(0, aluno_selecionado["aulas_totais"] - aluno_selecionado["aulas_feitas"] - 2)
        val_evolucao = aluno_selecionado.get("evolucao", 7.0)
        val_estudo = aluno_selecionado["horas_estudo"]

        st.info(f"Dados carregados: Pagamento `{aluno_selecionado['status_pagamento']}` | Frequência `{val_freq}%` | Estudo `{val_estudo}h/sem`")
    else:
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            val_inst = st.selectbox("INSTRUMENTO", ["Violino", "Piano"], key="man_inst")
            val_faixa = st.selectbox("FAIXA ETÁRIA", ["Infantil (6-12)", "Jovem (13-17)", "Adulto (18-59)", "Sênior (60+)"], key="man_faixa")
            val_tempo = st.slider("TEMPO DE MATRÍCULA (MESES)", 1, 48, 6, key="man_tempo")
        with col_s2:
            val_freq = st.slider("TAXA DE FREQUÊNCIA (%)", 0.0, 100.0, 75.0, key="man_freq")
            val_canceladas = st.number_input("AULAS CANCELADAS (3 MESES)", min_value=0, max_value=12, value=1, key="man_canc")
            val_atrasos = st.number_input("HISTÓRICO DE ATRASOS", min_value=0, max_value=10, value=0, key="man_atra")
        with col_s3:
            val_evolucao = st.slider("EVOLUÇÃO TÉCNICA (1 A 10)", 1.0, 10.0, 6.5, step=0.5, key="man_evol")
            val_estudo = st.slider("HORAS DE ESTUDO / SEMANA", 0.0, 15.0, 3.0, step=0.5, key="man_est")

    if st.button("⚡ PROCESSAR DIAGNÓSTICO", key="btn_rodar_ia", use_container_width=True):
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

        st.markdown("<br>", unsafe_allow_html=True)
        col_res1, col_res2 = st.columns([1.2, 2])
        with col_res1:
            if probabilidade < 35:
                st.markdown(f"""
                    <div class="metro-tile metro-tile-green">
                        <div style="font-size: 11px; text-transform: uppercase;">STATUS DO ALUNO</div>
                        <div style="font-size: 28px; font-weight: 300;">{probabilidade:.1f}%</div>
                        <div style="font-size: 13px;">Baixa probabilidade de evasão</div>
                    </div>
                """, unsafe_allow_html=True)
            elif probabilidade < 65:
                st.markdown(f"""
                    <div class="metro-tile metro-tile-orange">
                        <div style="font-size: 11px; text-transform: uppercase;">STATUS DO ALUNO</div>
                        <div style="font-size: 28px; font-weight: 300;">{probabilidade:.1f}%</div>
                        <div style="font-size: 13px;">Risco Moderado</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="metro-tile metro-tile-red">
                        <div style="font-size: 11px; text-transform: uppercase;">STATUS DO ALUNO</div>
                        <div style="font-size: 28px; font-weight: 300;">{probabilidade:.1f}%</div>
                        <div style="font-size: 13px;">Risco Crítico de Evasão</div>
                    </div>
                """, unsafe_allow_html=True)

        with col_res2:
            if probabilidade >= 50:
                st.warning("Ação Recomendada: Realizar contato preventivo com a coordenação pedagógica, flexibilizar cobranças ou reavaliar o ritmo de aprendizado.")
            else:
                st.success("Ação Recomendada: Aluno com adesão satisfatória. Manter planejamento e plano de aulas atual.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# TAB 5: BI & DADOS HISTÓRICOS
# ----------------------------------------------------
with tab_bi:
    st.markdown("### Painel de Inteligência de Negócio")

    kb1, kb2, kb3 = st.columns(3)
    kb1.markdown(f"""
        <div class="metro-tile metro-tile-teal">
            <div style="font-size: 11px; text-transform: uppercase; font-weight: 700;">AMOSTRAS HISTÓRICAS</div>
            <div style="font-size: 28px; font-weight: 300; margin: 5px 0;">{len(df_historico)}</div>
            <div style="font-size: 12px; opacity: 0.85;">Registros de calibração</div>
        </div>
    """, unsafe_allow_html=True)

    kb2.markdown(f"""
        <div class="metro-tile metro-tile-purple">
            <div style="font-size: 11px; text-transform: uppercase; font-weight: 700;">ACURÁCIA IA</div>
            <div style="font-size: 28px; font-weight: 300; margin: 5px 0;">{acuracia_ia * 100:.1f}%</div>
            <div style="font-size: 12px; opacity: 0.85;">Random Forest Classifier</div>
        </div>
    """, unsafe_allow_html=True)

    kb3.markdown(f"""
        <div class="metro-tile metro-tile-blue">
            <div style="font-size: 11px; text-transform: uppercase; font-weight: 700;">TAXA DE EVASÃO BASE</div>
            <div style="font-size: 28px; font-weight: 300; margin: 5px 0;">{(df_historico['Evasao'].mean() * 100):.1f}%</div>
            <div style="font-size: 12px; opacity: 0.85;">Média da série histórica</div>
        </div>
    """, unsafe_allow_html=True)

    col_bi1, col_bi2 = st.columns(2)
    with col_bi1:
        fig_hist_freq = px.histogram(
            df_historico,
            x="Taxa_Frequencia_Pct",
            color=df_historico["Evasao"].map({0: "Ativo", 1: "Evadido"}),
            barmode="overlay",
            title="Distribuição de Frequência x Status de Evasão",
            color_discrete_map={"Ativo": "#0078d7", "Evadido": "#e81123"},
            labels={"Taxa_Frequencia_Pct": "Frequência (%)", "color": "Status"}
        )
        fig_hist_freq.update_layout(font_family="Segoe UI")
        st.plotly_chart(fig_hist_freq, use_container_width=True)

    with col_bi2:
        importancias = pd.DataFrame({
            "Variável": colunas_modelo,
            "Importância": modelo_ia.feature_importances_
        }).sort_values(by="Importância", ascending=True)

        fig_imp = px.bar(
            importancias,
            x="Importância",
            y="Variável",
            orientation="h",
            title="Importância dos Fatores de Decisão (IA)",
            color_discrete_sequence=["#008272"]
        )
        fig_imp.update_layout(font_family="Segoe UI")
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("#### Amostra da Base de Dados")
    st.dataframe(df_historico.head(50), use_container_width=True)
