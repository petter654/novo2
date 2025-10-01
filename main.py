import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, f1_score
import joblib
from unidecode import unidecode
import re
import os
import numpy as np
from collections import defaultdict
import warnings
import yaml
from io import BytesIO
import pandera as pa
import nltk
from nltk.corpus import stopwords

# --- CONFIGURAÇÃO INICIAL E CONSTANTES ---

# Carrega a configuração do arquivo YAML
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    st.error("Erro: O arquivo config.yaml não foi encontrado. Usando configurações padrão.")
    # Use um config padrão se o arquivo não for encontrado
    config = {
        'data_paths': {'pdm': 'bases/base_pdm.XLSX', 'familias': 'bases/familias.xlsx', 'feedback': 'bases/Base_Treinamento_AMPLIADA_FEEDBACK.xlsx'},
        'reports_path': 'reports',
        'cache_dir': 'cache',
        'cache_filenames': ['embeddings_materiais.pkl', 'nn_macro_model.pkl', 'nn_micro_models.pkl'],
        'model': {'name': 'paraphrase-multilingual-MiniLM-L12-v2', 'k_neighbors': 10, 'confidence_threshold': 0.70, 'metric': 'cosine'},
        'cleaning': {'stop_words': ['tipo', 'modelo', 'uso', 'material', 'linha', 'produto', 'descricao', 'item', 'peca'], 'remove_units': True}
    }

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Classificação de Materiais",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes globais derivadas do config
FALLBACK_LABEL = 'familia nao identificada'
REPORTS_PATH = config['reports_path']
FEEDBACK_FILE = os.path.join(REPORTS_PATH, 'Fila_Revisao_Feedback.xlsx')
CACHE_DIR = config['cache_dir']
CACHE_FILENAMES = config['cache_filenames']
CORRECT_FAMILY_COLUMN = 'Família_CORRETA'

# Constantes do modelo
MODEL_NAME = config['model']['name']
K_NEIGHBORS = config['model']['k_neighbors']
CONFIDENCE_THRESHOLD = config['model']['confidence_threshold']

# --- CORREÇÃO ROBUSTA PARA O ERRO NLTK ---
# Esta função garante que o pacote 'stopwords' seja baixado e carregado corretamente.
def load_stopwords():
    try:
        # Tenta carregar as stopwords diretamente
        return stopwords.words('portuguese')
    except LookupError:
        # Se falhar (pacote não encontrado), baixa o pacote e tenta novamente.
        st.info("Baixando pacote de dados 'stopwords' da NLTK (necessário apenas na primeira vez)...")
        nltk.download('stopwords')
        st.success("Download concluído.")
        return stopwords.words('portuguese')

stop_pt = load_stopwords()
stop_words_list = stop_pt + config['cleaning']['stop_words']


# --- FUNÇÕES DE SUPORTE E PRÉ-PROCESSAMENTO ---

def clean_text(text):
    """Limpa o texto removendo acentos, stopwords e unidades técnicas."""
    if not isinstance(text, str): return ''
    text = unidecode(text.lower())
    text = ' '.join(word for word in text.split() if word not in stop_words_list)
    if config['cleaning']['remove_units']:
        text = re.sub(r'\s(mm|m|kg|un|pc|pç|par|cj|cx|lt|l|g|pn|dn|in|c|pol|diam|ext|ref)\b', ' ', text)
    return text.strip()

def get_macro_familia(familia):
    if not isinstance(familia, str): return 'D_DIVERSOS'
    code = str(familia)[:2]
    if code in ['01', '05', '09', '24', '28', '11', '13']: return 'H_HIDRAULICA'
    if code in ['04', '07', '08', '22', '29']: return 'C_CONSTRUCAO'
    if code in ['06', '12', '19', '20', '30']: return 'E_MECANICA'
    if code in ['03', '10', '14', '15', '18', '23']: return 'Q_QUIMICOS'
    return 'D_DIVERSOS'

def get_weighted_vote_and_confidence(indices, distances, y_labels):
    predictions, confidences = [], []
    for i in range(len(indices)):
        item_indices, item_distances = indices[i], distances[i]
        if np.any(item_distances < 1e-6):
            best_label = y_labels[item_indices[np.argmin(item_distances)]]
            confidence_score = 1.0
        else:
            weights = 1 / (item_distances + 1e-6)
            vote_tally = defaultdict(float)
            for label, weight in zip(y_labels[item_indices], weights):
                vote_tally[label] += weight
            if vote_tally:
                best_label = max(vote_tally, key=vote_tally.get)
                confidence_score = vote_tally[best_label] / np.sum(weights)
            else:
                best_label = FALLBACK_LABEL
                confidence_score = 0.0
        predictions.append(best_label)
        confidences.append(confidence_score)
    return np.array(predictions), np.array(confidences)

# --- FUNÇÕES DE CARREGAMENTO DE DADOS E MODELOS ---

@st.cache_resource
def load_embedding_model():
    """Carrega o modelo SentenceTransformer do cache."""
    try:
        return SentenceTransformer(MODEL_NAME)
    except Exception as e:
        st.error(f"Erro ao carregar o modelo SentenceTransformer: {e}")
        st.stop()

@st.cache_data
def load_training_data():
    """Carrega os dados de treinamento, priorizando a base ampliada."""
    try:
        feedback_file_path = config['data_paths']['feedback']
        if os.path.exists(feedback_file_path):
            st.sidebar.success("✅ Base AMPLIADA (com Feedback) carregada.")
            df_treinamento = pd.read_excel(feedback_file_path, engine='openpyxl')
        else:
            st.sidebar.warning("⚠️ Base ORIGINAL carregada (merge PDM + Famílias).")
            df_pdm = pd.read_excel(config['data_paths']['pdm'], engine='openpyxl')
            df_familias = pd.read_excel(config['data_paths']['familias'], engine='openpyxl')
            df_treinamento = pd.merge(df_pdm, df_familias, on='Família', how='left')
        
        # --- CORREÇÃO DEFINITIVA PARA O ERRO DE VALIDAÇÃO ---
        # Remove linhas onde a coluna 'Família' é nula ANTES de validar.
        # Isso limpa dados inválidos ou linhas vazias do Excel.
        df_treinamento.dropna(subset=['Família'], inplace=True)
        
        # Validação do esquema do DataFrame
        schema = pa.DataFrameSchema({
            "Família": pa.Column(pa.String, coerce=True),
            "Decrição": pa.Column(pa.String, coerce=True, nullable=True),
            "Descrição Curta": pa.Column(pa.String, coerce=True, nullable=True),
            "Descrição Longa": pa.Column(pa.String, coerce=True, nullable=True),
        })
        df_treinamento = schema.validate(df_treinamento)
        df_treinamento['Decrição'] = df_treinamento['Decrição'].fillna(FALLBACK_LABEL)
        return df_treinamento

    except FileNotFoundError as e:
        st.error(f"❌ Arquivo de base não encontrado: {e.filename}. Verifique os caminhos no config.yaml.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao ler ou validar os arquivos de base: {e}")
        st.stop()

# --- FUNÇÕES DO APLICATIVO ---

def process_feedback_and_amplify_base():
    """Processa o arquivo de feedback, cria a base ampliada e limpa o cache."""
    if not os.path.exists(FEEDBACK_FILE):
        st.error(f"Arquivo de Feedback '{FEEDBACK_FILE}' não encontrado.")
        return

    try:
        df_feedback = pd.read_excel(FEEDBACK_FILE, engine='openpyxl')
        df_feedback.dropna(subset=[CORRECT_FAMILY_COLUMN], inplace=True)

        if df_feedback.empty:
            st.warning(f"Nenhuma linha com dados na coluna '{CORRECT_FAMILY_COLUMN}' encontrada no arquivo de feedback.")
            return

        st.success(f"Encontradas {len(df_feedback)} correções humanas para o retreinamento.")

        df_novos_materiais = pd.DataFrame({
            'Família': df_feedback[CORRECT_FAMILY_COLUMN].apply(lambda x: str(x).split(' ')[0]),
            'Decrição': df_feedback[CORRECT_FAMILY_COLUMN],
            'Descrição Curta': df_feedback['Denominação'].apply(lambda x: str(x)[:40]),
            'Descrição Longa': df_feedback['Denominação'],
        })

        df_pdm_original = pd.read_excel(config['data_paths']['pdm'], engine='openpyxl')
        df_familias_map = pd.read_excel(config['data_paths']['familias'], engine='openpyxl')
        df_base_original = pd.merge(df_pdm_original, df_familias_map, on='Família', how='left')
        
        cols_to_keep = ['Descrição Curta', 'Descrição Longa', 'Família', 'Decrição']
        df_treinamento_final = pd.concat([
            df_base_original[cols_to_keep].dropna(subset=['Decrição']),
            df_novos_materiais[cols_to_keep]
        ], ignore_index=True)
        
        feedback_path = config['data_paths']['feedback']
        os.makedirs(os.path.dirname(feedback_path), exist_ok=True)
        df_treinamento_final.to_excel(feedback_path, index=False, engine='openpyxl')
        st.success(f"✅ Base ampliada salva em: {feedback_path}")

        # Limpeza de cache
        for filename in CACHE_FILENAMES:
            filepath = os.path.join(CACHE_DIR, filename)
            if os.path.exists(filepath):
                os.remove(filepath)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("✅ Cache limpo! O modelo será retreinado na próxima execução.")
        st.rerun()

    except Exception as e:
        st.error(f"❌ Erro ao processar o feedback: {e}")

def run_classification_pipeline(df_chamados_upload, df_treinamento, model):
    """Executa o pipeline completo de classificação."""
    st.info("Iniciando o pipeline de classificação...")
    df_chamados_materiais = df_chamados_upload[df_chamados_upload['Negócio'] == 'MAT..xlsx'].copy()
    if df_chamados_materiais.empty:
        st.warning("Nenhum chamado do negócio 'MAT..xlsx' foi encontrado.")
        return
    st.success(f"Foram filtrados **{len(df_chamados_materiais)}** chamados.")

    # Garante que o diretório de cache exista
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Preparação da base de referência
    df_referencia = df_treinamento.copy()
    df_referencia['Descrição Combinada'] = (df_referencia['Descrição Curta'].fillna('') + ' ' + df_referencia['Descrição Longa'].fillna(''))
    df_referencia['Desc_Clean'] = df_referencia['Descrição Combinada'].apply(clean_text)
    df_referencia['Família_Macro'] = df_referencia['Família'].apply(get_macro_familia)
    y_labels_micro = df_referencia['Decrição'].values
    y_labels_macro = df_referencia['Família_Macro'].values

    # Geração/Carregamento de Embeddings
    embeddings_path = os.path.join(CACHE_DIR, CACHE_FILENAMES[0])
    try:
        X_all = joblib.load(embeddings_path)
        st.caption("Embeddings carregados do cache.")
    except (FileNotFoundError, EOFError):
        with st.spinner("Gerando Embeddings... (pode demorar)"):
            X_all = model.encode(df_referencia['Desc_Clean'].tolist(), batch_size=64, show_progress_bar=False)
            joblib.dump(X_all, embeddings_path)
        st.success("Embeddings gerados e salvos em cache.")

    # Treinamento/Carregamento dos modelos K-NN
    X_valid_mask = (y_labels_micro != FALLBACK_LABEL)
    X_valid, y_valid_macro, y_valid_micro = X_all[X_valid_mask], y_labels_macro[X_valid_mask], y_labels_micro[X_valid_mask]
    
    macro_model_path = os.path.join(CACHE_DIR, CACHE_FILENAMES[1])
    micro_models_path = os.path.join(CACHE_DIR, CACHE_FILENAMES[2])
    try:
        nn_macro_model = joblib.load(macro_model_path)
        nn_micro_models = joblib.load(micro_models_path)
        st.caption("Modelos K-NN carregados do cache.")
    except (FileNotFoundError, EOFError):
        with st.spinner("Treinando modelos K-NN..."):
            nn_macro_model = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric=config['model']['metric']).fit(X_valid)
            joblib.dump(nn_macro_model, macro_model_path)
            
            nn_micro_models = defaultdict(dict)
            for macro_group in np.unique(y_valid_macro):
                if macro_group in ['D_DIVERSOS', FALLBACK_LABEL]: continue
                mask_group = (y_valid_macro == macro_group)
                X_group = X_valid[mask_group]
                n_neighbors_group = min(K_NEIGHBORS, len(X_group))
                if n_neighbors_group > 0:
                    nn_model_group = NearestNeighbors(n_neighbors=n_neighbors_group, metric=config['model']['metric']).fit(X_group)
                    nn_micro_models[macro_group] = {'model': nn_model_group, 'y': y_valid_micro[mask_group]}
            joblib.dump(nn_micro_models, micro_models_path)
        st.success(f"Treinamento de {len(nn_micro_models)} micro-modelos concluído.")

    # Classificação dos chamados
    with st.spinner("Classificando os novos chamados..."):
        df_chamados_materiais['Denom_Clean'] = df_chamados_materiais['Denominação'].fillna('').apply(clean_text)
        embeddings_chamados = model.encode(df_chamados_materiais['Denom_Clean'].tolist(), batch_size=64, show_progress_bar=False)
        distances_macro, indices_macro = nn_macro_model.kneighbors(embeddings_chamados)
        y_pred_macro, _ = get_weighted_vote_and_confidence(indices_macro, distances_macro, y_valid_macro)
        df_chamados_materiais['Macro_Previsao'] = y_pred_macro

        y_pred_final, confiancas_final = [], []
        for i, row in df_chamados_materiais.iterrows():
            macro_previsao = row['Macro_Previsao']
            embedding_chamado = embeddings_chamados[df_chamados_materiais.index.get_loc(i)].reshape(1, -1)
            
            if macro_previsao in nn_micro_models:
                group_data = nn_micro_models[macro_previsao]
                distances_micro, indices_micro = group_data['model'].kneighbors(embedding_chamado)
                pred_micro, conf = get_weighted_vote_and_confidence(indices_micro, distances_micro, group_data['y'])
                y_pred_final.append(pred_micro[0])
                confiancas_final.append(conf[0])
            else:
                y_pred_final.append(FALLBACK_LABEL)
                confiancas_final.append(0.0)

        df_chamados_materiais['Família_Identificada'] = y_pred_final
        df_chamados_materiais['Confiança'] = np.round(np.array(confiancas_final), 4)

    # Avaliação e Geração de Fila de Revisão
    if 'Família' in df_chamados_materiais.columns:
        df_avaliar = df_chamados_materiais.dropna(subset=['Família']).copy()
        if not df_avaliar.empty:
            # ... (código de métricas e geração de feedback)
            st.subheader("Resultados da Classificação (Métricas):")
            # ...
            df_revisao = pd.concat([
                df_avaliar[df_avaliar['Família_Identificada'] != df_avaliar['Família']],
                df_avaliar[df_avaliar['Confiança'] < CONFIDENCE_THRESHOLD]
            ]).drop_duplicates(subset=['Denominação']).sort_values(by='Confiança')

            if not df_revisao.empty:
                os.makedirs(REPORTS_PATH, exist_ok=True)
                df_revisao[CORRECT_FAMILY_COLUMN] = '' # Adiciona a coluna vazia
                cols_to_save = ['Denominação', 'Família', 'Família_Identificada', 'Confiança', CORRECT_FAMILY_COLUMN]
                df_revisao[cols_to_save].to_excel(FEEDBACK_FILE, index=False, engine='openpyxl')
                st.success(f"**[LOOP DE FEEDBACK]** Gerados **{len(df_revisao)}** casos para revisão em **{FEEDBACK_FILE}**.")

    # Saída Final
    st.subheader("Download dos Resultados Classificados")
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_chamados_materiais.to_excel(writer, sheet_name='Classificado', index=False)
    
    st.download_button(
        label="📥 Baixar Arquivo Classificado (.xlsx)",
        data=output.getvalue(),
        file_name='chamados_classificados_hierarquico.xlsx',
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.dataframe(df_chamados_materiais[['Denominação', 'Família_Identificada', 'Confiança']].head(10))

# --- INTERFACE STREAMLIT (main) ---

def main():
    st.title("🤖 Classificação Hierárquica de Materiais")
    st.markdown("---")

    # Carregamento inicial
    model = load_embedding_model()
    df_treinamento = load_training_data()
    
    # Barra Lateral
    st.sidebar.header("1. Classificar Novos Chamados")
    uploaded_file = st.sidebar.file_uploader(
        "Carregue o arquivo de Chamados (.xlsx)", type=["xlsx"]
    )
    
    st.sidebar.header("2. Melhorar o Modelo")
    if st.sidebar.button("♻️ Processar Feedback e Retreinar", help="Lê o arquivo de revisão, atualiza a base de treinamento e limpa o cache para forçar o retreinamento."):
        process_feedback_and_amplify_base()
    
    # Corpo principal
    st.info("""
        **Como usar:**
        1. **Classificar:** Carregue um arquivo de chamados na barra lateral e clique em 'Iniciar Classificação'.
        2. **Revisar:** O sistema irá gerar um arquivo em `reports/Fila_Revisao_Feedback.xlsx`.
        3. **Corrigir:** Edite este arquivo, preenchendo a coluna **Família_CORRETA** com os valores corretos.
        4. **Retreinar:** Clique no botão 'Processar Feedback e Retreinar' na barra lateral para que o modelo aprenda com suas correções!
    """)

    if uploaded_file:
        try:
            df_chamados_upload = pd.read_excel(uploaded_file, sheet_name='Base_Geral', engine='openpyxl')
            st.sidebar.info(f"Arquivo '{uploaded_file.name}' pronto para ser processado.")
            
            if st.button("🚀 Iniciar Classificação", type="primary", use_container_width=True):
                run_classification_pipeline(df_chamados_upload, df_treinamento, model)
        
        except Exception as e:
            st.error(f"Erro ao ler o arquivo de chamados. Verifique se ele contém uma aba chamada 'Base_Geral'. Detalhes: {e}")
    else:
        st.warning("Aguardando o upload do arquivo de chamados na barra lateral.")

if __name__ == "__main__":
    main()

