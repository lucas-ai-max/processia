import streamlit as st
from pathlib import Path
from processor import DocumentProcessor
from vectorstore import VectorStore
from analyzer import DocumentAnalyzer
from storage import ResponseStorage
from file_manager import FileManager
import glob
import os
from datetime import datetime
import pandas as pd
import json
import logging
import warnings
import sys

# Configurar logging para filtrar erros de WebSocket do Tornado
# Definir nível CRITICAL para suprimir completamente os logs do Tornado
tornado_loggers = [
    "tornado.access",
    "tornado.application", 
    "tornado.general",
    "tornado.websocket",
    "tornado.iostream",
    "tornado.curl_httpclient",
    "tornado.httpclient",
    "tornado",
]

for logger_name in tornado_loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.CRITICAL)
    # Adicionar handler que descarta todas as mensagens
    logger.propagate = False
    # Remover handlers existentes e adicionar NullHandler
    logger.handlers = []
    null_handler = logging.NullHandler()
    logger.addHandler(null_handler)

# Suprimir também os handlers de root logger que possam estar propagando mensagens do Tornado
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    # Não remover, mas configurar para filtrar mensagens do Tornado
    if hasattr(handler, 'addFilter'):
        class TornadoFilter(logging.Filter):
            def filter(self, record):
                # Filtrar mensagens relacionadas ao Tornado
                if hasattr(record, 'name') and record.name and 'tornado' in record.name.lower():
                    return False
                if hasattr(record, 'pathname') and record.pathname and 'tornado' in record.pathname.lower():
                    return False
                if hasattr(record, 'module') and record.module and 'tornado' in record.module.lower():
                    return False
                # Filtrar mensagens específicas
                msg = str(record.msg).lower() if record.msg else ''
                if any(keyword in msg for keyword in ['websocketclosederror', 'streamclosederror', 'task exception was never retrieved']):
                    return False
                return True
        handler.addFilter(TornadoFilter())

# Suprimir warnings de WebSocket fechado
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*WebSocket.*")
warnings.filterwarnings("ignore", message=".*Task exception was never retrieved.*")
warnings.filterwarnings("ignore", message=".*StreamClosedError.*")
warnings.filterwarnings("ignore", message=".*WebSocketClosedError.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tornado.*")

# Custom excepthook para filtrar erros de WebSocket do Tornado
_original_excepthook = sys.excepthook

def _custom_excepthook(exc_type, exc_value, exc_traceback):
    """Filtra erros de WebSocket do Tornado que não são críticos"""
    try:
        # Verificar se é um erro relacionado ao Tornado
        module_name = getattr(exc_type, '__module__', None) or ''
        if "tornado" in module_name:
            error_name = getattr(exc_type, '__name__', '')
            # Ignorar erros de WebSocket e Stream fechados
            if error_name in ("WebSocketClosedError", "StreamClosedError"):
                return
            # Ignorar se a mensagem contém referências a WebSocket/Stream fechados
            if exc_value and isinstance(exc_value, Exception):
                error_msg = str(exc_value).lower()
                if any(keyword in error_msg for keyword in ["websocket", "stream closed", "stream is closed"]):
                    return
        
        # Para outros erros, usar o handler original
        _original_excepthook(exc_type, exc_value, exc_traceback)
    except Exception:
        # Se houver qualquer erro no handler customizado, usar o original como fallback
        try:
            _original_excepthook(exc_type, exc_value, exc_traceback)
        except:
            pass

# Aplicar o excepthook customizado apenas se não estiver em modo de desenvolvimento rigoroso
# IMPORTANTE: Desabilitar temporariamente para evitar problemas de renderização
# O filtro de logging já é suficiente para suprimir os erros do Tornado
# if not os.environ.get("DEBUG_STREAMLIT"):
#     sys.excepthook = _custom_excepthook

# Redirecionar stderr para filtrar mensagens do Tornado antes de exibir
_original_stderr = sys.stderr

class FilteredStderr:
    """Wrapper para stderr que filtra mensagens do Tornado"""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.buffer = ""
        self.max_buffer_size = 10000  # Limitar tamanho do buffer
    
    def write(self, text):
        """Escreve apenas se não for mensagem do Tornado"""
        if not text:
            return
            
        # Acumular texto no buffer
        self.buffer += text
        
        # Se o buffer ficar muito grande, limpar e escrever tudo
        if len(self.buffer) > self.max_buffer_size:
            accumulated = self.buffer
            self.buffer = ""
            if not self._should_filter(accumulated):
                self.original_stderr.write(accumulated)
            return
        
        # Se o texto termina com nova linha ou é um traceback completo, verificar
        if text.endswith('\n') or 'traceback' in self.buffer.lower():
            # Verificar se deve filtrar o buffer acumulado
            if self._should_filter(self.buffer):
                self.buffer = ""
                return
            else:
                # Escrever buffer e limpar
                accumulated = self.buffer
                self.buffer = ""
                self.original_stderr.write(accumulated)
    
    def _should_filter(self, text):
        """Verifica se o texto deve ser filtrado"""
        if not text:
            return False
            
        text_lower = text.lower()
        
        # Palavras-chave que indicam erros do Tornado
        tornado_keywords = [
            'websocketclosederror',
            'streamclosederror', 
            'stream is closed',
            'task exception was never retrieved',
            'tornado.websocket',
            'tornado.iostream',
            'raise websocketclosederror',
        ]
        
        # Verificar se contém keywords do Tornado
        has_tornado_keywords = any(keyword in text_lower for keyword in tornado_keywords)
        has_tornado_module = 'tornado' in text_lower and ('file "' in text_lower or 'traceback' in text_lower)
        
        # Filtrar se for claramente um erro do Tornado
        if has_tornado_keywords or has_tornado_module:
            # Verificar se é realmente relacionado a WebSocket fechado
            if any(err in text_lower for err in ['websocketclosederror', 'streamclosederror', 'stream is closed']):
                return True
            # Verificar se é "task exception was never retrieved" relacionado ao Tornado
            if 'task exception was never retrieved' in text_lower and 'tornado' in text_lower:
                return True
        
        return False
    
    def flush(self):
        if self.buffer:
            if not self._should_filter(self.buffer):
                self.original_stderr.write(self.buffer)
            self.buffer = ""
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)

# Aplicar filtro apenas se não estiver em modo debug
# IMPORTANTE: Não aplicar o filtro de stderr pois pode interferir com o Streamlit
# O filtro de logging e excepthook já são suficientes
# if not os.environ.get("DEBUG_STREAMLIT"):
#     sys.stderr = FilteredStderr(_original_stderr)

# Configurar asyncio para não mostrar warnings de exceções não tratadas em tasks
try:
    import asyncio
    
    # Função customizada para lidar com exceções não tratadas em callbacks
    def _custom_exception_handler(loop, context):
        """Filtra exceções de WebSocket do Tornado em callbacks assíncronos"""
        try:
            exception = context.get('exception')
            if exception:
                # Verificar se é um erro relacionado ao Tornado WebSocket
                exc_type = type(exception)
                module_name = getattr(exc_type, '__module__', None) or ''
                if "tornado" in module_name:
                    error_name = getattr(exc_type, '__name__', '')
                    if error_name in ("WebSocketClosedError", "StreamClosedError"):
                        return  # Ignorar silenciosamente
                
                # Verificar mensagem de contexto
                message = str(context.get('message', '')).lower()
                if any(keyword in message for keyword in ["websocket", "stream closed"]):
                    return  # Ignorar silenciosamente
            
            # Para outros erros, usar comportamento padrão (mas silenciar tasks do Tornado)
            message_str = str(context.get('message', ''))
            if 'Task exception was never retrieved' in message_str:
                # Verificar se a exceção é relacionada ao Tornado
                if exception:
                    module_name = getattr(type(exception), '__module__', None) or ''
                    if "tornado" in module_name:
                        return  # Ignorar tasks do Tornado
            # Não fazer nada - deixar comportamento padrão para outros erros
        except Exception:
            # Se houver erro no handler, simplesmente ignorar
            pass
    
    # Tentar configurar o exception handler quando o loop for criado
    # Usar um hook para quando o Streamlit criar o loop
    def _setup_asyncio_handler():
        try:
            loop = asyncio.get_running_loop()
            if loop and not loop.is_closed():
                loop.set_exception_handler(_custom_exception_handler)
        except RuntimeError:
            # Sem loop rodando ainda, tentar novamente depois
            pass
    
    # Tentar configurar imediatamente se houver loop
    try:
        _setup_asyncio_handler()
    except:
        pass
        
except ImportError:
    pass

# #region agent log
def debug_log(location, message, data, hypothesis_id="A", session_id="debug-session", run_id="run1"):
    """Debug logging helper"""
    try:
        log_entry = {
            "sessionId": session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        log_path = os.path.join(os.path.dirname(__file__), ".cursor", "debug.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        # Fallback: tentar escrever no diretório atual
        try:
            with open("debug.log", "a", encoding="utf-8") as f:
                f.write(json.dumps({"error": str(e), "original": log_entry}, ensure_ascii=False) + "\n")
        except:
            pass
# Teste inicial
debug_log("app.py:31", "DEBUG LOG INICIALIZADO", {"test": True}, "TEST")
# #endregion

# Configuração da página
st.set_page_config(
    page_title="Análise de Processos Jurídicos",
    page_icon="⚖️",
    layout="wide"
)

# Título
st.title("⚖️ Análise Inteligente de Processos Jurídicos")
st.markdown("**Powered by GPT-4.1 + pypdf + Supabase**")
st.markdown("---")

# Inicializar componentes
@st.cache_resource
def init_components():
    try:
        vectorstore = VectorStore()
        return {
            "processor": DocumentProcessor(),
            "vectorstore": vectorstore,
            "analyzer": DocumentAnalyzer(vectorstore=vectorstore),
            "storage": ResponseStorage(),
            "file_manager": FileManager()
        }
    except Exception as e:
        st.error(f"❌ Erro ao inicializar componentes: {str(e)}")
        st.warning("⚠️ Verifique se o arquivo .env está configurado corretamente")
        return None

components = init_components()

if components is None:
    st.error("⚠️ Não foi possível inicializar a aplicação. Verifique o arquivo .env")
    st.stop()

# Inicializar estado da sessão
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'logs' not in st.session_state:
    st.session_state.logs = []

# Sidebar - Configurações
st.sidebar.header("⚙️ Configurações")

folder_path = st.sidebar.text_input(
    "📁 Caminho da Pasta com PDFs",
    placeholder=r"E:\Documentos\PDFs",
    help="Digite o caminho completo da pasta contendo os arquivos PDF"
)

batch_size = st.sidebar.number_input(
    "📦 Quantidade de documentos por lote",
    min_value=1,
    max_value=100,
    value=5,
    help="Quantos documentos processar por vez"
)

# Botões de controle
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎮 Controles")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_button = st.button("▶️ Iniciar", type="primary", disabled=st.session_state.processing or not folder_path, use_container_width=True)
with col2:
    stop_button = st.button("⏹️ Parar", disabled=not st.session_state.processing, use_container_width=True)

if stop_button:
    st.session_state.stop_requested = True
    st.sidebar.warning("⚠️ Parada solicitada...")

# Estatísticas na sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Estatísticas")
try:
    all_files = components["file_manager"].get_all()
    if all_files:
        status_counts = {}
        for file_data in all_files:
            status = file_data.get("status", "PENDENTE")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        st.sidebar.metric("✅ Concluídos", status_counts.get("CONCLUIDO", 0))
        st.sidebar.metric("⏳ Processando", status_counts.get("PROCESSANDO", 0))
        st.sidebar.metric("⏸️ Pendentes", status_counts.get("PENDENTE", 0))
        st.sidebar.metric("❌ Erros", status_counts.get("ERRO", 0))
        st.sidebar.metric("✓ Já Processados", status_counts.get("JA_PROCESSADO", 0))
except:
    pass

# Função para adicionar log
def add_log(message, level="INFO"):
    """Adiciona mensagem ao log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{timestamp}] {level}: {message}")

# Funções helper para atualizações seguras de UI (evitam erros de WebSocket fechado)
def safe_update_logs(log_display, logs, max_lines=50):
    """Atualiza logs de forma segura, ignorando erros de WebSocket fechado"""
    try:
        if log_display is not None:
            log_text = "\n".join(logs[-max_lines:])
            log_display.code(log_text, language="text")
    except (Exception, RuntimeError, AttributeError, TypeError):
        # Ignorar erros de WebSocket fechado e outros erros assíncronos
        pass
    except:
        # Capturar qualquer outro erro silenciosamente
        pass

def safe_update_progress(progress_bar, value):
    """Atualiza barra de progresso de forma segura"""
    try:
        if progress_bar is not None:
            progress_bar.progress(min(max(value, 0.0), 1.0))
    except (Exception, RuntimeError, AttributeError, TypeError):
        # Ignorar erros de WebSocket fechado e outros erros assíncronos
        pass
    except:
        # Capturar qualquer outro erro silenciosamente
        pass

def safe_streamlit_call(func, *args, **kwargs):
    """Chama funções do Streamlit de forma segura"""
    try:
        return func(*args, **kwargs)
    except (Exception, RuntimeError, AttributeError, TypeError):
        # Ignorar erros de WebSocket fechado e outros erros assíncronos
        pass
    except:
        # Capturar qualquer outro erro silenciosamente
        pass

def safe_rerun():
    """Chama st.rerun() de forma segura com fallback para versões antigas"""
    try:
        # Tentar usar st.rerun() (Streamlit >= 1.18.0)
        if hasattr(st, 'rerun'):
            st.rerun()
        # Fallback para versões antigas
        elif hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
        else:
            # Se nenhum método estiver disponível, não fazer nada
            pass
    except (Exception, RuntimeError, AttributeError, TypeError):
        # Ignorar erros de WebSocket fechado ao fazer rerun
        pass
    except:
        # Capturar qualquer outro erro silenciosamente
        pass

# Criar tabs
tab1, tab2 = st.tabs(["📤 Processamento", "📚 Documentos Analisados"])

# Aba 1: Processamento
with tab1:
    # Opção: Upload de arquivo único ou processar pasta
    st.markdown("### 📤 Escolha o método de processamento")
    opcao_processamento = st.radio(
        "Selecione:",
        ["📁 Processar pasta", "📄 Fazer upload de arquivo único"],
        horizontal=True
    )
    
    st.markdown("---")
    
    # Upload de arquivo único
    if opcao_processamento == "📄 Fazer upload de arquivo único":
        uploaded_file = st.file_uploader(
            "Escolha um arquivo PDF",
            type=["pdf"],
            help="Faça upload de um único arquivo PDF para processar"
        )
        
        if uploaded_file is not None:
            # Salvar arquivo temporariamente
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_path = tmp_file.name
            
            filename = uploaded_file.name
            file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
            
            st.success(f"✅ Arquivo carregado: **{filename}** ({file_size_mb:.2f} MB)")
            
            # Verificar se já foi processado
            try:
                existing = components["file_manager"].get_by_filename(filename)
                if existing and existing.get("status") == "CONCLUIDO":
                    st.info(f"ℹ️ **{filename}** já foi processado anteriormente.")
                    if st.button(f"🔄 Reprocessar {filename}", type="primary"):
                        # Resetar status para processar novamente
                        components["file_manager"].update_status(filename, "PENDENTE", existing_data=existing)
                        safe_rerun()
                elif existing and existing.get("status") == "PROCESSANDO":
                    st.warning(f"⚠️ **{filename}** está sendo processado. Aguarde a conclusão.")
                else:
                    # Registrar no banco se necessário
                    if not existing:
                        components["file_manager"].register_file(filename, file_size_mb, tmp_path)
                    
                    # Botão para processar
                    if st.button(f"▶️ Processar {filename}", type="primary", disabled=st.session_state.processing):
                        st.session_state.processing = True
                        st.session_state.stop_requested = False
                        st.session_state.logs = []
                        
                        add_log(f"Iniciando processamento de {filename}")
                        
                        # Logs em tempo real
                        st.markdown("### 📋 Logs do Processamento")
                        log_display = st.empty()
                        progress_bar = st.progress(0)
                        
                        chunks_count = [0]
                        
                        try:
                            # Atualizar status para PROCESSANDO
                            components["file_manager"].update_status(filename, "PROCESSANDO")
                            add_log(f"Iniciando: {filename}")
                            
                            safe_update_logs(log_display, st.session_state.logs)
                            
                            def save_chunks_batch(chunks_batch):
                                chunks_count[0] += len(chunks_batch)
                                add_log(f"{filename}: {chunks_count[0]} chunks processados")
                                components["vectorstore"].store_chunks(chunks_batch)
                            
                            add_log(f"{filename}: Extraindo texto...")
                            safe_update_progress(progress_bar, 0.1)
                            
                            doc = components["processor"].process_incremental(
                                tmp_path,
                                filename=filename,
                                chunk_callback=save_chunks_batch,
                                batch_size=50
                            )
                            
                            total_pages = doc.get('metadata', {}).get('total_pages', 0)
                            add_log(f"{filename}: {total_pages} páginas, {chunks_count[0]} chunks")
                            
                            # Análise automática
                            if chunks_count[0] > 0:
                                add_log(f"{filename}: Iniciando análise RAG...")
                                safe_update_progress(progress_bar, 0.7)
                                
                                try:
                                    add_log(f"{filename}: Chamando GPT-4.1 para análise...")
                                    
                                    analise_result, resposta_bruta = components["analyzer"].analyze_full_document_rag(
                                        doc["document_id"],
                                        filename,
                                        return_raw_response=True
                                    )
                                    
                                    add_log(f"{filename}: Análise GPT-4.1 concluída. Resposta: {len(resposta_bruta)} chars")
                                    
                                    # Mostrar resposta da IA
                                    safe_streamlit_call(st.markdown, f"#### 🤖 Resposta da IA - {filename}")
                                    try:
                                        with st.expander("📄 Ver resposta completa", expanded=False):
                                            safe_streamlit_call(st.markdown, resposta_bruta)
                                    except Exception:
                                        pass
                                    
                                    safe_update_progress(progress_bar, 0.9)
                                    add_log(f"{filename}: Salvando no banco...")
                                    
                                    # Salvar análise
                                    try:
                                        components["storage"].save_analysis(**analise_result)
                                        add_log(f"{filename}: Análise salva com sucesso")
                                    except Exception as save_error:
                                        add_log(f"{filename}: ERRO ao salvar - {str(save_error)}", "ERROR")
                                        raise
                                    
                                    # Atualizar status para CONCLUIDO
                                    components["file_manager"].update_status(
                                        filename,
                                        "CONCLUIDO",
                                        document_id=doc["document_id"],
                                        total_chunks=chunks_count[0],
                                        total_pages=total_pages
                                    )
                                    add_log(f"{filename}: Status CONCLUIDO atualizado")
                                    
                                    safe_update_progress(progress_bar, 1.0)
                                    add_log(f"✅ {filename} concluído!")
                                    
                                    safe_streamlit_call(st.success, f"✅ **{filename}** concluído! ({chunks_count[0]} chunks)")
                                    safe_streamlit_call(st.balloons)
                                    
                                except Exception as e:
                                    error_msg = str(e)
                                    add_log(f"{filename}: ERRO na análise - {error_msg}", "ERROR")
                                    components["file_manager"].update_status(
                                        filename,
                                        "ERRO",
                                        error_message=f"Erro análise: {error_msg[:200]}"
                                    )
                                    safe_streamlit_call(st.error, f"❌ **{filename}**: {error_msg}")
                            else:
                                add_log(f"{filename}: Nenhum chunk criado", "WARNING")
                                components["file_manager"].update_status(
                                    filename,
                                    "ERRO",
                                    error_message="Nenhum chunk foi criado"
                                )
                                safe_streamlit_call(st.error, f"❌ **{filename}**: Nenhum chunk foi criado")
                            
                        except Exception as e:
                            error_msg = str(e)
                            add_log(f"{filename}: ERRO no processamento - {error_msg}", "ERROR")
                            components["file_manager"].update_status(
                                filename,
                                "ERRO",
                                error_message=f"Erro processamento: {error_msg[:200]}"
                            )
                            safe_streamlit_call(st.error, f"❌ **{filename}**: {error_msg}")
                        finally:
                            st.session_state.processing = False
                            safe_update_logs(log_display, st.session_state.logs)
                            
                            # Limpar arquivo temporário
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                            
                            safe_rerun()
            except Exception as e:
                st.error(f"❌ Erro ao verificar status: {str(e)}")
    
    # Processar pasta (código existente)
    elif opcao_processamento == "📁 Processar pasta":
        # Área principal
        if folder_path and os.path.exists(folder_path):
            # Buscar PDFs na pasta
            pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
            
            if pdf_files:
                # Ordenar por tamanho (menor para maior)
                pdf_files_with_size = [(f, os.path.getsize(f)) for f in pdf_files]
                pdf_files_with_size.sort(key=lambda x: x[1])  # Ordenar por tamanho
                pdf_files = [f[0] for f in pdf_files_with_size]  # Pegar apenas os caminhos
                
                st.success(f"✅ {len(pdf_files)} arquivo(s) PDF encontrado(s) na pasta (ordenados do menor para o maior)")
                
                # Registrar arquivos no banco se necessário
                for pdf_file in pdf_files:
                    filename = os.path.basename(pdf_file)
                    file_size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
                    
                    try:
                        # Verificar se já existe no banco
                        existing = components["file_manager"].get_by_filename(filename)
                        if not existing:
                            # Registrar novo arquivo
                            try:
                                components["file_manager"].register_file(filename, file_size_mb, pdf_file)
                            except Exception as e:
                                pass  # Ignorar se já existe ou erro de conexão
                        elif existing and existing.get("status") not in ["CONCLUIDO", "JA_PROCESSADO"]:
                            # Atualizar tamanho se necessário
                            if existing.get("file_size_mb") != file_size_mb:
                                try:
                                    components["file_manager"].update_status(
                                        filename, 
                                        existing.get("status"),
                                        existing_data=existing  # Passar existing para evitar nova query
                                    )
                                except Exception:
                                    pass  # Ignorar erro de conexão
                    except Exception:
                        pass  # Continuar mesmo se houver erro
            
            # Buscar status de todos os arquivos do banco
            all_file_statuses = {}
            try:
                db_files = components["file_manager"].get_all()
                for db_file in db_files:
                    all_file_statuses[db_file["filename"]] = db_file
            except:
                pass
            
            # Mostrar tabela de documentos
            st.markdown("### 📊 Status dos Documentos")
            
            # Criar DataFrame para exibição
            docs_data = []
            for pdf_file in pdf_files:
                filename = os.path.basename(pdf_file)
                size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
                
                db_file = all_file_statuses.get(filename, {})
                status = db_file.get("status", "PENDENTE")
                
                # Formatar status para exibição
                error_msg = db_file.get('error_message') or 'Erro desconhecido'
                if isinstance(error_msg, str) and len(error_msg) > 30:
                    error_msg = error_msg[:30] + "..."
                
                status_display = {
                    "PENDENTE": "⏸️ Pendente",
                    "PROCESSANDO": "⏳ Processando",
                    "CONCLUIDO": "✅ Concluído",
                    "ERRO": f"❌ Erro: {error_msg}",
                    "JA_PROCESSADO": "✅ Já processado"
                }.get(status, status)
                
                chunks = db_file.get("total_chunks", 0)
                pages = db_file.get("total_pages", 0)
                
                docs_data.append({
                    "Arquivo": filename,
                    "Tamanho (MB)": f"{size_mb:.2f}",
                    "Status": status_display,
                    "Chunks": str(chunks) if chunks else "-",
                    "Páginas": str(pages) if pages else "-"
                })
            
            # Exibir tabela única com todos os documentos
            if docs_data:
                df = pd.DataFrame(docs_data)
                st.dataframe(df, width='stretch', height=400)
                
                # Processar documentos
                if start_button:
                    st.session_state.processing = True
                    st.session_state.stop_requested = False
                    st.session_state.logs = []
                    
                    add_log(f"Iniciando processamento em lote de até {batch_size} documentos")
                    
                    # Resetar documentos travados em PROCESSANDO de execuções anteriores
                    try:
                        stuck_files = components["file_manager"].get_all("PROCESSANDO")
                        for stuck in stuck_files:
                            components["file_manager"].update_status(
                                stuck["filename"],
                                "PENDENTE",
                                existing_data=stuck
                            )
                        if stuck_files:
                            add_log(f"Resetados {len(stuck_files)} arquivo(s) travados em PROCESSANDO")
                    except:
                        pass
                    
                    # REBUSCAR status atualizado do banco após o reset
                    all_file_statuses = {}
                    try:
                        db_files = components["file_manager"].get_all()
                        for db_file in db_files:
                            all_file_statuses[db_file["filename"]] = db_file
                        add_log(f"Status atualizado do banco: {len(all_file_statuses)} arquivo(s) carregados")
                    except Exception as e:
                        add_log(f"Erro ao rebuscar status: {str(e)}", "WARNING")
                    
                    # Logs em tempo real
                    st.markdown("---")
                    st.markdown("### 📋 Logs do Processamento")
                    log_display = st.empty()
                    progress_bar = st.progress(0)
                    
                    # Filtrar documentos pendentes (ordenados por tamanho) usando status ATUALIZADO
                    docs_to_process = []
                    for pdf_file in pdf_files:  # Já ordenados por tamanho
                        filename = os.path.basename(pdf_file)
                        db_file = all_file_statuses.get(filename, {})  # Usar status ATUALIZADO
                        status = db_file.get("status", "PENDENTE")
                        
                        if status in ["PENDENTE", "ERRO"]:
                            docs_to_process.append(pdf_file)
                            add_log(f"Adicionado à fila: {filename} (status: {status})", "DEBUG")
                            if len(docs_to_process) >= batch_size:
                                break
                    
                    add_log(f"Documentos pendentes selecionados: {len(docs_to_process)} (ordenados do menor para o maior)")
                    total_docs = len(docs_to_process)
                    
                    if total_docs == 0:
                        st.info("ℹ️ Nenhum documento pendente para processar!")
                        st.session_state.processing = False
                    else:
                        for idx, pdf_file in enumerate(docs_to_process):
                            if st.session_state.stop_requested:
                                add_log("⏹️ Processamento interrompido pelo usuário", "WARNING")
                                # Resetar status de documentos que estavam processando
                                for remaining_file in docs_to_process[idx:]:
                                    remaining_filename = os.path.basename(remaining_file)
                                    try:
                                        existing = components["file_manager"].get_by_filename(remaining_filename)
                                        if existing and existing.get("status") == "PROCESSANDO":
                                            components["file_manager"].update_status(remaining_filename, "PENDENTE", existing_data=existing)
                                            add_log(f"Resetado {remaining_filename} para PENDENTE", "INFO")
                                    except:
                                        pass
                                break
                            
                            filename = os.path.basename(pdf_file)
                            
                            # Atualizar status para PROCESSANDO
                            try:
                                components["file_manager"].update_status(filename, "PROCESSANDO")
                            except:
                                pass
                            
                            add_log(f"[{idx+1}/{total_docs}] Iniciando: {filename}")
                            
                            # Atualizar display de logs
                            safe_update_logs(log_display, st.session_state.logs)
                            
                            # Verificar se já tem chunks/embeddings salvos
                            existing_file_data = components["file_manager"].get_by_filename(filename)
                            doc = None
                            chunks_count = [0]
                            
                            # Verificar se já tem chunks no banco
                            if existing_file_data and existing_file_data.get("document_id") and existing_file_data.get("total_chunks", 0) > 0:
                                document_id_existing = existing_file_data["document_id"]
                                
                                # Verificar se chunks realmente existem no banco
                                if components["vectorstore"].has_chunks(document_id=document_id_existing):
                                    add_log(f"[{idx+1}/{total_docs}] {filename}: Chunks já existem no banco, reutilizando...")
                                    chunks_count[0] = existing_file_data.get("total_chunks", 0)
                                    total_pages = existing_file_data.get("total_pages", 0)
                                    
                                    # Criar objeto doc simulado com o document_id existente
                                    doc = {
                                        "document_id": document_id_existing,
                                        "filename": filename,
                                        "metadata": {
                                            "total_pages": total_pages
                                        }
                                    }
                                    add_log(f"[{idx+1}/{total_docs}] {filename}: Reutilizando {chunks_count[0]} chunks existentes")
                                else:
                                    # Document_id salvo mas chunks não existem mais, reprocessar
                                    add_log(f"[{idx+1}/{total_docs}] {filename}: Document ID encontrado mas chunks não existem, reprocessando...")
                                    doc = None
                            
                            # Se não tem chunks, processar PDF
                            if doc is None:
                                chunks_count = [0]
                                
                                try:
                                    def save_chunks_batch(chunks_batch):
                                        chunks_count[0] += len(chunks_batch)
                                        add_log(f"[{idx+1}/{total_docs}] {filename}: {chunks_count[0]} chunks processados")
                                        components["vectorstore"].store_chunks(chunks_batch)
                                    
                                    add_log(f"[{idx+1}/{total_docs}] {filename}: Extraindo texto...")
                                    safe_update_progress(progress_bar, (idx + 0.1) / total_docs)
                                    
                                    doc = components["processor"].process_incremental(
                                        pdf_file,
                                        filename=filename,
                                        chunk_callback=save_chunks_batch,
                                        batch_size=50
                                    )
                                    
                                    total_pages = doc.get('metadata', {}).get('total_pages', 0)
                                    add_log(f"[{idx+1}/{total_docs}] {filename}: {total_pages} páginas, {chunks_count[0]} chunks")
                                
                                except Exception as proc_error:
                                    add_log(f"[{idx+1}/{total_docs}] {filename}: ERRO no processamento - {str(proc_error)}", "ERROR")
                                    raise
                            
                            # Continuar com análise se tiver chunks
                            try:
                                # Análise automática
                                if chunks_count[0] > 0:
                                    add_log(f"[{idx+1}/{total_docs}] {filename}: Iniciando análise RAG...")
                                    safe_update_progress(progress_bar, (idx + 0.7) / total_docs)
                                    
                                    try:
                                        add_log(f"[{idx+1}/{total_docs}] {filename}: Chamando GPT-4.1 para análise...")
                                        
                                        analise_result, resposta_bruta = components["analyzer"].analyze_full_document_rag(
                                            doc["document_id"],
                                            filename,
                                            return_raw_response=True
                                        )
                                        
                                        add_log(f"[{idx+1}/{total_docs}] {filename}: Análise GPT-4.1 concluída. Resposta: {len(resposta_bruta)} chars")
                                        
                                        # Mostrar resposta da IA
                                        safe_streamlit_call(st.markdown, f"#### 🤖 Resposta da IA - {filename}")
                                        try:
                                            with st.expander("📄 Ver resposta completa", expanded=False):
                                                safe_streamlit_call(st.markdown, resposta_bruta)
                                        except Exception:
                                            pass
                                        
                                        add_log(f"[{idx+1}/{total_docs}] {filename}: Resposta exibida")
                                        
                                        safe_update_progress(progress_bar, (idx + 0.9) / total_docs)
                                        add_log(f"[{idx+1}/{total_docs}] {filename}: Salvando no banco...")
                                        
                                        # Tentar salvar com log detalhado
                                        try:
                                            components["storage"].save_analysis(**analise_result)
                                            add_log(f"[{idx+1}/{total_docs}] {filename}: Análise salva com sucesso")
                                        except Exception as save_error:
                                            add_log(f"[{idx+1}/{total_docs}] {filename}: ERRO ao salvar - {str(save_error)}", "ERROR")
                                            raise
                                        
                                        # Contar campos extraídos
                                        campos_extraidos = len([k for k in analise_result.keys() if k.startswith('p')])
                                        add_log(f"[{idx+1}/{total_docs}] {filename}: {campos_extraidos} campos extraídos")
                                        
                                        # Mostrar número do processo
                                        if analise_result.get('numero_processo'):
                                            add_log(f"[{idx+1}/{total_docs}] {filename}: Processo {analise_result.get('numero_processo')}")
                                        else:
                                            add_log(f"[{idx+1}/{total_docs}] {filename}: ATENÇÃO - Processo não identificado!", "WARNING")
                                        
                                        # Atualizar status para CONCLUIDO
                                        add_log(f"[{idx+1}/{total_docs}] {filename}: Atualizando status...")
                                        # #region agent log
                                        debug_log("app.py:378", "ANTES update_status CONCLUIDO", {"filename": filename, "document_id": doc.get("document_id"), "chunks": chunks_count[0]}, "A")
                                        # #endregion
                                        try:
                                            status_result = components["file_manager"].update_status(
                                                filename,
                                                "CONCLUIDO",
                                                document_id=doc["document_id"],
                                                total_chunks=chunks_count[0],
                                                total_pages=total_pages
                                            )
                                            # #region agent log
                                            debug_log("app.py:367", "DEPOIS update_status CONCLUIDO", {"filename": filename, "status_result": status_result is not None, "returned_status": status_result.get("status") if status_result else None}, "A")
                                            # #endregion
                                            if status_result:
                                                add_log(f"[{idx+1}/{total_docs}] {filename}: Status CONCLUIDO atualizado")
                                            else:
                                                add_log(f"[{idx+1}/{total_docs}] {filename}: AVISO - update_status retornou None", "WARNING")
                                                # #region agent log
                                                debug_log("app.py:371", "update_status retornou None", {"filename": filename}, "B")
                                                # #endregion
                                        except Exception as status_error:
                                            add_log(f"[{idx+1}/{total_docs}] {filename}: ERRO status - {str(status_error)}", "ERROR")
                                            # #region agent log
                                            debug_log("app.py:376", "EXCEÇÃO update_status", {"filename": filename, "error": str(status_error)}, "C")
                                            # #endregion
                                        
                                        safe_update_progress(progress_bar, (idx + 1) / total_docs)
                                        add_log(f"[{idx+1}/{total_docs}] ✅ {filename} concluído!")
                                        
                                        safe_streamlit_call(st.success, f"✅ **{filename}** concluído! ({chunks_count[0]} chunks, {campos_extraidos} campos)")
                                        
                                    except Exception as e:
                                        import traceback
                                        error_msg = str(e)
                                        tb_str = traceback.format_exc()
                                        add_log(f"[{idx+1}/{total_docs}] {filename}: ERRO DETALHADO:", "ERROR")
                                        add_log(f"Mensagem: {error_msg}", "ERROR")
                                        add_log(f"Traceback: {tb_str[:500]}", "ERROR")
                                        try:
                                            components["file_manager"].update_status(
                                                filename,
                                                "ERRO",
                                                error_message=f"Erro análise: {error_msg[:200]}"
                                            )
                                            add_log(f"[{idx+1}/{total_docs}] {filename}: Status ERRO atualizado", "INFO")
                                        except Exception as status_error:
                                            add_log(f"[{idx+1}/{total_docs}] {filename}: Falha status - {str(status_error)}", "ERROR")
                                        safe_streamlit_call(st.error, f"❌ **{filename}**: {error_msg}")
                                else:
                                    add_log(f"[{idx+1}/{total_docs}] {filename}: Nenhum chunk criado", "WARNING")
                                    try:
                                        components["file_manager"].update_status(
                                            filename,
                                            "ERRO",
                                            error_message="Nenhum chunk foi criado"
                                        )
                                    except:
                                        pass
                            
                            except Exception as e:
                                import traceback
                                error_msg = str(e)
                                tb_str = traceback.format_exc()
                                add_log(f"[{idx+1}/{total_docs}] {filename}: ERRO no processamento - {error_msg}", "ERROR")
                                add_log(f"Traceback: {tb_str[:500]}", "ERROR")
                                try:
                                    components["file_manager"].update_status(
                                        filename,
                                        "ERRO",
                                        error_message=f"Erro no processamento: {error_msg[:200]}"
                                    )
                                    add_log(f"[{idx+1}/{total_docs}] {filename}: Status atualizado para ERRO", "INFO")
                                except Exception as status_error:
                                    add_log(f"[{idx+1}/{total_docs}] {filename}: Falha ao atualizar status - {str(status_error)}", "ERROR")
                                safe_streamlit_call(st.error, f"❌ **{filename}**: {error_msg}")
                            finally:
                                # GARANTIR que o status seja atualizado se ainda estiver PROCESSANDO
                                try:
                                    # #region agent log
                                    debug_log("app.py:422", "ENTRADA finally block", {"filename": filename}, "D")
                                    # #endregion
                                    current = components["file_manager"].get_by_filename(filename)
                                    # #region agent log
                                    debug_log("app.py:424", "DEPOIS get_by_filename no finally", {"filename": filename, "current_exists": current is not None, "current_status": current.get("status") if current else None, "chunks_count": chunks_count[0]}, "D")
                                    # #endregion
                                    if current and current.get("status") == "PROCESSANDO":
                                        # Se ainda está PROCESSANDO, resetar para ERRO
                                        if chunks_count[0] == 0:
                                            components["file_manager"].update_status(
                                                filename,
                                                "ERRO",
                                                error_message="Processamento incompleto - sem chunks gerados",
                                                existing_data=current
                                            )
                                            add_log(f"[{idx+1}/{total_docs}] {filename}: Status corrigido para ERRO (travado em PROCESSANDO)", "WARNING")
                                        else:
                                            # Se tem chunks mas ainda está PROCESSANDO, algo deu errado na análise
                                            components["file_manager"].update_status(
                                                filename,
                                                "ERRO",
                                                error_message="Análise não concluída",
                                                total_chunks=chunks_count[0],
                                                existing_data=current
                                            )
                                            add_log(f"[{idx+1}/{total_docs}] {filename}: Status corrigido para ERRO (análise incompleta)", "WARNING")
                                except Exception as e:
                                    add_log(f"[{idx+1}/{total_docs}] {filename}: Erro ao verificar status final - {str(e)}", "WARNING")
                                
                                # Atualizar display de logs
                                safe_update_logs(log_display, st.session_state.logs)
                    
                    # Finalizar
                    st.session_state.processing = False
                    safe_update_progress(progress_bar, 1.0)
                    add_log("✅ Processamento finalizado!")
                    safe_update_logs(log_display, st.session_state.logs)
                    
                    if not st.session_state.stop_requested:
                        safe_streamlit_call(st.balloons)
                        safe_streamlit_call(st.success, "🎉 Processamento em lote concluído!")
                    
                    safe_rerun()
        else:
            st.warning("⚠️ Nenhum arquivo PDF encontrado na pasta especificada")
    elif folder_path:
        st.error(f"❌ Caminho não encontrado: {folder_path}")
    else:
        st.info("👆 Digite o caminho da pasta com os PDFs no painel lateral")

    # Exibir logs se houver
    if st.session_state.logs:
        st.markdown("---")
        st.markdown("### 📋 Logs Recentes")
        log_text = "\n".join(st.session_state.logs[-100:])
        st.code(log_text, language="text")

# Aba 2: Visualizar Documentos Analisados
with tab2:
    st.markdown("### 📚 Documentos Analisados")
    st.markdown("Visualize e consulte as análises jurídicas já processadas.")
    
    try:
        # Buscar todas as análises
        all_analyses = components["storage"].get_all()
        
        if not all_analyses:
            st.info("📭 Nenhum documento analisado ainda. Processe alguns documentos na aba de Processamento.")
        else:
            # Filtros
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_processo = st.text_input(
                    "🔍 Filtrar por Número do Processo",
                    placeholder="0000000-00.0000.0.00.0000",
                    help="Digite o número do processo para buscar"
                )
            
            with col2:
                filter_arquivo = st.text_input(
                    "📄 Filtrar por Nome do Arquivo",
                    placeholder="Nome do arquivo",
                    help="Digite parte do nome do arquivo"
                )
            
            with col3:
                filter_juiz = st.text_input(
                    "⚖️ Filtrar por Juiz",
                    placeholder="Nome do juiz",
                    help="Digite o nome do juiz"
                )
            
            # Aplicar filtros
            filtered_analyses = all_analyses
            
            if filter_processo:
                filtered_analyses = [a for a in filtered_analyses if filter_processo.lower() in a.get("numero_processo", "").lower()]
            
            if filter_arquivo:
                filtered_analyses = [a for a in filtered_analyses if filter_arquivo.lower() in a.get("arquivo_original", "").lower()]
            
            if filter_juiz:
                filtered_analyses = [a for a in filtered_analyses if filter_juiz.lower() in (a.get("juiz") or "").lower()]
            
            st.markdown(f"**Total encontrado:** {len(filtered_analyses)} documento(s)")
            st.markdown("---")
            
            # Lista de documentos
            if filtered_analyses:
                for idx, analysis in enumerate(filtered_analyses):
                    with st.expander(
                        f"📄 {analysis.get('arquivo_original', 'Sem nome')} - {analysis.get('numero_processo', 'Sem processo')}",
                        expanded=False
                    ):
                        col_info1, col_info2 = st.columns(2)
                        
                        with col_info1:
                            st.markdown("**📋 Informações do Processo**")
                            st.markdown(f"**Número do Processo:** {analysis.get('numero_processo', 'N/A')}")
                            st.markdown(f"**Arquivo:** {analysis.get('arquivo_original', 'N/A')}")
                            st.markdown(f"**Juiz:** {analysis.get('juiz', 'N/A')}")
                            st.markdown(f"**Vara:** {analysis.get('vara', 'N/A')}")
                            st.markdown(f"**Tribunal:** {analysis.get('tribunal', 'N/A')}")
                        
                        with col_info2:
                            st.markdown("**📅 Informações da Decisão**")
                            st.markdown(f"**Data da Decisão:** {analysis.get('data_decisao', 'N/A')}")
                            st.markdown(f"**Tipo de Decisão:** {analysis.get('tipo_decisao', 'N/A')}")
                            st.markdown(f"**Analisado por:** {analysis.get('analisado_por', 'N/A')}")
                            created_at = analysis.get('created_at')
                            if created_at:
                                try:
                                    from datetime import datetime
                                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                                    st.markdown(f"**Data de Análise:** {dt.strftime('%d/%m/%Y %H:%M')}")
                                except:
                                    st.markdown(f"**Data de Análise:** {created_at}")
                        
                        # Decisão do Juiz
                        decisao_resposta = analysis.get('decisao_resposta')
                        if decisao_resposta:
                            st.markdown("---")
                            st.markdown("### 🎯 Decisão do Juiz sobre Desconsideração")
                            st.markdown(f"**Resposta:** {decisao_resposta}")
                            
                            decisao_justificativa = analysis.get('decisao_justificativa')
                            if decisao_justificativa:
                                st.markdown(f"**Justificativa:** {decisao_justificativa}")
                            
                            decisao_referencia = analysis.get('decisao_referencia')
                            if decisao_referencia:
                                st.markdown(f"**Referência:** {decisao_referencia}")
                        
                        # Listar todas as perguntas e respostas
                        st.markdown("---")
                        st.markdown("### 📝 Respostas às Perguntas")
                        
                        # Carregar perguntas do prompt
                        def carregar_perguntas():
                            perguntas_map = {}
                            try:
                                with open("prompt_analise.txt", "r", encoding="utf-8") as f:
                                    content = f.read()
                                    import re
                                    # Buscar padrão [1.1], [1.2], etc. seguido da pergunta
                                    pattern = r'\[\*?(\d+)\.(\d+)\*?\]\s+(.+?)(?=\n\n\*\*p\d+_|$)'
                                    matches = re.finditer(pattern, content, re.DOTALL)
                                    for match in matches:
                                        bloco = match.group(1)
                                        num = match.group(2)
                                        pergunta_texto = match.group(3).strip()
                                        # Limpar marcação markdown e quebras de linha
                                        pergunta_texto = re.sub(r'\*\*', '', pergunta_texto)
                                        pergunta_texto = re.sub(r'\n+', ' ', pergunta_texto).strip()
                                        key = f"p{bloco}_{num}"
                                        perguntas_map[key] = pergunta_texto
                            except Exception as e:
                                pass
                            return perguntas_map
                        
                        perguntas_map = carregar_perguntas()
                        
                        # Agrupar campos por pergunta (p1_1_resposta, p1_1_justificativa, p1_1_referencia)
                        perguntas_dict = {}
                        for key, value in analysis.items():
                            if key.startswith('p') and ('_resposta' in key or '_justificativa' in key or '_referencia' in key):
                                # Extrair ID da pergunta (ex: p1_1 de p1_1_resposta)
                                pergunta_id = key.rsplit('_', 1)[0]
                                tipo = key.split('_')[-1]
                                
                                if pergunta_id not in perguntas_dict:
                                    perguntas_dict[pergunta_id] = {}
                                perguntas_dict[pergunta_id][tipo] = value
                        
                        # Ordenar perguntas
                        sorted_perguntas = sorted(perguntas_dict.items(), key=lambda x: (
                            int(x[0].split('_')[0][1:]),  # Número do bloco (p1, p2, etc)
                            int(x[0].split('_')[1])        # Número da pergunta (1, 2, etc)
                        ))
                        
                        # Exibir cada pergunta
                        for pergunta_id, campos in sorted_perguntas:
                            # Buscar texto da pergunta ou usar ID como fallback
                            pergunta_texto = perguntas_map.get(pergunta_id, pergunta_id.upper().replace('_', '.'))
                            # Limitar tamanho do título se muito longo
                            titulo = pergunta_texto if len(pergunta_texto) <= 100 else pergunta_texto[:97] + "..."
                            
                            with st.expander(f"**{titulo}**", expanded=False):
                                if 'resposta' in campos:
                                    st.markdown(f"**Resposta:** {campos['resposta']}")
                                if 'justificativa' in campos:
                                    st.markdown(f"**Justificativa:** {campos['justificativa']}")
                                if 'referencia' in campos:
                                    st.markdown(f"**Referência:** {campos['referencia']}")
                        
                        st.markdown("---")
            
            else:
                st.warning("🔍 Nenhum documento encontrado com os filtros aplicados.")
                
    except Exception as e:
        st.error(f"❌ Erro ao buscar análises: {str(e)}")
        st.exception(e)
