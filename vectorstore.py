from openai import OpenAI
from supabase import create_client, Client
from typing import List, Dict
from config import settings
from concurrent.futures import ThreadPoolExecutor
from postgrest.exceptions import APIError
import json
import re
import time

class VectorStore:
    """Gerenciador de embeddings otimizado"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.supabase: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_KEY
        )
    
    def _clean_text(self, text: str) -> str:
        """Remove caracteres Unicode inválidos que causam erro no PostgreSQL"""
        if not text:
            return ""
        
        # Substituir null bytes por espaço
        text = text.replace('\x00', ' ')
        text = text.replace('\u0000', ' ')
        
        # Remover outros caracteres de controle não imprimíveis
        # Mas manter \n, \r, \t (quebras de linha e tab)
        cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', ' ', text)
        
        # Normalizar espaços múltiplos
        cleaned = re.sub(r' +', ' ', cleaned)
        
        # Normalizar quebras de linha múltiplas
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        
        return cleaned.strip()
    
    def store_chunks(self, chunks: List[Dict]):
        """Armazena chunks com embeddings em paralelo"""
        
        # Limpar conteúdo de cada chunk e filtrar vazios
        cleaned_chunks = []
        for chunk in chunks:
            cleaned_chunk = chunk.copy()
            cleaned_content = self._clean_text(chunk["content"])
            
            # Só incluir chunks com conteúdo válido
            if cleaned_content and cleaned_content.strip():
                cleaned_chunk["content"] = cleaned_content
                cleaned_chunks.append(cleaned_chunk)
        
        if not cleaned_chunks:
            print("Aviso: Nenhum chunk válido para processar")
            return
        
        # Obter document_id do primeiro chunk (todos devem ter o mesmo)
        new_document_id = cleaned_chunks[0]["document_id"]
        
        # Aplicar limite de documentos (remover os mais antigos se necessário)
        self.enforce_document_limit(new_document_id)
        
        # Criar embeddings em batch (rápido)
        texts = [chunk["content"] for chunk in cleaned_chunks]
        embeddings = self._create_embeddings_batch(texts)
        
        # Preparar registros
        records = []
        for chunk, embedding in zip(cleaned_chunks, embeddings):
            # Só incluir se tiver embedding válido
            if embedding:
                records.append({
                    "document_id": chunk["document_id"],
                    "filename": chunk["filename"],
                    "page_number": chunk["page_number"],
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],  # Já limpo
                    "embedding": embedding
                })
        
        # Inserir em batch com delay antes para não sobrecarregar o banco
        if records:
            # Pequeno delay antes de inserir para evitar sobrecarga
            time.sleep(0.5)
            self._insert_batch(records)
    
    def _create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Cria embeddings em batch (econômico e rápido)"""
        all_embeddings = []
        batch_size = 100
        
        # Filtrar textos vazios ou inválidos
        valid_texts = []
        text_indices = []  # Mapear índices válidos para índices originais
        
        for idx, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_texts.append(text)
                text_indices.append(idx)
        
        if not valid_texts:
            # Retornar embeddings vazios se não houver textos válidos
            return [[] for _ in texts]
        
        # Criar embeddings apenas para textos válidos
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            
            # Validar que batch não está vazio
            if not batch:
                continue
            
            # Usar dimensions=1536 para compatibilidade com Supabase (limite de 2000)
            embedding_params = {
                "model": settings.MODEL_EMBEDDING,
                "input": batch
            }
            
            # Se usar text-embedding-3-large, reduzir para 1536 dimensões
            if "3-large" in settings.MODEL_EMBEDDING:
                embedding_params["dimensions"] = 1536
            
            try:
                response = self.client.embeddings.create(**embedding_params)
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Pequeno delay entre batches de embeddings para não sobrecarregar a API
                if i + batch_size < len(valid_texts):
                    time.sleep(0.2)  # 200ms entre batches de embeddings
            except Exception as e:
                # Log do erro e dados do batch para debug
                print(f"Erro ao criar embeddings para batch: {e}")
                print(f"Tamanho do batch: {len(batch)}")
                print(f"Primeiros caracteres do primeiro texto: {batch[0][:100] if batch else 'N/A'}")
                raise
        
        # Mapear embeddings de volta para os índices originais
        result = [[] for _ in texts]  # Inicializar com listas vazias
        embedding_idx = 0
        for valid_idx, original_idx in enumerate(text_indices):
            if embedding_idx < len(all_embeddings):
                result[original_idx] = all_embeddings[embedding_idx]
                embedding_idx += 1
        
        return result
    
    def _insert_batch(self, records: List[Dict]):
        """Insere em batch no Supabase com retry e batches menores"""
        batch_size = 3  # Reduzir para 3 registros para evitar timeout e reduzir carga
        delay_between_batches = 1.0  # Aumentar delay entre batches para 1s
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            max_retries = 3
            retry_delay = 3  # Aumentar delay inicial para 3s para dar mais tempo ao banco
            inserted = False
            
            for attempt in range(max_retries):
                try:
                    self.supabase.table(settings.TABLE_EMBEDDINGS).insert(batch).execute()
                    inserted = True
                    # Pequeno delay após inserção bem-sucedida
                    if i + batch_size < len(records):
                        time.sleep(delay_between_batches)
                    break  # Sucesso, sair do loop de retry
                except APIError as e:
                    error_code = e.code if hasattr(e, 'code') else str(getattr(e, 'message', {}))
                    
                    # Se for timeout (57014) e ainda houver tentativas
                    if error_code == '57014' and attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Backoff exponencial (2s, 4s, 8s)
                        print(f"Timeout ao inserir batch ({len(batch)} registros), tentando novamente em {wait_time}s... (tentativa {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        
                        # Reduzir batch na primeira tentativa falha
                        if attempt == 0 and len(batch) > 2:
                            mid = len(batch) // 2
                            first_half = batch[:mid]
                            second_half = batch[mid:]
                            
                            # Tentar inserir a primeira metade
                            try:
                                self.supabase.table(settings.TABLE_EMBEDDINGS).insert(first_half).execute()
                                print(f"Primeira metade ({len(first_half)} registros) inserida com sucesso")
                                time.sleep(delay_between_batches)
                                
                                # Tentar a segunda metade
                                try:
                                    self.supabase.table(settings.TABLE_EMBEDDINGS).insert(second_half).execute()
                                    print(f"Segunda metade ({len(second_half)} registros) inserida com sucesso")
                                    inserted = True
                                    break
                                except APIError as e3:
                                    print(f"Erro ao inserir segunda metade: {e3}")
                                    batch = second_half  # Continuar tentando com a segunda metade
                            except APIError as e2:
                                print(f"Erro ao inserir primeira metade: {e2}")
                                # Tentar a segunda metade mesmo se a primeira falhou
                                if len(second_half) > 0:
                                    batch = second_half
                    else:
                        # Outro erro ou esgotou tentativas
                        print(f"Erro ao inserir batch: {e}")
                        if error_code == '57014':
                            print("Timeout persistente. Tentando inserir registros individualmente com delay...")
                            # Inserir registros um por um com delay entre cada
                            success_count = 0
                            for idx, record in enumerate(batch):
                                try:
                                    self.supabase.table(settings.TABLE_EMBEDDINGS).insert(record).execute()
                                    success_count += 1
                                    # Delay entre inserções individuais
                                    if idx < len(batch) - 1:
                                        time.sleep(1.0)  # 1s entre cada registro para reduzir carga
                                except Exception as e3:
                                    print(f"Erro ao inserir registro individual {idx+1}/{len(batch)}: {e3}")
                                    # Continuar tentando os próximos
                                    time.sleep(2.0)  # Delay maior após erro (2s)
                            
                            if success_count > 0:
                                print(f"Inseridos {success_count}/{len(batch)} registros individualmente")
                            inserted = True  # Considerar como inserido mesmo se alguns falharam
                            break
                        else:
                            raise
                except Exception as e:
                    print(f"Erro inesperado ao inserir batch: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(retry_delay)
            
            if not inserted:
                print(f"Aviso: Não foi possível inserir batch de {len(batch)} registros após {max_retries} tentativas")
            else:
                # Delay entre batches para não sobrecarregar o banco
                if i + batch_size < len(records):
                    time.sleep(delay_between_batches)
    
    def search(self, query: str, document_id: str = None, limit: int = 8) -> List[Dict]:
        """Busca rápida com pgvector"""
        
        # Embedding da query
        query_embedding = self._create_embeddings_batch([query])[0]
        
        # Buscar
        params = {
            "query_embedding": query_embedding,
            "match_count": limit
        }
        
        if document_id:
            params["filter_document_id"] = document_id
        
        try:
            results = self.supabase.rpc("match_chunks", params).execute()
            return results.data if results.data else []
        except Exception as e:
            # Fallback: busca simples sem RPC
            print(f"Erro na busca RPC: {e}")
            return self._fallback_search(query, document_id, limit)
    
    def _fallback_search(self, query: str, document_id: str = None, limit: int = 8) -> List[Dict]:
        """Busca alternativa caso RPC não esteja disponível"""
        query_obj = self.supabase.table(settings.TABLE_EMBEDDINGS).select("*")
        
        if document_id:
            query_obj = query_obj.eq("document_id", document_id)
        
        results = query_obj.limit(limit).execute()
        return results.data if results.data else []
    
    def has_chunks(self, document_id: str = None, filename: str = None) -> bool:
        """Verifica se já existem chunks para um documento"""
        try:
            query = self.supabase.table(settings.TABLE_EMBEDDINGS).select("id", count="exact")
            
            if document_id:
                query = query.eq("document_id", document_id)
            elif filename:
                query = query.eq("filename", filename)
            else:
                return False
            
            result = query.limit(1).execute()
            return result.count > 0 if hasattr(result, 'count') else len(result.data) > 0
        except Exception as e:
            print(f"Erro ao verificar chunks: {e}")
            return False
    
    def get_document_id_by_filename(self, filename: str) -> str:
        """Retorna o document_id associado a um filename (se existir)"""
        try:
            result = self.supabase.table(settings.TABLE_EMBEDDINGS)\
                .select("document_id")\
                .eq("filename", filename)\
                .limit(1)\
                .execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]["document_id"]
            return None
        except Exception as e:
            print(f"Erro ao buscar document_id: {e}")
            return None
    
    def count_unique_documents(self) -> int:
        """Conta quantos documentos únicos existem na tabela"""
        try:
            result = self.supabase.table(settings.TABLE_EMBEDDINGS)\
                .select("document_id", count="exact")\
                .execute()
            return result.count if hasattr(result, 'count') else len(set(r.get('document_id') for r in result.data))
        except Exception as e:
            print(f"Erro ao contar documentos: {e}")
            return 0
    
    def get_oldest_document(self) -> str:
        """Retorna o document_id do documento mais antigo (primeiro inserido)"""
        try:
            result = self.supabase.table(settings.TABLE_EMBEDDINGS)\
                .select("document_id, created_at")\
                .order("created_at", desc=False)\
                .limit(1)\
                .execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0]["document_id"]
            return None
        except Exception as e:
            print(f"Erro ao buscar documento mais antigo: {e}")
            return None
    
    def delete_document_chunks(self, document_id: str) -> int:
        """Remove todos os chunks de um documento específico"""
        try:
            result = self.supabase.table(settings.TABLE_EMBEDDINGS)\
                .delete()\
                .eq("document_id", document_id)\
                .execute()
            
            deleted_count = len(result.data) if result.data else 0
            print(f"Removidos {deleted_count} chunks do documento {document_id}")
            return deleted_count
        except Exception as e:
            print(f"Erro ao remover chunks do documento {document_id}: {e}")
            return 0
    
    def enforce_document_limit(self, new_document_id: str):
        """Remove 3 documentos mais antigos para cada novo documento adicionado"""
        try:
            # Cache para evitar processar o mesmo documento múltiplas vezes
            if not hasattr(self, '_processed_docs'):
                self._processed_docs = set()
            
            # Se já processamos este documento nesta execução, pular
            if new_document_id in self._processed_docs:
                return
            
            # Verificar se o novo documento já existe no banco (com chunks significativos)
            try:
                result = self.supabase.table(settings.TABLE_EMBEDDINGS)\
                    .select("id", count="exact")\
                    .eq("document_id", new_document_id)\
                    .execute()
                
                chunk_count = result.count if hasattr(result, 'count') else len(result.data)
                
                # Se já tem chunks significativos (mais de 10), considerar que já existe
                if chunk_count > 10:
                    print(f"Documento {new_document_id} já existe ({chunk_count} chunks), não será necessário remover documentos antigos")
                    self._processed_docs.add(new_document_id)
                    return
            except Exception as e:
                # Se houver erro, continuar normalmente
                pass
            
            # Remover 3 documentos mais antigos para cada novo documento
            documents_to_remove = 3
            removed = 0
            
            print(f"Preparando para remover {documents_to_remove} documento(s) antigo(s) antes de adicionar o novo...")
            
            for i in range(documents_to_remove):
                oldest_doc_id = self.get_oldest_document()
                
                # Não remover o documento que está sendo inserido agora
                if oldest_doc_id and oldest_doc_id != new_document_id:
                    deleted_count = self.delete_document_chunks(oldest_doc_id)
                    if deleted_count > 0:
                        removed += 1
                        print(f"Documento antigo {i+1}/{documents_to_remove} removido: {oldest_doc_id}")
                        # Pequeno delay para não sobrecarregar o banco
                        time.sleep(0.3)
                    else:
                        # Se não conseguiu deletar, continuar tentando os próximos
                        print(f"Aviso: Não foi possível remover o documento {oldest_doc_id}")
                else:
                    # Se não há mais documentos antigos para remover
                    if not oldest_doc_id:
                        print(f"Não há mais documentos antigos para remover (total removido: {removed})")
                    break
            
            if removed > 0:
                print(f"✅ Removidos {removed} documento(s) antigo(s) antes de adicionar o novo documento")
            else:
                print(f"ℹ️ Nenhum documento antigo foi removido (tabela pode estar vazia ou só contém o documento atual)")
            
            # Marcar este documento como processado
            self._processed_docs.add(new_document_id)
                    
        except Exception as e:
            print(f"Erro ao aplicar limite de documentos: {e}")