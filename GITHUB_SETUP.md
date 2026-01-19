# Como fazer push para o GitHub

## ✅ Passo 1: Criar repositório no GitHub

1. Acesse https://github.com
2. Clique em "New repository" (ou https://github.com/new)
3. Nome do repositório: `ProcessIA` (ou o nome que preferir)
4. Escolha se será **Private** ou **Public**
5. **NÃO** marque "Initialize this repository with a README"
6. Clique em "Create repository"

## ✅ Passo 2: Conectar e enviar o código

Execute os seguintes comandos no terminal (substitua `SEU_USUARIO` pelo seu usuário do GitHub):

```bash
cd "E:\Projetos Cursor\ProcessIA"

# Adicionar o remote do GitHub (SUBSTITUA SEU_USUARIO e NOME_DO_REPO)
git remote add origin https://github.com/SEU_USUARIO/ProcessIA.git

# Verificar o remote
git remote -v

# Fazer o push para o GitHub
git push -u origin main
```

### Se você usar SSH ao invés de HTTPS:

```bash
git remote add origin git@github.com:SEU_USUARIO/ProcessIA.git
git push -u origin main
```

## ✅ Passo 3: Verificar

Após o push, acesse seu repositório no GitHub e verifique se todos os arquivos foram enviados corretamente.

## 📝 Arquivos que NÃO serão enviados (protegidos pelo .gitignore)

- `.env` - Suas chaves de API (nunca commite isso!)
- `venv/` - Ambiente virtual Python
- `__pycache__/` - Cache do Python
- `.cursor/` - Arquivos do Cursor IDE

## 🔐 Segurança

⚠️ **IMPORTANTE**: O arquivo `.env` com suas chaves de API está no `.gitignore` e NÃO será enviado para o GitHub. Mantenha suas chaves seguras!
