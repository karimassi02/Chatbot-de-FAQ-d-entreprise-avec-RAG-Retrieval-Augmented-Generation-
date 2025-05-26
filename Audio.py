import streamlit as st
from openai import OpenAI
import tempfile
import os
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Configuration des clés API
#le projet se lance correctement lorsque j'avais les clés API définies ici
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# Ajout de la clé API Mistral 
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

# Initialisation du client OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialisation de la base de connaissances vectorielle
def initialize_vector_db():
    # Chemin vers les documents
    documents_folder = "./documents"
    
    # Liste pour stocker les documents
    docs = []
    
    # Parcourir les fichiers dans le dossier documents
    for file_name in os.listdir(documents_folder):
        file_path = os.path.join(documents_folder, file_name)
        
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                docs.append(Document(page_content=content, metadata={"source": file_name}))
    
    # Découper les documents en morceaux plus petits avec de meilleurs paramètres
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Taille réduite pour des chunks plus précis
        chunk_overlap=150,  # Chevauchement augmenté pour une meilleure continuité 
        separators=["\n\n", "\n", ". ", " ", ""],  # Séparateurs plus précis
        length_function=len
    )
    split_docs = text_splitter.split_documents(docs)
    
    # Journalisation du nombre de chunks créés
    st.sidebar.info(f"{len(split_docs)} chunks créés à partir de {len(docs)} documents")
    
    # Créer l'index vectoriel avec Chroma et les embeddings d'OpenAI
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(documents=split_docs, embedding=embeddings, persist_directory="./chroma_db")
    
    return vector_store

# Fonction pour obtenir une réponse du chatbot
def get_chatbot_response(query, vector_store, model="openai", chat_history=[]):
    # Template pour améliorer la contextualisation des réponses
    template = """
    Vous êtes un assistant d'entreprise professionnel qui répond aux questions des clients.
    Utilisez le contexte fourni pour répondre à la question de l'utilisateur de manière précise et concise.
    Si vous ne connaissez pas la réponse ou si elle ne se trouve pas dans le contexte, dites simplement que vous n'avez pas cette information.
    Ne fabriquez pas de réponse.

    Contexte de la base de connaissances:
    {context}

    Historique des conversations:
    {chat_history}

    Question de l'utilisateur: {question}

    Votre réponse:
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=template,
    )
    
    # Initialisation de la mémoire de conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )
    
    # Sélection du modèle de langage en fonction du choix utilisateur
    if model == "mistral":
        llm = ChatMistralAI(
            model="mistral-large",  # Vous pouvez utiliser "mistral-small" pour un modèle plus léger
            temperature=0.3,
            max_tokens=1024
        )
    else:  # Par défaut, utilisez OpenAI
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=1024
        )
    
    # Configurez le retriever pour améliorer la pertinence
    retriever = vector_store.as_retriever(
        search_type="mmr",  # Utilise Maximum Marginal Relevance pour diversifier les résultats
        search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.8}
    )
    
    # Configuration avancée du système RAG
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt},
        memory=memory
    )
    
    result = qa_chain.invoke({"question": query, "chat_history": chat_history})
    return result["answer"], result["source_documents"]

# Interface Streamlit
st.title("Chatbot de FAQ d'entreprise avec RAG (Retrieval Augmented Generation)")

# Configuration de la sidebar pour les paramètres
with st.sidebar:
    st.header("Configuration")
    ai_model = st.selectbox(
        "Choisir le modèle d'IA",
        ["openai", "mistral"],
        index=0,
        help="OpenAI est généralement plus rapide, Mistral peut offrir des alternatives pour certaines langues"
    )
    
    # Paramètres avancés
    with st.expander("Paramètres avancés"):
        temperature = st.slider("Température (créativité)", min_value=0.0, max_value=1.0, value=0.3, step=0.1, 
                               help="Plus la valeur est élevée, plus les réponses seront créatives mais potentiellement moins précises")
        max_tokens = st.slider("Longueur maximale des réponses", min_value=256, max_value=4096, value=1024, step=128,
                              help="Nombre maximum de tokens dans la réponse")

# Initialisation des variables de session
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = initialize_vector_db()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
# Initialisation des variables pour la gestion des documents
documents_folder = "./documents"
doc_files = [f for f in os.listdir(documents_folder) if os.path.isfile(os.path.join(documents_folder, f)) and f.endswith('.txt')]
for file_name in doc_files:
    if f"show_edit_{file_name}" not in st.session_state:
        st.session_state[f"show_edit_{file_name}"] = False
    if f"confirm_delete_{file_name}" not in st.session_state:
        st.session_state[f"confirm_delete_{file_name}"] = False

# Onglets pour les différentes fonctionnalités
tab1, tab2, tab3, tab4 = st.tabs(["Transcription Audio", "Chatbot FAQ", "Gestion Documents", "Explorer la Base de Connaissances"])

# Onglet 1: Transcription Audio
with tab1:
    st.header("Transcription Audio")
    
    # Option 1: Enregistrement audio direct
    audio_value = st.audio_input("Enregistrer un message vocal")
    if audio_value:
        st.audio(audio_value)
        with st.spinner("Transcription en cours..."):
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_value
            )
        st.write("**Transcription (enregistrement):**")
        st.write(transcription.text)
          # Ajouter un bouton pour poser la question au chatbot
        if st.button("Poser cette question au chatbot", key="ask_recorded"):
            with st.spinner("Recherche en cours..."):
                answer, sources = get_chatbot_response(
                    query=transcription.text, 
                    vector_store=st.session_state.vector_store, 
                    model=ai_model,
                    chat_history=st.session_state.chat_history
                )
                st.session_state.chat_history.append((transcription.text, answer))
            
            st.write("**Réponse du chatbot:**")
            st.write(answer)
            
            with st.expander("Sources"):
                for i, source in enumerate(sources):
                    st.write(f"**Source {i+1}:** {source.metadata['source']}")
                    st.write(source.page_content[:200] + "...")

    # Option 2: Téléchargement de fichier MP3
    st.write("---")
    st.write("### Ou téléchargez un fichier audio")
    uploaded_file = st.file_uploader("Choisissez un fichier audio (MP3, WAV, etc.)", type=["mp3", "wav", "m4a", "ogg"])

    if uploaded_file is not None:
        # Afficher le fichier audio
        st.audio(uploaded_file)
        
        # Créer un fichier temporaire pour OpenAI
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Transcrire le fichier audio
        if st.button("Transcrire le fichier", key="transcribe_file"):
            with st.spinner("Transcription en cours..."):
                with open(tmp_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file
                    )
            st.write("**Transcription (fichier téléchargé):**")
            st.write(transcription.text)
              # Ajouter un bouton pour poser la question au chatbot
            if st.button("Poser cette question au chatbot", key="ask_uploaded_file"):
                with st.spinner("Recherche en cours..."):
                    answer, sources = get_chatbot_response(
                        query=transcription.text, 
                        vector_store=st.session_state.vector_store, 
                        model=ai_model,
                        chat_history=st.session_state.chat_history
                    )
                    st.session_state.chat_history.append((transcription.text, answer))
                
                st.write("**Réponse du chatbot:**")
                st.write(answer)
                
                with st.expander("Sources"):
                    for i, source in enumerate(sources):
                        st.write(f"**Source {i+1}:** {source.metadata['source']}")
                        st.write(source.page_content[:200] + "...")

# Onglet 2: Chatbot FAQ
with tab2:
    st.header("Chatbot FAQ")
    
    # Modifier l'affichage pour inclure l'évaluation
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(answer)
            col1, col2, col3 = st.columns([1, 1, 5])
            with col1:
                if st.button("👍", key=f"like_{i}"):
                    st.success("Merci pour votre feedback positif!")
            with col2:
                if st.button("👎", key=f"dislike_{i}"):
                    st.error("Désolé pour cette réponse. Nous allons l'améliorer.")
                    feedback = st.text_area("Comment pouvons-nous améliorer cette réponse?", key=f"feedback_{i}")
                    if st.button("Envoyer feedback", key=f"send_feedback_{i}"):
                        st.info("Feedback envoyé, merci!")
    
    # Zone de texte pour poser une question
    user_question = st.text_input("Posez votre question :")
    
    if st.button("Envoyer", key="send_text_question"):
        if user_question:
            with st.spinner("Recherche en cours..."):
                answer, sources = get_chatbot_response(
                    query=user_question, 
                    vector_store=st.session_state.vector_store, 
                    model=ai_model,
                    chat_history=st.session_state.chat_history
                )
                st.session_state.chat_history.append((user_question, answer))
            
            with st.chat_message("user"):
                st.write(user_question)
            with st.chat_message("assistant"):
                st.write(answer)
                
                with st.expander("Sources"):
                    for i, source in enumerate(sources):
                        st.write(f"**Source {i+1}:** {source.metadata['source']}")
                        st.write(source.page_content[:200] + "...")
                          # Nouvelle évaluation pour la réponse fraîche
                feedback_idx = len(st.session_state.chat_history) - 1
                col1, col2, col3 = st.columns([1, 1, 5]) 
                with col1:
                    if st.button("👍", key=f"like_latest_{feedback_idx}"):
                        st.success("Merci pour votre feedback positif!")
                with col2:
                    if st.button("👎", key=f"dislike_latest_{feedback_idx}"):
                        st.error("Désolé pour cette réponse. Nous allons l'améliorer.")

# Onglet 3: Gestion des documents
with tab3:
    st.header("Gestion des documents")
    
    # Liste des documents actuels
    st.write("### Documents actuels dans la base de connaissances")
    documents_folder = "./documents"
    
    # Obtenir la liste des documents
    doc_files = [f for f in os.listdir(documents_folder) if os.path.isfile(os.path.join(documents_folder, f)) and f.endswith('.txt')]
    
    if not doc_files:
        st.warning("Aucun document trouvé dans la base de connaissances.")
    else:
        # Afficher les documents avec options pour éditer/supprimer
        for file_name in doc_files:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.write(f"{file_name}")
            
            with col2:
                # Bouton pour voir le contenu
                if st.button("Voir", key=f"view_{file_name}"):
                    file_path = os.path.join(documents_folder, file_name)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        st.session_state[f"edit_content_{file_name}"] = content
                        st.session_state[f"show_edit_{file_name}"] = True
            
            with col3:
                # Bouton pour éditer
                if st.button("Éditer", key=f"edit_{file_name}"):
                    file_path = os.path.join(documents_folder, file_name)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        st.session_state[f"edit_content_{file_name}"] = content
                        st.session_state[f"show_edit_{file_name}"] = True
            
            with col4:
                # Bouton pour supprimer
                if st.button("Supprimer", key=f"delete_{file_name}"):
                    st.session_state[f"confirm_delete_{file_name}"] = True
            
            # Afficher le contenu si demandé
            if st.session_state.get(f"show_edit_{file_name}", False):
                edit_content = st.text_area(f"Contenu de {file_name}", value=st.session_state[f"edit_content_{file_name}"], height=300, key=f"edit_area_{file_name}")
                
                col1, col2 = st.columns([1, 5])
                with col1:
                    if st.button("Sauvegarder", key=f"save_{file_name}"):
                        file_path = os.path.join(documents_folder, file_name)
                        with open(file_path, 'w', encoding='utf-8') as file:
                            file.write(edit_content)
                        st.success(f"Document {file_name} mis à jour!")
                        st.session_state[f"show_edit_{file_name}"] = False
                        
                        # Réinitialiser la base de connaissances vectorielle
                        st.session_state.vector_store = initialize_vector_db()
                        st.experimental_rerun()
                
                with col2:
                    if st.button("Annuler", key=f"cancel_{file_name}"):
                        st.session_state[f"show_edit_{file_name}"] = False
                        st.experimental_rerun()
            
            # Confirmation de suppression
            if st.session_state.get(f"confirm_delete_{file_name}", False):
                st.warning(f"Êtes-vous sûr de vouloir supprimer {file_name}?")
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("Oui, supprimer", key=f"confirm_yes_{file_name}"):
                        file_path = os.path.join(documents_folder, file_name)
                        os.remove(file_path)
                        st.error(f"Document {file_name} supprimé!")
                        
                        # Réinitialiser la base de connaissances vectorielle
                        st.session_state.vector_store = initialize_vector_db()
                        st.experimental_rerun()
                
                with col2:
                    if st.button("Annuler", key=f"confirm_no_{file_name}"):
                        st.session_state[f"confirm_delete_{file_name}"] = False
                        st.experimental_rerun()
            
            st.write("---")
    
    # Option pour ajouter un nouveau document
    st.write("### Ajouter un nouveau document")
    new_doc_name = st.text_input("Nom du document (avec extension .txt):")
    new_doc_content = st.text_area("Contenu du document:", height=200)
    
    if st.button("Ajouter ce document", key="add_doc"):
        if new_doc_name and new_doc_content:
            if not new_doc_name.endswith('.txt'):
                new_doc_name += '.txt'
            
            file_path = os.path.join(documents_folder, new_doc_name)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_doc_content)
                
            st.success(f"Document {new_doc_name} ajouté avec succès!")
            
            # Réinitialiser la base de connaissances vectorielle
            st.session_state.vector_store = initialize_vector_db()
            
            st.experimental_rerun()
              # Option pour importer un document depuis un fichier
    st.write("### Importer un document")
    uploaded_file = st.file_uploader("Importer un fichier texte", type=["txt"], key="document_uploader")
    
    if uploaded_file is not None:
        import_name = st.text_input("Nom du document à importer (laissez vide pour utiliser le nom du fichier):")
        if import_name == "":
            import_name = uploaded_file.name
        
        if not import_name.endswith('.txt'):
            import_name += '.txt'
            
        if st.button("Importer"):
            file_path = os.path.join(documents_folder, import_name)
            
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
                
            st.success(f"Document {import_name} importé avec succès!")
            
            # Réinitialiser la base de connaissances vectorielle
            st.session_state.vector_store = initialize_vector_db()
            
            st.experimental_rerun()

# Onglet 4: Explorer la Base de Connaissances
with tab4:
    st.header("Explorer la Base de Connaissances")
    
    st.write("""
    Cette section vous permet d'explorer et de visualiser la base de connaissances utilisée par le chatbot.
    Vous pouvez rechercher des sujets spécifiques et voir comment le système y répond.
    """)
    
    # Recherche par mots-clés
    st.subheader("Recherche par mots-clés")
    search_term = st.text_input("Entrez des mots-clés pour rechercher dans la base de connaissances:", key="kb_search")
    
    if st.button("Rechercher", key="search_kb"):
        if search_term:
            with st.spinner("Recherche en cours..."):
                # Utiliser la recherche par similarité pour trouver des documents pertinents
                results = st.session_state.vector_store.similarity_search(search_term, k=5)
                
                if results:
                    st.success(f"{len(results)} résultats trouvés")
                    
                    for i, doc in enumerate(results):
                        with st.expander(f"Résultat {i+1} - Source: {doc.metadata['source']}"):
                            st.write(doc.page_content)
                            
                            # Bouton pour générer une réponse basée sur ce document spécifique
                            if st.button("Générer une réponse avec ce contenu", key=f"gen_ans_{i}"):
                                with st.spinner("Génération de réponse..."):
                                    if ai_model == "mistral":
                                        llm = ChatMistralAI(model="mistral-small", temperature=0.3)
                                    else:
                                        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
                                    
                                    response = llm.invoke(
                                        f"En te basant UNIQUEMENT sur le texte suivant, réponds à cette question: '{search_term}'\n\nTEXTE: {doc.page_content}"
                                    )
                                    st.info("Réponse générée:")
                                    st.write(response.content)
                else:
                    st.warning("Aucun résultat trouvé pour cette recherche.")
      # Visualisation des documents
    st.subheader("Documents disponibles")
    documents_folder = "./documents"
    
    # Create a unique identifier for each file to avoid key collisions between tabs
    explorer_files = list(enumerate(os.listdir(documents_folder)))
    
    for idx, file_name in explorer_files:
        file_path = os.path.join(documents_folder, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.txt'):
            with st.expander(f"Contenu de {file_name}"):
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Use unique keys with an explorer prefix and index
                    st.text_area("Contenu:", value=content, height=200, disabled=True, key=f"explorer_view_{idx}_{file_name}")
                    
                    # Option pour tester le document
                    test_q = st.text_input("Posez une question sur ce document:", key=f"explorer_test_q_{idx}_{file_name}")
                    if st.button("Tester", key=f"explorer_test_btn_{idx}_{file_name}"):
                        with st.spinner("Génération de réponse..."):
                            if ai_model == "mistral":
                                llm = ChatMistralAI(model="mistral-small", temperature=0.3)
                            else:
                                llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
                        
                            response = llm.invoke(
                                f"En te basant UNIQUEMENT sur le texte suivant, réponds à cette question: '{test_q}'\n\nTEXTE: {content}"
                            )
                            st.info("Réponse générée:")
                            st.write(response.content)