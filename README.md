# Chatbot de FAQ d'entreprise avec RAG (Retrieval Augmented Generation)

## Objectif
Construire un assistant intelligent capable de répondre aux questions des employés sur la base de documents internes, en utilisant les techniques de RAG (Retrieval Augmented Generation).

## Technologies utilisées
- **ChromaDB** - Base de données vectorielle pour stocker les embeddings
- **Whisper** - API d'OpenAI pour la transcription audio de réunions internes
- **OpenAI/Mistral** - Modèles de langage pour la génération de réponses
- **LangChain** - Framework pour construire des applications basées sur des LLMs
- **Streamlit** - Framework pour l'interface utilisateur web

## Challenge relevé
- Prompting optimisé pour des réponses précises
- Amélioration de la pertinence des réponses grâce à RAG
- Interface utilisateur intuitive et multifonctionnelle
- Système de feedback pour améliorer les réponses
- Support pour multiple modèles d'IA (OpenAI et Mistral)

## Fonctionnalités

### 1. Transcription Audio (Whisper)
- Enregistrement direct via le microphone
- Téléchargement de fichiers audio
- Transcription automatique et traitement par le chatbot

### 2. Chatbot FAQ
- Interface conversationnelle pour poser des questions
- Historique des conversations
- Sources des réponses
- Système d'évaluation des réponses (feedback)

### 3. Gestion des Documents
- Visualisation des documents existants
- Ajout de nouveaux documents
- Modification et suppression de documents
- Import de documents depuis des fichiers texte

### 4. Exploration de la Base de Connaissances
- Recherche par mots-clés
- Visualisation des documents
- Tests directs sur des documents spécifiques

## Configuration requise

1. **Installez les dépendances :**
```bash
pip install chromadb langchain langchain-openai langchain-community langchain-mistralai openai-whisper streamlit streamlit-audiorecorder
```

2. **Configurez vos clés API :**
   - Obtenez une clé API OpenAI sur [https://platform.openai.com](https://platform.openai.com)
   - Si vous souhaitez utiliser Mistral, obtenez une clé API sur [https://mistral.ai](https://mistral.ai)
   - Remplacez les clés API dans le code (`OPENAI_API_KEY` et `MISTRAL_API_KEY`)

3. **Créez un dossier `documents` pour stocker vos fichiers de FAQ**

## Exécution de l'application

Lancez l'application avec la commande :

```bash
streamlit run Audio.py
```

## Structure du projet

```
├── Audio.py                # Application principale
├── README.md               # Ce fichier
├── documents/              # Dossier contenant les documents FAQ
│   ├── faq.txt            # Exemple de FAQ
│   └── faq_produits.txt   # FAQ sur les produits
├── chroma_db/              # Base de données vectorielle (générée automatiquement)
└── assets/                 # Fichiers média supplémentaires
    └── test_recording.mp3  # Exemple d'enregistrement audio
```

## TPs

Ces liens pointent vers des ressources Colab qui ont servi au développement de ce projet:

1.(https://colab.research.google.com/drive/13DjLiP0d3Z9s2mhy1zFcY4xfbbvmL7OV?usp=sharing)
2.(https://colab.research.google.com/drive/1VzlN1MPmu-MK1y0lyN07G_hepC_zk_fC?usp=sharing)
3.(https://colab.research.google.com/drive/1HJ2R7cG8AvL8zqmMm76eMFdFEzXiZqP4?usp=sharing)
4.(https://colab.research.google.com/drive/1IC_90iHcmsTqBAo4JqKTn89Hv-LFEpvT?usp=sharing)
5.(https://colab.research.google.com/drive/14Gh1wc6OUYafH-G072XffLCBWzzQkRHM?usp=sharing)

## Améliorations futures

- Support pour d'autres formats de documents (PDF, Word, etc.)
- Visualisation avancée des similitudes entre documents
- Système d'apprentissage continu basé sur les feedbacks utilisateurs
- Support multilingue
- Intégration avec des systèmes CRM et helpdesk


TPs:


1 : https://colab.research.google.com/drive/13DjLiP0d3Z9s2mhy1zFcY4xfbbvmL7OV?usp=sharing

2: https://colab.research.google.com/drive/1VzlN1MPmu-MK1y0lyN07G_hepC_zk_fC?usp=sharing

3: https://colab.research.google.com/drive/1HJ2R7cG8AvL8zqmMm76eMFdFEzXiZqP4?usp=sharing

4: https://colab.research.google.com/drive/1IC_90iHcmsTqBAo4JqKTn89Hv-LFEpvT?usp=sharing

5: https://colab.research.google.com/drive/14Gh1wc6OUYafH-G072XffLCBWzzQkRHM?usp=sharing


j'ai des soucis avec le déploiement je réessaie demain (26/05)
