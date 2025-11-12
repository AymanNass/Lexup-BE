#!/bin/bash

# Script per configurare il deployment manuale su Cloud Run

# Controlla se gcloud CLI è installato
if ! command -v gcloud &> /dev/null
then
    echo "gcloud CLI non trovato. Installa Google Cloud SDK."
    exit 1
fi

# Richiedi il Project ID se non fornito
if [ -z "$1" ]
then
    read -p "Inserisci il tuo Project ID GCP: " PROJECT_ID
else
    PROJECT_ID=$1
fi

# Configura il progetto
gcloud config set project $PROJECT_ID

# Abilita le API necessarie
echo "Abilitazione delle API necessarie..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable discoveryengine.googleapis.com
gcloud services enable aiplatform.googleapis.com

# Crea un trigger Cloud Build
echo "Vuoi configurare un trigger Cloud Build per GitHub/GitLab? (y/n)"
read CREATE_TRIGGER

if [ "$CREATE_TRIGGER" = "y" ] || [ "$CREATE_TRIGGER" = "Y" ]
then
    read -p "Inserisci l'URL del repository (es. https://github.com/username/repo): " REPO_URL
    read -p "Inserisci il nome del branch da monitorare (es. main): " BRANCH
    
    # Estrai nome repository e provider
    if [[ $REPO_URL == *"github.com"* ]]; then
        PROVIDER="github"
        REPO_NAME=$(echo $REPO_URL | sed 's/.*github.com\/\([^\/]*\/[^\/]*\).*/\1/')
    elif [[ $REPO_URL == *"gitlab.com"* ]]; then
        PROVIDER="gitlab"
        REPO_NAME=$(echo $REPO_URL | sed 's/.*gitlab.com\/\([^\/]*\/[^\/]*\).*/\1/')
    else
        echo "Provider non supportato. Supportiamo solo GitHub e GitLab."
        exit 1
    fi
    
    echo "Collega il tuo account $PROVIDER a Cloud Build nella console GCP"
    echo "Vai a: https://console.cloud.google.com/cloud-build/triggers"
    echo "Clicca su 'Collega repository' e segui le istruzioni"
    echo "Premi INVIO quando hai completato il collegamento"
    read
    
    # Crea il trigger
    gcloud builds triggers create $PROVIDER \
        --repo=$REPO_NAME \
        --branch-pattern=$BRANCH \
        --build-config=cloudbuild.yaml
    
    echo "Trigger Cloud Build creato con successo!"
fi

echo "Configurazione completata!"
echo "Per eseguire un deployment manuale, usa:"
echo "gcloud builds submit --config=cloudbuild.yaml"
echo ""
echo "Ricorda di configurare le seguenti variabili nel cloudbuild.yaml o come variabili d'ambiente in Cloud Run:"
echo "- PROJECT_ID: $PROJECT_ID"
echo "- DISCOVERY_ENGINE_ID: [il tuo engine ID]"
echo "- DATA_STORE_ID: [il tuo data store ID]"
echo ""
echo "Buon deployment!"