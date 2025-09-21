#!/bin/bash

# Script de test pour simuler l'expérience d'un nouveau développeur
echo "=== Test d'un nouveau développeur clonant le repository ==="
echo

# Simuler un environnement propre (sans venv existant)
echo "1. Clonage du repository..."
echo "   git clone <repo-url>"
echo "   cd final_project"
echo

echo "2. Création d'un environnement virtuel..."
python3 -m venv test_venv
source test_venv/bin/activate
echo "   ✓ Environnement virtuel créé et activé"
echo

echo "3. Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt
echo "   ✓ Dépendances installées"
echo

echo "4. Exécution des tests..."
pytest tests/ -v
echo "   ✓ Tests passés"
echo

echo "5. Test d'entraînement du modèle..."
python -m src.train_model
echo "   ✓ Modèle entraîné avec succès"
echo

echo "6. Vérification des fichiers de modèle..."
python -c "import pickle; pickle.load(open('app/model.pkl', 'rb')); pickle.load(open('app/scaler.pkl', 'rb')); print('✓ Modèles chargés avec succès')"
echo

echo "7. Test d'import de l'application Streamlit..."
python -c "import streamlit; import app.app; print('✓ Application Streamlit importée avec succès')"
echo

echo "8. Test du build Docker..."
docker build -t diabetes-app-test .
echo "   ✓ Build Docker réussi"
echo

echo "9. Nettoyage..."
deactivate
rm -rf test_venv
docker rmi diabetes-app-test
echo "   ✓ Nettoyage terminé"
echo

echo "=== Tous les tests sont passés ! Le projet est prêt pour un nouveau développeur ==="
