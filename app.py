from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

# Charger le modèle sauvegardé
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Créer l'application Flask
app = Flask(__name__)

######################################
# Route principale : page HTML
######################################
@app.route("/", methods=["GET"])
def index():
    # Retourne simplement le template 'index.html'
    return render_template("index.html")

######################################
# Route pour les prédictions
######################################
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer les données envoyées dans la requête
        data = request.json
        if "features" not in data:
            return jsonify({"error": "Les données doivent contenir une clé 'features' avec une liste de valeurs."}), 400
        
        # Convertir les données en tableau numpy
        features = np.array(data["features"]).reshape(1, -1)
        
        # Faire une prédiction
        prediction = model.predict(features)
        resultp = 0
        # On peut imaginer : 0 = sain, 1 = anomalie
        if prediction[0] <0.5:
            result_label = "Patient sain"

        else:
            result_label = "Patient avec anomalie"
            resultp= 1
        return jsonify({"prediction": int(resultp), "label": result_label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

######################################
# Route health check
######################################
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "API is running."})

if __name__ == "__main__":
    # Dans un vrai contexte, utilisez éventuellement 'host="0.0.0.0"'
    # et un numéro de port adapté à votre config
    app.run(debug=True, host="0.0.0.0", port=3000)
