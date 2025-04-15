from src.predict.classify import classify_with_all_models


if __name__ == "__main__":
    image_path = "resources/custom/raw-0-digit.0.0.png"  # image prétaitée

    results = classify_with_all_models(image_path)

    for model, (pred, proba) in results.items():
        print(f"\n🔹 Modèle : {model}")
        print(f"  ➤ Prédiction : {pred}")
        print(f"  ➤ Probabilités : {proba}")
