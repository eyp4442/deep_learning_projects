import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# -------------------------
# CONFIG
# -------------------------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
RUNS_DIR = OUTPUT_DIR / "runs"

TEST_PATH = DATA_DIR / "test_prepared.csv"
LABEL_MAP_PATH = DATA_DIR / "label_map.json"

MAX_LENGTH = 256

DEFAULT_HIGH_CONFIDENCE_THRESHOLD = 0.90
RISKY_HIGH_CONFIDENCE_THRESHOLD = 0.97
MEDIUM_CONFIDENCE_THRESHOLD = 0.65

MIN_AUTO_MARGIN = 0.20

RISKY_CATEGORIES = {"dünya", "siyaset", "gündem"}


# -------------------------
# UTILS
# -------------------------

def find_latest_berturk_run():
    runs = sorted(
        RUNS_DIR.glob("berturk_news_classifier_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not runs:
        raise FileNotFoundError("BERTurk run klasörü bulunamadı.")

    return runs[0]


def load_label_map():
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    id_to_label_raw = label_map["id_to_label"]

    id_to_label = {
        int(idx): label_name
        for idx, label_name in id_to_label_raw.items()
    }

    class_names = [
        id_to_label[idx]
        for idx in sorted(id_to_label.keys())
    ]

    return class_names


def get_category_route(category_name):
    routes = {
        "dünya": "Dünya haberleri editör masası",
        "ekonomi": "Ekonomi haberleri editör masası",
        "eğitim": "Eğitim haberleri editör masası",
        "gündem": "Gündem editör masası",
        "kültürsanat": "Kültür-sanat editör masası",
        "magazin": "Magazin editör masası",
        "sağlık": "Sağlık haberleri editör masası",
        "siyaset": "Siyaset haberleri editör masası",
        "spor": "Spor haberleri editör masası",
        "teknoloji": "Teknoloji haberleri editör masası",
    }

    return routes.get(category_name, "Genel editör masası")


def make_agent_decision(predicted_category, confidence, top2_margin):
    route = get_category_route(predicted_category)

    if predicted_category in RISKY_CATEGORIES:
        high_threshold = RISKY_HIGH_CONFIDENCE_THRESHOLD
    else:
        high_threshold = DEFAULT_HIGH_CONFIDENCE_THRESHOLD

    if confidence >= high_threshold and top2_margin >= MIN_AUTO_MARGIN:
        decision = "Otomatik yönlendirme"
        explanation = (
            f"Model tahmine yüksek güven duyuyor ve ikinci tahminle arasındaki fark yeterli. "
            f"Haber doğrudan {route} birimine yönlendirildi."
        )
        needs_human_review = False

    elif confidence >= MEDIUM_CONFIDENCE_THRESHOLD:
        decision = "Editör kontrollü yönlendirme"

        if top2_margin < MIN_AUTO_MARGIN:
            reason = (
                f"Modelin ilk iki tahmini birbirine yakın. "
                f"Top-1/Top-2 güven farkı: {top2_margin:.4f}."
            )
        else:
            reason = (
                f"Model güveni otomatik yönlendirme için yeterli değil veya kategori riskli grupta."
            )

        explanation = (
            f"{reason} Haber için önerilen kategori: {predicted_category}. "
            f"{route} birimine yönlendirme önerilir ancak editör kontrolü gereklidir."
        )
        needs_human_review = True

    else:
        decision = "Manuel sınıflandırma gerekli"
        explanation = (
            f"Modelin güven skoru düşük. "
            f"Haber otomatik yönlendirilmemeli, editör tarafından manuel incelenmelidir."
        )
        needs_human_review = True

    return decision, explanation, route, needs_human_review


def predict_single_text(model, tokenizer, text, device):
    encoded = tokenizer(
        str(text),
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top2_values, top2_indices = torch.topk(probs, k=2)

    pred_id = int(top2_indices[0].item())
    confidence = float(top2_values[0].item())

    second_pred_id = int(top2_indices[1].item())
    second_confidence = float(top2_values[1].item())

    top2_margin = confidence - second_confidence

    return pred_id, confidence, second_pred_id, second_confidence, top2_margin, probs.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="BERTurk run klasörü. Boş bırakılırsa en son BERTurk run kullanılır.",
    )

    parser.add_argument(
        "--sample_size",
        type=int,
        default=50,
        help="Agent analizi için test setinden kaç örnek alınacak.",
    )

    args = parser.parse_args()

    if args.run_dir is None:
        run_dir = find_latest_berturk_run()
    else:
        run_dir = Path(args.run_dir)

    best_model_dir = run_dir / "best_model"

    if not best_model_dir.exists():
        raise FileNotFoundError(f"Best model klasörü bulunamadı: {best_model_dir}")

    agent_output_path = run_dir / "agent_decisions.csv"
    agent_summary_path = run_dir / "agent_summary.txt"

    print("Run directory:", run_dir)
    print("Best model:", best_model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    class_names = load_label_map()

    tokenizer = AutoTokenizer.from_pretrained(best_model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
    model.to(device)
    model.eval()

    test_df = pd.read_csv(TEST_PATH)

    sample_size = min(args.sample_size, len(test_df))

    # Farklı sınıflardan örnek gelsin diye category_name üzerinden karışık örnek seçiyoruz.
    sample_df = (
        test_df
        .sample(n=sample_size, random_state=42)
        .reset_index(drop=True)
    )

    results = []

    for _, row in sample_df.iterrows():
        text = str(row["content"])
        true_label_name = str(row["category_name"])

        pred_id, confidence, second_pred_id, second_confidence, top2_margin, probs = predict_single_text(
            model=model,
            tokenizer=tokenizer,
            text=text,
            device=device,
)

        predicted_category = class_names[pred_id]
        second_predicted_category = class_names[second_pred_id]

        decision, explanation, route, needs_human_review = make_agent_decision(
            predicted_category=predicted_category,
            confidence=confidence,
            top2_margin=top2_margin,
        )

        is_correct = predicted_category == true_label_name

        results.append({
            "short_content": text[:400],
            "true_category": true_label_name,
            "predicted_category": predicted_category,
            "confidence": confidence,
            "second_predicted_category": second_predicted_category,
            "second_confidence": second_confidence,
            "top2_margin": top2_margin,
            "is_correct": is_correct,
            "agent_decision": decision,
            "assigned_route": route,
            "needs_human_review": needs_human_review,
            "agent_explanation": explanation,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(agent_output_path, index=False, encoding="utf-8-sig")

    total = len(results_df)
    correct = int(results_df["is_correct"].sum())
    human_review = int(results_df["needs_human_review"].sum())
    auto_routed = total - human_review

    avg_confidence = results_df["confidence"].mean()

    summary = f"""
BERTurk Haber Sınıflandırma Karar Ajanı Özeti
==================================================

Kullanılan run:
{run_dir}

Analiz edilen örnek sayısı: {total}

Doğru tahmin sayısı: {correct}
Doğruluk oranı: {correct / total:.4f}

Ortalama güven skoru: {avg_confidence:.4f}

Otomatik yönlendirilen haber sayısı: {auto_routed}
Editör / manuel kontrol isteyen haber sayısı: {human_review}

Güven eşikleri:
- Riskli kategoriler için >= {RISKY_HIGH_CONFIDENCE_THRESHOLD}: Otomatik yönlendirme adayı
- Diğer kategoriler için >= {DEFAULT_HIGH_CONFIDENCE_THRESHOLD}: Otomatik yönlendirme adayı
- Otomatik yönlendirme için ayrıca Top-1 / Top-2 güven farkı >= {MIN_AUTO_MARGIN} olmalıdır
- >= {MEDIUM_CONFIDENCE_THRESHOLD} ve otomatik koşulları sağlamayanlar: Editör kontrollü yönlendirme
- < {MEDIUM_CONFIDENCE_THRESHOLD}: Manuel sınıflandırma gerekli

Kategori bazında agent karar dağılımı:
{results_df.groupby(["predicted_category", "agent_decision"]).size()}
"""

    with open(agent_summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print(summary)

    print(f"Agent kararları kaydedildi: {agent_output_path}")
    print(f"Agent özeti kaydedildi: {agent_summary_path}")

    print()
    print("İlk 5 agent kararı:")
    print(results_df[[
        "true_category",
        "predicted_category",
        "confidence",
        "is_correct",
        "agent_decision",
        "assigned_route",
    ]].head())


if __name__ == "__main__":
    main()