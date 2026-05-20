from pathlib import Path

import pandas as pd


RUN_DIR = Path("outputs/runs/berturk_news_classifier_2026-05-08_20-15-53")
AGENT_CSV_PATH = RUN_DIR / "agent_decisions.csv"


def main():
    df = pd.read_csv(AGENT_CSV_PATH)

    print("Toplam örnek:", len(df))
    print("Genel doğruluk:", df["is_correct"].mean())

    print()
    print("CSV kolonları:")
    print(df.columns.tolist())

    print()
    print("Agent kararına göre doğruluk:")
    decision_summary = df.groupby("agent_decision").agg(
        sample_count=("is_correct", "count"),
        correct_count=("is_correct", "sum"),
        accuracy=("is_correct", "mean"),
        avg_confidence=("confidence", "mean"),
    )

    if "top2_margin" in df.columns:
        margin_summary = df.groupby("agent_decision").agg(
            avg_top2_margin=("top2_margin", "mean"),
            min_top2_margin=("top2_margin", "min"),
            max_top2_margin=("top2_margin", "max"),
        )

        decision_summary = decision_summary.join(margin_summary)

    print(decision_summary)

    print()
    print("Otomatik yönlendirmede yanlış olan örnekler:")

    auto_wrong = df[
        (df["agent_decision"] == "Otomatik yönlendirme") &
        (df["is_correct"] == False)
    ].copy()

    print("Yanlış otomatik yönlendirme sayısı:", len(auto_wrong))

    if len(auto_wrong) > 0:
        columns_to_show = [
            "true_category",
            "predicted_category",
            "confidence",
        ]

        if "second_predicted_category" in df.columns:
            columns_to_show.append("second_predicted_category")

        if "second_confidence" in df.columns:
            columns_to_show.append("second_confidence")

        if "top2_margin" in df.columns:
            columns_to_show.append("top2_margin")

        columns_to_show.append("short_content")

        print(auto_wrong[columns_to_show].head(15))

    if "top2_margin" in df.columns:
        print()
        print("Top-2 margin genel istatistikleri:")
        print(df["top2_margin"].describe())

        print()
        print("Doğru / yanlış tahmine göre top2_margin ortalaması:")
        print(df.groupby("is_correct")["top2_margin"].mean())

        print()
        print("Yanlış otomatik yönlendirmelerde top2_margin istatistikleri:")
        if len(auto_wrong) > 0:
            print(auto_wrong["top2_margin"].describe())
        else:
            print("Yanlış otomatik yönlendirme yok.")

        low_margin_auto = df[
            (df["agent_decision"] == "Otomatik yönlendirme") &
            (df["top2_margin"] < 0.20)
        ]

        print()
        print("Top2 margin < 0.20 olmasına rağmen otomatik yönlendirilen örnek sayısı:")
        print(len(low_margin_auto))

        if len(low_margin_auto) > 0:
            print("UYARI: Bu sayı 0 değilse decision_agent.py içinde margin kuralı doğru uygulanmamış olabilir.")
        else:
            print("Margin kuralı doğru çalışmış görünüyor.")

    else:
        print()
        print("top2_margin kolonu bulunamadı. decision_agent.py çıktı CSV'si güncel olmayabilir.")

    output_path = RUN_DIR / "agent_decision_analysis.csv"
    decision_summary.to_csv(output_path, encoding="utf-8-sig")

    wrong_output_path = RUN_DIR / "wrong_auto_routing_examples.csv"
    auto_wrong.to_csv(wrong_output_path, index=False, encoding="utf-8-sig")

    print()
    print(f"Agent karar analizi kaydedildi: {output_path}")
    print(f"Yanlış otomatik yönlendirme örnekleri kaydedildi: {wrong_output_path}")


if __name__ == "__main__":
    main()