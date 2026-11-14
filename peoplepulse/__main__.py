"""python -m peoplepulse: recompute every statistic and model result into results/ (the README quotes these files)."""
import json

from . import config, data, model, stats


def main() -> None:
    df = data.load()
    out = config.RESULTS_DIR
    out.mkdir(exist_ok=True)
    stats.univariate_tests(df).to_csv(out / "univariate_tests.csv", index=False)
    odds = stats.adjusted_odds_ratios(df)
    odds.to_csv(out / "adjusted_odds_ratios.csv", index=False)
    r = model.run(df)
    r["metrics"].to_csv(out / "model_metrics.csv", index=False)
    r["shap"].to_csv(out / "shap_importance.csv", index=False)
    r["audit"].to_csv(out / "group_audit.csv", index=False)
    summary = {"mode": data.mode(), "overall": stats.rate(df), "logit_pseudo_r2": odds.attrs["pseudo_r2"],
               "best_model_by_cv": r["best_model"], "holdout_size": r["holdout_size"],
               "holdout_leavers": r["holdout_leavers"], "pr_auc_random_baseline": r["pr_auc_baseline"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
