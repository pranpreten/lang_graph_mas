# data_summary 함수만 따로 테스트한 뒤, 통과하면 data_prep.py에 붙인다.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as c
import data_prep as dp


def data_summary(X_train, task):
    """인지(perception) 단계가 SLM에 넘길 '데이터 요약'을 만든다.

    SLM은 raw 수만 행을 다 못 보므로, 스키마+통계 요약만 준다.
    포함: 표 크기, 피처 목록, 타깃 설명, 피처별 기초통계, 저분산(거의 상수) 피처 표시.
    """
    spec = c.TASK_SPECS[task]
    desc = X_train.describe().T            # 피처별 mean/std/min/max 등
    low_var = desc.index[desc["std"] < 1e-6].tolist()   # 거의 상수인 센서

    summary = {
        "task": task,
        "kind": spec["kind"],                       # supervised / unsupervised
        "target_desc": spec.get("target", "없음(비지도)"),
        "metric": spec["metric"],
        "n_samples": int(X_train.shape[0]),
        "n_features": int(X_train.shape[1]),
        "features": list(X_train.columns),
        "low_variance_features": low_var,           # SLM이 "이건 빼라" 판단 힌트
        "feature_stats": {                          # 앞 몇 개만 예시로(프롬프트 길이 절약)
            col: {"mean": round(desc.loc[col, "mean"], 3),
                  "std":  round(desc.loc[col, "std"], 3),
                  "min":  round(desc.loc[col, "min"], 3),
                  "max":  round(desc.loc[col, "max"], 3)}
            for col in X_train.columns
        },
    }
    return summary


if __name__ == "__main__":
    import json
    Xtr, ytr, Xte, yte = dp.make_task("regression")
    s = data_summary(Xtr, "regression")
    print("=== data_summary (회귀) ===")
    print("task:", s["task"], "| kind:", s["kind"], "| target:", s["target_desc"], "| metric:", s["metric"])
    print("샘플 수:", s["n_samples"], "| 피처 수:", s["n_features"])
    print("거의 상수인(쓸모없는) 센서:", s["low_variance_features"])
    print()
    print("피처 통계 예시 (s2, s11):")
    print("  s2 :", s["feature_stats"]["s2"])
    print("  s11:", s["feature_stats"]["s11"])
