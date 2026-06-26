import json
from pathlib import Path

import pandas as pd
import timm
import torch
import torch.nn as nn


OUT_DIR = Path("reports/exp98_model_family_precheck")
OUT_DIR.mkdir(parents=True, exist_ok=True)


CANDIDATES = [
    {
        "rank": 1,
        "family": "MaxViT",
        "model_name": "maxvit_tiny_rw_224.sw_in1k",
        "img_size": 224,
        "batch_size": 16,
        "fine_tune_plan": "classifier/head + last stage",
        "reason": "CNN+attention hybrid. ConvNeXt/EfficientNet과 다른 inductive bias를 줄 가능성이 큼.",
        "risk": "MPS 메모리와 학습 속도. 단독 성능보다 OOF stacking diversity 목적.",
    },
    {
        "rank": 2,
        "family": "CoAtNet",
        "model_name": "coatnet_nano_rw_224.sw_in1k",
        "img_size": 224,
        "batch_size": 16,
        "fine_tune_plan": "classifier/head + last blocks",
        "reason": "convolution+attention 계열. MaxViT보다 가볍게 시작 가능.",
        "risk": "timm 모델별 head/layer 명칭 차이로 학습 범위 구현 확인 필요.",
    },
    {
        "rank": 3,
        "family": "SwinV2",
        "model_name": "swinv2_tiny_window8_256.ms_in1k",
        "img_size": 256,
        "batch_size": 12,
        "fine_tune_plan": "head + layers[-1]",
        "reason": "window attention 계열. 기존 CNN 계열과 오류 방향이 다를 수 있음.",
        "risk": "이전 Swin 계열이 강하지 않았으므로 stage 일부만 신중히.",
    },
    {
        "rank": 4,
        "family": "DeiT",
        "model_name": "deit3_small_patch16_224.fb_in1k",
        "img_size": 224,
        "batch_size": 16,
        "fine_tune_plan": "head + last blocks",
        "reason": "순수 ViT 계열. DINOv2 feature와 달리 supervised fine-tuning 후보.",
        "risk": "데이터 5k 규모에서는 CNN보다 불안정할 수 있음.",
    },
    {
        "rank": 5,
        "family": "CoAtNet larger",
        "model_name": "coatnet_0_rw_224.sw_in1k",
        "img_size": 224,
        "batch_size": 8,
        "fine_tune_plan": "classifier/head + last blocks",
        "reason": "nano보다 표현력이 크지만 10시간 이상 후보가 될 수 있음.",
        "risk": "장기 계산 후보. smoke 후에만 5-fold 검토.",
    },
]


def set_classifier(model, num_classes=2):
    try:
        model.reset_classifier(num_classes)
        return "reset_classifier"
    except Exception:
        pass

    for attr in ["head", "classifier", "fc"]:
        if hasattr(model, attr):
            layer = getattr(model, attr)
            if isinstance(layer, nn.Linear):
                setattr(model, attr, nn.Linear(layer.in_features, num_classes))
                return attr
            if isinstance(layer, nn.Sequential):
                for i in reversed(range(len(layer))):
                    if isinstance(layer[i], nn.Linear):
                        layer[i] = nn.Linear(layer[i].in_features, num_classes)
                        return f"{attr}[{i}]"
    raise ValueError(f"Cannot replace classifier for {model.__class__.__name__}")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def describe_module_names(model):
    names = []
    for name, module in model.named_children():
        child_params = sum(p.numel() for p in module.parameters())
        names.append({"child": name, "params": child_params, "class": module.__class__.__name__})
    return names


def freeze_and_mark_guess(model):
    for p in model.parameters():
        p.requires_grad = False

    marked = []
    for attr in ["head", "classifier", "fc"]:
        if hasattr(model, attr):
            module = getattr(model, attr)
            for p in module.parameters():
                p.requires_grad = True
            marked.append(attr)

    # Common timm container names. We only mark the last top-level stage/block in precheck.
    for attr in ["stages", "layers", "blocks"]:
        if hasattr(model, attr):
            module = getattr(model, attr)
            try:
                last = module[-1]
            except Exception:
                continue
            for p in last.parameters():
                p.requires_grad = True
            marked.append(f"{attr}[-1]")
            break
    return marked


def main():
    rows = []
    details = {}
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    for spec in CANDIDATES:
        row = dict(spec)
        row["timm_available"] = spec["model_name"] in timm.list_models(pretrained=True)
        try:
            model = timm.create_model(spec["model_name"], pretrained=False, num_classes=2)
            classifier_method = set_classifier(model, 2)
            total_params, _ = count_params(model)
            marked = freeze_and_mark_guess(model)
            _, trainable_guess = count_params(model)
            row.update(
                {
                    "status": "ok",
                    "classifier_method": classifier_method,
                    "total_params": total_params,
                    "trainable_guess_params": trainable_guess,
                    "trainable_guess_ratio": trainable_guess / total_params if total_params else 0.0,
                    "device": device,
                    "recommended_first_run": "fold0 smoke 2 epochs",
                    "estimated_5fold_cost": estimate_cost(row["family"], total_params),
                }
            )
            details[spec["model_name"]] = {
                "marked_trainable": marked,
                "children": describe_module_names(model),
            }
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "error": repr(exc),
                    "total_params": None,
                    "trainable_guess_params": None,
                    "trainable_guess_ratio": None,
                    "recommended_first_run": "blocked",
                    "estimated_5fold_cost": "unknown",
                }
            )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "exp98_model_family_precheck.csv", index=False)
    (OUT_DIR / "exp98_model_family_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    write_report(df, details)
    print(df.to_string(index=False))


def estimate_cost(family, total_params):
    if total_params is None:
        return "unknown"
    if total_params < 15_000_000:
        return "2~4h for 5-fold if stable"
    if total_params < 35_000_000:
        return "4~8h for 5-fold"
    if family.lower().startswith("coatnet larger"):
        return "10h+ candidate after smoke"
    return "8~12h depending MPS memory"


def write_report(df, details):
    lines = [
        "# EXP-98 Model Family Precheck",
        "",
        "목적: 새 제출 후보를 바로 만들기 전에 ConvNeXt/EfficientNet과 다른 오류 방향을 줄 수 있는 모델 family를 사전 검토한다.",
        "",
        "## 결론",
        "",
        "1순위는 `maxvit_tiny_rw_224.sw_in1k` fold0 smoke다. hybrid attention 계열이라 기존 CNN OOF pool과 다른 정보를 줄 가능성이 있고, tiny급이라 계산 리스크가 비교적 작다.",
        "",
        "2순위는 `coatnet_nano_rw_224.sw_in1k`다. MaxViT smoke가 메모리/속도 문제를 보이면 더 가벼운 hybrid 후보로 진행한다.",
        "",
        "SwinV2/DeiT는 diversity 후보지만, 기존 Swin/DINOv2 결과를 감안하면 바로 5-fold로 가지 않고 smoke 결과를 본 뒤 결정한다.",
        "",
        "## Candidate Summary",
        "",
        "| rank | family | model | params | trainable guess | first run | cost |",
        "|---:|---|---|---:|---:|---|---|",
    ]
    for row in df.sort_values("rank").itertuples(index=False):
        params = "" if pd.isna(row.total_params) else f"{int(row.total_params):,}"
        trainable = "" if pd.isna(row.trainable_guess_params) else f"{int(row.trainable_guess_params):,}"
        lines.append(
            f"| {row.rank} | {row.family} | `{row.model_name}` | {params} | {trainable} | {row.recommended_first_run} | {row.estimated_5fold_cost} |"
        )
    lines.extend(
        [
            "",
            "## 실행 게이트",
            "",
            "- 바로 5-fold 학습하지 않는다.",
            "- 먼저 fold0 smoke 2 epoch로 OOF 성능, confusion matrix, guardrail 샘플 방향을 확인한다.",
            "- smoke가 EXP-65처럼 보호 샘플을 대량으로 깨면 단독 OOF 확장은 중단한다.",
            "- smoke가 약하더라도 오류 방향이 기존 모델과 다르면 stacking feature 후보로 보류한다.",
            "",
            "## Outputs",
            "",
            "- `reports/exp98_model_family_precheck/exp98_model_family_precheck.csv`",
            "- `reports/exp98_model_family_precheck/exp98_model_family_details.json`",
            "- `reports/exp98_model_family_precheck/report.md`",
        ]
    )
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
