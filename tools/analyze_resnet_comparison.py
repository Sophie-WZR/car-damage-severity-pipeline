#!/usr/bin/env python3
"""
分析 ResNet 对比结果并生成可视化报告。

Usage:
  python tools/analyze_resnet_comparison.py \
    --results-dir artifacts/resnet_comparison \
    --output-report artifacts/resnet_comparison/analysis_report.md
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def load_results(results_dir: Path) -> dict:
    """Load all variant results from JSON files."""
    variants = {}
    for json_file in sorted(results_dir.glob("resnet*_results.json")):
        with open(json_file) as f:
            data = json.load(f)
            variants[data["variant"]] = data
    return variants


def generate_report(variants: dict, output_file: Path):
    """Generate markdown report with analysis and recommendations."""
    
    report = """# ResNet 变体对比分析报告

## 概览

对比了以下 ResNet 变体在车损严重度分类任务上的性能：
- ResNet18（基础）
- ResNet34
- ResNet50
- ResNet101

## 性能对比表

| 模型 | 参数数 | 测试准确率 | 宏平均 F1 | AIC | BIC | 推理时间(ms) |
|------|--------|-----------|---------|-----|-----|-------------|
"""
    
    # Sort by accuracy descending
    sorted_vars = sorted(variants.items(), 
                         key=lambda x: x[1]["test_accuracy"], 
                         reverse=True)
    
    for name, res in sorted_vars:
        report += f"| {name} | {res['n_params']:,} | {res['test_accuracy']:.4f} | {res['test_macro_f1']:.4f} | {res['aic']:.1f} | {res['bic']:.1f} | {res['inference_ms_per_sample']:.4f} |\n"
    
    # Best model analysis
    best_acc = max(variants.values(), key=lambda x: x["test_accuracy"])
    best_f1 = max(variants.values(), key=lambda x: x["test_macro_f1"])
    best_aic = min(variants.values(), key=lambda x: x["aic"])
    
    report += f"""
## 推荐方案

### 精度最高
**{best_acc['variant']}** 达到最高准确率 {best_acc['test_accuracy']:.4f}

### F1 分数最高
**{best_f1['variant']}** 宏平均 F1 = {best_f1['test_macro_f1']:.4f}

### 模型复杂度最优（AIC）
**{best_aic['variant']}** AIC = {best_aic['aic']:.1f}（平衡精度与复杂度）

## 详细结果

"""
    
    for name, res in sorted_vars:
        report += f"""
### {name}

**基础信息**
- 参数数：{res['n_params']:,}
- 推理时间：{res['inference_ms_per_sample']:.4f} ms/sample

**测试集性能**
- 准确率：{res['test_accuracy']:.4f}
- 宏平均 F1：{res['test_macro_f1']:.4f}
- AIC：{res['aic']:.1f}
- BIC：{res['bic']:.1f}

**按类别性能**
"""
        report_dict = res["test_report"]
        for cls_name in ["minor", "moderate", "severe"]:
            if cls_name in report_dict:
                cls_data = report_dict[cls_name]
                report += f"""
| {cls_name:8s} | Precision: {cls_data['precision']:.3f} | Recall: {cls_data['recall']:.3f} | F1: {cls_data['f1-score']:.3f} |
"""
        
        report += f"\n**混淆矩阵**\n```\n{np.array(res['confusion_matrix'])}\n```\n"
    
    report += """
## 建议

1. **若优先精度**：选择准确率最高的模型（可能需要接受更高推理成本）
2. **若平衡效果与速度**：选择 AIC 最低的模型（模型复杂度调整后的最优）
3. **若资源受限**：保留 ResNet18（性能相近但更轻量）

## 后续优化方向

- 微调学习率和 batch size
- 应用数据增强（如 mixup、cutmix）
- 集成多个最佳模型
- 进行 5-fold 交叉验证以获得更稳健的性能估计
"""
    
    with open(output_file, "w") as f:
        f.write(report)
    
    print(f"Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ResNet comparison results")
    parser.add_argument("--results-dir", required=True, help="Directory containing JSON results")
    parser.add_argument("--output-report", required=True, help="Output markdown report path")
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output_file = Path(args.output_report)
    
    variants = load_results(results_dir)
    
    if not variants:
        print(f"No results found in {results_dir}")
        return
    
    print(f"Loaded {len(variants)} variant results")
    generate_report(variants, output_file)


if __name__ == "__main__":
    main()
