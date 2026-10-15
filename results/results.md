| Model | Params | Test UAR | Test WAR | Val UAR | Train time |
|---|---|---|---|---|---|
| VideoMAE-B, K400 supervised (full FT) | 86M | **58.3%** | 70.2% | 61.0% | 2.5 h |
| VideoMAE-B, self-supervised (full FT) | 86M | **56.3%** | 66.5% | 56.7% | 1.9 h |
| VideoMAE-B, self-supervised (2-stage FT) | 86M | **54.9%** | 66.4% | 59.7% | 1.2 h |
| V-JEPA 2 ViT-L, self-supervised (2-stage FT) | 375M | **51.4%** | 60.1% | 53.2% | 4.7 h |

| Model | Happy | Sad | Neutral | Angry | Surprise | Disgust | Fear |
|---|---|---|---|---|---|---|---|
| VideoMAE-B, K400 supervised (full FT) | 89% | 68% | 70% | 74% | 64% | 7% | 36% |
| VideoMAE-B, self-supervised (full FT) | 88% | 69% | 51% | 73% | 67% | 3% | 43% |
| VideoMAE-B, self-supervised (2-stage FT) | 89% | 63% | 60% | 72% | 63% | 3% | 35% |
| V-JEPA 2 ViT-L, self-supervised (2-stage FT) | 75% | 48% | 57% | 71% | 59% | 10% | 41% |
