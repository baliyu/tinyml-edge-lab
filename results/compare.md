| Model | Features | Choice | Nodes | CV macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---|---|---|---|---|
| Tree, depth 6 fixed in advance | baseline | fixed depth 6 | 61 | 0.816 (SE 0.034) | 0.797 | 0.790 |
| Tree, depth chosen by CV | baseline | depth 4 (CV) | 17 | 0.819 (SE 0.032) | 0.810 | 0.807 |
| Tree, depth chosen by CV | extended | depth 5 (CV) | 33 | 0.833 (SE 0.014) | 0.839 | 0.834 |
| Random forest, size chosen by CV | extended | 30 trees, depth 6 (CV) | 2186 | 0.883 (SE 0.026) | 0.879 | 0.877 |
