# Phase 1 — Baseline boundary metrics (R50 FPN, PACO-LVIS val mini)

- predictions: `/xuanwu-tank/north/jade/Paco/output/snake_fpn_full_20260601_200407/refined_val/lvis_instances_results.json`
- matching: greedy per (image, part-category) at IoU >= 0.5
- boundary-F tolerance: 2.0px

## Overall
| group | mean IoU | boundary-F | matched/GT | recall@0.5 |
|---|---|---|---|---|
| all parts | 0.740 | 0.803 | 9654/20945 | 0.461 |

## Per object category
| group | mean IoU | boundary-F | matched/GT | recall@0.5 |
|---|---|---|---|---|
| basket | 0.729 | 0.791 | 237/484 | 0.490 |
| belt | 0.655 | 0.841 | 115/260 | 0.442 |
| bench | 0.713 | 0.699 | 229/464 | 0.494 |
| bicycle | 0.675 | 0.800 | 421/897 | 0.469 |
| blender | 0.787 | 0.839 | 46/70 | 0.657 |
| book | 0.733 | 0.880 | 519/1138 | 0.456 |
| bottle | 0.756 | 0.815 | 664/1540 | 0.431 |
| bowl | 0.766 | 0.817 | 304/639 | 0.476 |
| box | 0.778 | 0.739 | 248/479 | 0.518 |
| broom | 0.965 | 1.000 | 1/2 | 0.500 |
| bucket | 0.804 | 0.891 | 38/94 | 0.404 |
| calculator | 0.769 | 0.753 | 1/4 | 0.250 |
| can | 0.757 | 0.781 | 60/152 | 0.395 |
| car_(automobile) | 0.714 | 0.859 | 1016/2192 | 0.464 |
| carton | 0.857 | 0.799 | 4/11 | 0.364 |
| cellular_telephone | 0.772 | 0.755 | 158/275 | 0.575 |
| chair | 0.753 | 0.811 | 381/1074 | 0.355 |
| clock | 0.664 | 0.875 | 76/154 | 0.494 |
| crate | 0.745 | 0.749 | 51/120 | 0.425 |
| cup | 0.765 | 0.904 | 116/230 | 0.504 |
| dog | 0.720 | 0.625 | 458/856 | 0.535 |
| drill | nan | nan | 0/2 | 0.000 |
| drum_(musical_instrument) | nan | nan | 0/4 | 0.000 |
| earphone | 0.713 | 0.907 | 11/45 | 0.244 |
| fan | 0.746 | 0.826 | 14/40 | 0.350 |
| glass_(drink_container) | 0.791 | 0.827 | 338/576 | 0.587 |
| guitar | 0.722 | 0.790 | 18/34 | 0.529 |
| hammer | 0.595 | 0.573 | 2/4 | 0.500 |
| handbag | 0.765 | 0.794 | 183/420 | 0.436 |
| hat | 0.746 | 0.898 | 84/306 | 0.275 |
| helmet | 0.703 | 0.901 | 54/145 | 0.372 |
| jar | 0.790 | 0.849 | 133/354 | 0.376 |
| kettle | 0.704 | 0.758 | 13/20 | 0.650 |
| knife | 0.787 | 0.845 | 184/263 | 0.700 |
| ladder | 0.594 | 0.611 | 7/43 | 0.163 |
| lamp | 0.807 | 0.904 | 239/388 | 0.616 |
| laptop_computer | 0.800 | 0.794 | 365/507 | 0.720 |
| microwave_oven | 0.763 | 0.862 | 86/121 | 0.711 |
| mirror | 0.765 | 0.814 | 35/96 | 0.365 |
| mouse_(computer_equipment) | 0.744 | 0.828 | 128/225 | 0.569 |
| mug | 0.799 | 0.879 | 175/324 | 0.540 |
| newspaper | 0.788 | 0.532 | 6/17 | 0.353 |
| pan_(for_cooking) | 0.738 | 0.811 | 26/89 | 0.292 |
| pen | 0.741 | 0.909 | 32/123 | 0.260 |
| pencil | 0.794 | 0.961 | 6/29 | 0.207 |
| pillow | 0.685 | 0.543 | 17/52 | 0.327 |
| pipe | 0.710 | 0.881 | 13/42 | 0.310 |
| plastic_bag | 0.766 | 0.717 | 101/177 | 0.571 |
| plate | 0.661 | 0.558 | 148/456 | 0.325 |
| pliers | nan | nan | 0/0 | nan |
| remote_control | 0.713 | 0.805 | 75/191 | 0.393 |
| scarf | 0.756 | 0.706 | 41/71 | 0.577 |
| scissors | 0.712 | 0.733 | 84/217 | 0.387 |
| screwdriver | 0.876 | 1.000 | 1/3 | 0.333 |
| shoe | 0.644 | 0.839 | 419/1464 | 0.286 |
| slipper_(footwear) | 0.735 | 0.474 | 2/9 | 0.222 |
| soap | 0.737 | 0.881 | 63/125 | 0.504 |
| sponge | 0.716 | 1.000 | 1/2 | 0.500 |
| spoon | 0.749 | 0.847 | 183/317 | 0.577 |
| stool | 0.822 | 0.925 | 21/57 | 0.368 |
| sweater | 0.665 | 0.518 | 137/398 | 0.344 |
| table | 0.708 | 0.585 | 95/234 | 0.406 |
| tape_(sticky_cloth_or_paper) | 0.657 | 0.815 | 1/3 | 0.333 |
| telephone | 0.706 | 0.815 | 53/104 | 0.510 |
| television_set | 0.751 | 0.812 | 153/233 | 0.657 |
| tissue_paper | nan | nan | 0/0 | nan |
| towel | 0.762 | 0.735 | 58/146 | 0.397 |
| trash_can | 0.824 | 0.877 | 185/256 | 0.723 |
| tray | 0.678 | 0.625 | 64/269 | 0.238 |
| vase | 0.778 | 0.808 | 356/597 | 0.596 |
| wallet | 0.703 | 0.633 | 4/4 | 1.000 |
| watch | 0.695 | 0.896 | 97/202 | 0.480 |
| wrench | nan | nan | 0/6 | 0.000 |

## Thin / elongated parts
| group | mean IoU | boundary-F | matched/GT | recall@0.5 |
|---|---|---|---|---|
| thin parts | 0.683 | 0.827 | 1456/4137 | 0.352 |
