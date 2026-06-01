# Phase 1 — Baseline boundary metrics (R50 FPN, PACO-LVIS val mini)

- predictions: `output/r50_mini/lvis_instances_results.json`
- matching: greedy per (image, part-category) at IoU >= 0.5
- boundary-F tolerance: 2.0px

## Overall
| group | mean IoU | boundary-F | matched/GT | recall@0.5 |
|---|---|---|---|---|
| all parts | 0.745 | 0.838 | 2777/5637 | 0.493 |

## Per object category
| group | mean IoU | boundary-F | matched/GT | recall@0.5 |
|---|---|---|---|---|
| basket | nan | nan | 0/0 | nan |
| belt | nan | nan | 0/0 | nan |
| bench | nan | nan | 0/0 | nan |
| bicycle | nan | nan | 0/0 | nan |
| blender | nan | nan | 0/0 | nan |
| book | nan | nan | 0/0 | nan |
| bottle | 0.755 | 0.814 | 663/1540 | 0.431 |
| bowl | nan | nan | 0/0 | nan |
| box | nan | nan | 0/0 | nan |
| broom | nan | nan | 0/0 | nan |
| bucket | nan | nan | 0/0 | nan |
| calculator | nan | nan | 0/0 | nan |
| can | nan | nan | 0/0 | nan |
| car_(automobile) | 0.714 | 0.866 | 1109/2192 | 0.506 |
| carton | nan | nan | 0/0 | nan |
| cellular_telephone | nan | nan | 0/0 | nan |
| chair | 0.747 | 0.817 | 422/1074 | 0.393 |
| clock | nan | nan | 0/0 | nan |
| crate | nan | nan | 0/0 | nan |
| cup | nan | nan | 0/0 | nan |
| dog | nan | nan | 0/0 | nan |
| drill | nan | nan | 0/0 | nan |
| drum_(musical_instrument) | nan | nan | 0/0 | nan |
| earphone | nan | nan | 0/0 | nan |
| fan | nan | nan | 0/0 | nan |
| glass_(drink_container) | nan | nan | 0/0 | nan |
| guitar | nan | nan | 0/0 | nan |
| hammer | nan | nan | 0/0 | nan |
| handbag | nan | nan | 0/0 | nan |
| hat | nan | nan | 0/0 | nan |
| helmet | nan | nan | 0/0 | nan |
| jar | nan | nan | 0/0 | nan |
| kettle | nan | nan | 0/0 | nan |
| knife | nan | nan | 0/0 | nan |
| ladder | nan | nan | 0/0 | nan |
| lamp | nan | nan | 0/0 | nan |
| laptop_computer | 0.797 | 0.799 | 384/507 | 0.757 |
| microwave_oven | nan | nan | 0/0 | nan |
| mirror | nan | nan | 0/0 | nan |
| mouse_(computer_equipment) | nan | nan | 0/0 | nan |
| mug | 0.781 | 0.883 | 199/324 | 0.614 |
| newspaper | nan | nan | 0/0 | nan |
| pan_(for_cooking) | nan | nan | 0/0 | nan |
| pen | nan | nan | 0/0 | nan |
| pencil | nan | nan | 0/0 | nan |
| pillow | nan | nan | 0/0 | nan |
| pipe | nan | nan | 0/0 | nan |
| plastic_bag | nan | nan | 0/0 | nan |
| plate | nan | nan | 0/0 | nan |
| pliers | nan | nan | 0/0 | nan |
| remote_control | nan | nan | 0/0 | nan |
| scarf | nan | nan | 0/0 | nan |
| scissors | nan | nan | 0/0 | nan |
| screwdriver | nan | nan | 0/0 | nan |
| shoe | nan | nan | 0/0 | nan |
| slipper_(footwear) | nan | nan | 0/0 | nan |
| soap | nan | nan | 0/0 | nan |
| sponge | nan | nan | 0/0 | nan |
| spoon | nan | nan | 0/0 | nan |
| stool | nan | nan | 0/0 | nan |
| sweater | nan | nan | 0/0 | nan |
| table | nan | nan | 0/0 | nan |
| tape_(sticky_cloth_or_paper) | nan | nan | 0/0 | nan |
| telephone | nan | nan | 0/0 | nan |
| television_set | nan | nan | 0/0 | nan |
| tissue_paper | nan | nan | 0/0 | nan |
| towel | nan | nan | 0/0 | nan |
| trash_can | nan | nan | 0/0 | nan |
| tray | nan | nan | 0/0 | nan |
| vase | nan | nan | 0/0 | nan |
| wallet | nan | nan | 0/0 | nan |
| watch | nan | nan | 0/0 | nan |
| wrench | nan | nan | 0/0 | nan |

## Thin / elongated parts
| group | mean IoU | boundary-F | matched/GT | recall@0.5 |
|---|---|---|---|---|
| thin parts | 0.692 | 0.856 | 367/975 | 0.376 |

