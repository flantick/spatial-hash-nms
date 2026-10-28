# Data

**English** | [Русский](README.ru.md)

The datasets are not stored in the repository. Both are taken from Roboflow Universe in **YOLOv8**
format (CC BY 4.0 license) and should be extracted here with the following layout:

```
data/
├── CrowdHuman/
│   ├── data.yaml
│   ├── train/  valid/
│   └── test/
│       ├── images/
│       └── labels/
└── BloodCell/
    ├── data.yaml
    ├── train/  valid/
    └── test/
        ├── images/
        └── labels/
```

| Dataset | Source | Test split |
|---|---|---|
| CrowdHuman | [keio-dba-team/crowdhuman-nur7g](https://universe.roboflow.com/keio-dba-team/crowdhuman-nur7g), version 3 (export of [CrowdHuman](https://www.crowdhuman.org/)) | 929 images, 25,816 objects, class `body` |
| Blood Cell | [tfg-2nmge/yolo-yejbs](https://universe.roboflow.com/tfg-2nmge/yolo-yejbs), version 1 (based on [BCCD](https://github.com/Shenggan/BCCD_Dataset)) | 87 images, 1,140 objects, classes `Platelets`, `RBC`, `WBC` |

The images themselves are not required to evaluate the NMS algorithms: raw detector predictions
on the test splits are already stored in `predictions/`. The datasets are needed only to regenerate
these predictions (delete the corresponding `.pkl` file and run `Evaluate.py`).
