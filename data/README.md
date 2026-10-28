# Данные

Датасеты не хранятся в репозитории. Оба набора взяты с Roboflow Universe в формате **YOLOv8**
(лицензия CC BY 4.0) и распаковываются сюда так, чтобы получилась структура:

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

| Набор | Источник | Тестовая выборка |
|---|---|---|
| CrowdHuman | [keio-dba-team/crowdhuman-nur7g](https://universe.roboflow.com/keio-dba-team/crowdhuman-nur7g), версия 3 (экспорт [CrowdHuman](https://www.crowdhuman.org/)) | 929 изображений, 25 816 объектов, класс `body` |
| Blood Cell | [tfg-2nmge/yolo-yejbs](https://universe.roboflow.com/tfg-2nmge/yolo-yejbs), версия 1 (на основе [BCCD](https://github.com/Shenggan/BCCD_Dataset)) | 87 изображений, 1 140 объектов, классы `Platelets`, `RBC`, `WBC` |

Для оценки алгоритмов NMS сами изображения не обязательны: сырые предсказания детекторов
на тестовых выборках уже сохранены в `predictions/`. Датасеты нужны только для того,
чтобы получить эти предсказания заново (удалите соответствующий `.pkl` и запустите `Evaluate.py`).
