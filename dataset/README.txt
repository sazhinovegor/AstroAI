Поместите изображения в структуру:

dataset/train/galaxies/
dataset/train/nebulae/
dataset/val/galaxies/
dataset/val/nebulae/
dataset/test/galaxies/
dataset/test/nebulae/

Для конференционного исследования разделяйте снимки по астрономическим объектам:
все изображения, кадрирования и варианты обработки одного объекта должны быть
только в одном разделе. Скопируйте metadata_template.csv в metadata.csv и
заполните реальными относительными путями, object_id, источниками и лицензиями.
Перед запуском обучения выполните python audit_dataset.py --data_dir dataset.
