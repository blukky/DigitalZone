# ЦИФРА.ZONE | Создание нейросети для контроля СИЗ и безопасности на производстве

## Функциональные возможности:
Используется последня версия всеми известной модели YOLO, которая позволяет практически в реальном времени 
детектировать объекты на видео потоке.

## Проделанная работа:
- Мы взяли уже готовые веса, разбили весь видео поток на кадры и разметили его.
Затем перееобучили модель под наши видео.

- Для определения СИЗ также использовали легковестную модель, 
которая давала не плохие результаты на наборе данных ImageNet,
Ее мы дообучили с помощью TransferLearing.

- Приложение представляет и себя Desktop 
приложение, что упрощает с ним взаимодействие и убирает
нужду в соединении с интернетом.

## Инструкция по развертыванияю
1. Создать окружение python3 и активировать его `source venv/bin/activate` – для Unix; `venv/Scripts/activate.bat` – для Windows
2. Установить зависимости из requirements.txt командой `pip install -r [path to]/requirements.txt`
3. Запустить программу одной из команд:
### Для видео
`python detect_people.py --video path_to_video`

или
### Для изображений
`python detect_people.py --img path_to_img`

## Демонстрация работы
![alt text](https://github.com/blukky/DigitalZone/blob/master/view.jpeg?raw=true)

## Тепловая дорожка
![alt text](https://github.com/blukky/DigitalZone/blob/master/ir2.jpg?raw=true)
![alt text](https://github.com/blukky/DigitalZone/blob/master/ir.jpg?raw=true)
