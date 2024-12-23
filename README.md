# 🌏🧳🛩️  AI Travel Guide
> Ассистент путешественника, который предоставляет информацию о культурных особенностях, обычаях и достопримечательностях по запросу пользователя о месте.

## Запуск

Для запуска приложения локально выполните следующие шаги:

1. Установите необходимые зависимости:
   ```
   pip install -r requirements.txt
   ```

2. Если запускаете приложение впервые, то скачайте данные для RAG из [папки](https://drive.google.com/drive/folders/1vZmVLdmalDOYs8N7aTzUBaJ6sprVpo5f?usp=sharing) и положите в папку data в корне репозитория. При запуске данные будут проиндексированы и записаны в vectorstore.

3. Приложение использует LLM GigaChat по API, поэтому для коррректной работы необходимо добавить ключ API_KEY в файл .env

4. Запустите Gradio приложение следующей командой:
   ```
   python app.py
   ```

5. После запуска приложение будет доступно на localhost или по публичной ссылке

## Структура репозитория
```
├── src
│   ├── data -- парсинг и предобработка данных
|       ├── data_parsing.py -- парсинг данных с wiki-ресурсов
|       └── data_processing.py -- предобработка данных перед индексацией
|   ├── rag.py -- собственно RAG
|   ├── retriever.py -- Retriever для индексации и создания vectorstore
|   ├── interface.py -- интерфейс на Gradio
|
├── app.py -- основное приложение
└── requirements.txt -- зависимости
```

## Валидация результатов

### Генерация вопросов для валидации

Модель - gpt-4o-mini

Вход - чанки текстов, которые получается при препроцессинге. Количество - 100.

Промпт - "Напиши по одному вопросу для каждого текста. Тексты отделены запятой."

### Генерация ответов

- c помощью GigaChat-Lite:
    - с использование контекста (rag_answers)
    - без использования контекста, только с промптом (giga_chat_answers)
Часть вопросов были удалены из-за "Giga generation stopped with reason: blacklist"
Результат - 64 вопроса с ответами

### Оценка

Модель - gpt-4o-mini
Промпт - "Ответь 1, если лучше answer_a, иначе ответь 2. Формат ответа json."
Ответы были случайным образом перемещаны, чтобы минимизировать влияние позиции ответа на оценку модели

# Результат
|  |  Количество выбранных ответов моделью |
|---------------------------------------|---|
| rag_answers                           | 44|
| giga_chat_answers                     | 20|
| Всего                                 | 64|

Итоговая метрика побед RAG-системы: 68.75 %
