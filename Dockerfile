# Используем Python 3.9.21
FROM python:3.9.21-slim

# Устанавливаем зависимости системы
RUN apt-get update && apt-get install -y \
    libpoppler-cpp-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы проекта в контейнер
COPY . .

# Устанавливаем зависимости Python
RUN pip install --no-cache-dir -r requirements.txt

# Открываем порт для взаимодействия с ботом
EXPOSE 80

# Команда для запуска бота
CMD ["python", "gyn_assistant_tg.py"]