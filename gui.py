from SettingsManager import SettingsManager, CollapsibleSection
from chat_model import ChatModel
from server import ChatServer

from Silero import TelegramBotHandler

import os
import base64
from pathlib import Path
import json
import glob

import asyncio
import threading

import tkinter as tk
from tkinter import ttk

from utils import SH

import sounddevice as sd
from SpeechRecognition import SpeechRecognition


class ChatGUI:
    def __init__(self):

        self.silero_connected = False
        self.game_connected = False
        self.ConnectedToGame = False

        self.chat_window = None
        self.token_count_label = None

        self.bot_handler = None
        self.bot_handler_ready = False

        self.selected_microphone = ""
        self.device_id = 0
        self.user_entry = None
        self.user_input = ""

        self.api_key = ""
        self.api_key_res = ""
        self.api_url = ""
        self.api_model = ""

        self.makeRequest = False
        self.api_hash = ""
        self.api_id = ""
        self.phone = ""

        self.settings = SettingsManager("Settings/settings.json")

        try:
            target_folder = "Settings"
            os.makedirs(target_folder, exist_ok=True)
            self.config_path = os.path.join(target_folder, "settings.json")
            self.load_api_settings(False)  # Загружаем настройки при инициализации
        except Exception as e:
            print("Не удалось удачно получить из системных переменных все данные", e)

        self.model = ChatModel(self, self.api_key, self.api_key_res, self.api_url, self.api_model, self.gpt4free_model, self.makeRequest)
        self.server = ChatServer(self, self.model)
        self.server_thread = None
        self.running = False
        self.start_server()
        self.textToTalk = ""
        self.textSpeaker = "/Speaker Mita"
        self.silero_turn_off_video = False
        self.patch_to_sound_file = ""

        self.root = tk.Tk()
        self.root.title("Чат с NeuroMita")

        self.last_price = ""

        self.delete_all_wav_files()
        self.setup_ui()

        try:
            self.load_mic_settings()
        except Exception as e:
            print("Не удалось удачно получить настройки микрофона", e)

        # Событие для синхронизации потоков
        self.loop_ready_event = threading.Event()

        self.loop = None  # Переменная для хранения ссылки на цикл событий
        self.asyncio_thread = threading.Thread(target=self.start_asyncio_loop, daemon=True)
        self.asyncio_thread.start()

        self.start_silero_async()

        SpeechRecognition.speach_recognition_start(self.device_id, self.loop)

        # Запуск проверки переменной textToTalk через after
        self.root.after(150, self.check_text_to_talk_or_send)

    def start_asyncio_loop(self):
        """Запускает цикл событий asyncio в отдельном потоке."""
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            print("Цикл событий asyncio успешно запущен.")
            self.loop_ready_event.set()  # Сигнализируем, что цикл событий готов
            self.loop.run_forever()
        except Exception as e:
            print(f"Ошибка при запуске цикла событий asyncio: {e}")

    def start_silero_async(self):
        """Отправляет задачу для запуска Silero в цикл событий."""
        print("Ожидание готовности цикла событий...")
        self.loop_ready_event.wait()  # Ждем, пока цикл событий будет готов
        if self.loop and self.loop.is_running():
            print("Запускаем Silero через цикл событий.")
            asyncio.run_coroutine_threadsafe(self.startSilero(), self.loop)
        else:
            print("Ошибка: Цикл событий asyncio не запущен.")

    async def startSilero(self):
        """Асинхронный запуск обработчика Telegram Bot."""
        print("Telegram Bot запускается!")
        try:
            print(f"Передаю в тг {SH(self.api_id)},{SH(self.api_hash)},{SH(self.phone)} (Должно быть не пусто)")
            self.bot_handler = TelegramBotHandler(self, self.api_id, self.api_hash, self.phone)
            await self.bot_handler.start()
            self.bot_handler_ready = True
            if self.silero_connected:
                print("ТГ успешно подключен")
            else:
                print("ТГ не подключен")

        except Exception as e:
            print(f"Ошибка при запуске Telegram Bot: {e}")
            self.silero_connected = False

    def run_in_thread(self, response):
        """Запуск асинхронной задачи в отдельном потоке."""
        # Убедимся, что цикл событий готов и запускаем задачу в том же цикле
        self.loop_ready_event.wait()  # Ждем, пока цикл событий будет готов
        if self.loop and self.loop.is_running():
            print("Запускаем асинхронную задачу в цикле событий...")
            # Здесь мы вызываем асинхронную задачу через главный цикл
            self.loop.create_task(self.run_send_and_receive(self.textToTalk, self.textSpeaker))
        else:
            print("Ошибка: Цикл событий asyncio не готов.")

    async def run_send_and_receive(self, response, speaker_command):
        """Асинхронный метод для вызова send_and_receive."""
        print("Попытка получить фразу")
        await self.bot_handler.send_and_receive(response, speaker_command)
        print("Завершение получения фразы")

    def check_text_to_talk_or_send(self):
        """Периодическая проверка переменной self.textToTalk."""

        if self.textToTalk != "":  #and not self.ConnectedToGame:
            print(f"Есть текст для отправки: {self.textToTalk}")
            # Вызываем метод для отправки текста, если переменная не пуста
            if self.loop and self.loop.is_running():

                if bool(self.settings.get("SILERO_USE")):
                    print("Цикл событий готов. Отправка текста.")
                    asyncio.run_coroutine_threadsafe(self.run_send_and_receive(self.textToTalk, self.textSpeaker),
                                                     self.loop)
                self.textToTalk = ""  # Очищаем текст после отправки
                print("Выполнено")
            else:
                print("Ошибка: Цикл событий не готов.")


        text_from_recognition = SpeechRecognition.receive_text()
        if bool(self.settings.get("MIC_ACTIVE")) and text_from_recognition and self.user_entry:
            self.user_entry.insert(tk.END, text_from_recognition)
            self.user_input = self.user_entry.get("1.0", "end-1c").strip()

        # Перезапуск проверки через 100 миллисекунд
        self.root.after(100, self.check_text_to_talk_or_send)  # Это обеспечит постоянную проверку

    def clear_user_input(self):
        self.user_input = ""
        self.user_entry.delete(1.0, 'end')

    def start_server(self):
        """Запускает сервер в отдельном потоке."""
        if not self.running:
            self.running = True
            self.server.start()  # Инициализация сокета
            self.server_thread = threading.Thread(target=self.run_server_loop, daemon=True)
            self.server_thread.start()
            print("Сервер запущен.")

    def stop_server(self):
        """Останавливает сервер."""
        if self.running:
            self.running = False
            self.server.stop()
            print("Сервер остановлен.")

    def run_server_loop(self):
        """Цикл обработки подключений сервера."""
        while self.running:
            needUpdate = self.server.handle_connection()
            if needUpdate:
                self.load_chat_history()

    def setup_ui(self):
        self.root.config(bg="#2c2c2c")  # Установите темный цвет фона для всего окна

        main_frame = tk.Frame(self.root, bg="#2c2c2c")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Первый столбец
        left_frame = tk.Frame(main_frame, bg="#2c2c2c")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.chat_window = tk.Text(
            left_frame, height=30, width=80, state=tk.NORMAL,
            bg="#1e1e1e", fg="#ffffff", insertbackground="white", wrap=tk.WORD,
            font=("Arial", 12)
        )

        self.chat_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        input_frame = tk.Frame(left_frame, bg="#2c2c2c")
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        self.user_entry = tk.Text(input_frame, height=3, width=50, bg="#1e1e1e", fg="#ffffff", insertbackground="white",
                                  font=("Arial", 12))
        self.user_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.user_entry.bind("<KeyRelease>", self.update_token_count)

        self.send_button = tk.Button(
            input_frame, text="Отправить", command=self.send_message,
            bg="#9370db", fg="#ffffff", font=("Arial", 12)
        )
        self.send_button.pack(side=tk.RIGHT, padx=5)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Второй столбец
        right_frame = tk.Frame(main_frame, bg="#2c2c2c")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.setup_microphone_controls(right_frame)

        self.setup_silero_controls(right_frame)
        self.setup_mita_controls(right_frame)
        self.setup_model_controls(right_frame)
        # Передаем right_frame как родителя
        self.setup_status_indicators(right_frame)

        # Настройка элементов управления
        # Создаем контейнер для всех элементов управления
        self.controls_frame = tk.Frame(right_frame, bg="#2c2c2c")
        self.controls_frame.pack(fill=tk.X, pady=3)

        # Настройка элементов управления
        #self.setup_control("Отношение к игроку", "attitude", self.model.attitude)
        #self.setup_control("Скука", "boredom", self.model.boredom)
        #self.setup_control("Стресс", "stress", self.model.stress)
        #self.setup_secret_control()

        self.setup_history_controls(right_frame)
        self.setup_debug_controls(right_frame)
        self.setup_api_controls(right_frame)

        #self.setup_advanced_controls(right_frame)

        self.load_chat_history()

    def insert_message(self, role, content):
        if role == "user":
            # Вставляем имя пользователя с зеленым цветом, а текст — обычным
            self.chat_window.insert(tk.END, "Вы: ", "user_name")
            self.chat_window.insert(tk.END, f"{content}\n")
        elif role == "assistant":
            # Вставляем имя Миты с синим цветом, а текст — обычным
            self.chat_window.insert(tk.END, f"{self.model.current_character.name}: ", "gpt_name")
            self.chat_window.insert(tk.END, f"{content}\n\n")

    def setup_status_indicators(self, parent):
        # Создаем фрейм для индикаторов
        status_frame = tk.Frame(parent, bg="#2c2c2c")
        status_frame.pack(fill=tk.X, pady=3)

        # Переменные статуса
        self.game_connected = tk.BooleanVar(value=False)  # Статус подключения к игре
        self.silero_connected = tk.BooleanVar(value=False)  # Статус подключения к Silero

        # Галки для подключения
        self.game_status_checkbox = tk.Checkbutton(
            status_frame,
            text="Подключение к игре",
            variable=self.game_connected,
            state="disabled",
            bg="#2c2c2c",
            fg="#ffffff",
            selectcolor="#2c2c2c"
        )
        self.game_status_checkbox.pack(side=tk.LEFT, padx=5, pady=4)

        self.silero_status_checkbox = tk.Checkbutton(
            status_frame,
            text="Подключение к Silero",
            variable=self.silero_connected,
            state="disabled",
            bg="#2c2c2c",
            fg="#ffffff",
            selectcolor="#2c2c2c"
        )
        self.silero_status_checkbox.pack(side=tk.LEFT, padx=5, pady=4)

    def updateAll(self):
        self.update_status_colors()
        self.update_debug_info()

    def update_status_colors(self):
        self.game_connected = tk.BooleanVar(value=self.ConnectedToGame)  # Статус подключения к игре
        # Обновление цвета для подключения к игре
        game_color = "#00ff00" if self.ConnectedToGame else "#ffffff"
        self.game_status_checkbox.config(fg=game_color)

        # Обновление цвета для подключения к Silero
        silero_color = "#00ff00" if self.silero_connected else "#ffffff"
        self.silero_status_checkbox.config(fg=silero_color)

    def setup_control(self, label_text, attribute_name, initial_value):
        """
        Создает элементы управления для настроения, скуки и стресса.
        :param label_text: Текст метки (например, "Настроение").
        :param attribute_name: Имя атрибута модели (например, "attitude").
        :param initial_value: Начальное значение атрибута.
        """
        frame = tk.Frame(self.controls_frame, bg="#2c2c2c")
        frame.pack(fill=tk.X, pady=5)

        # Метка для отображения текущего значения
        label = tk.Label(frame, text=f"{label_text}: {initial_value}", bg="#2c2c2c", fg="#ffffff")
        label.pack(side=tk.LEFT, padx=5)

        # Кнопки для увеличения и уменьшения значения
        up_button = tk.Button(
            frame, text="+", command=lambda: self.adjust_value(attribute_name, 15, label),
            bg="#8a2be2", fg="#ffffff"
        )
        up_button.pack(side=tk.RIGHT, padx=5)

        down_button = tk.Button(
            frame, text="-", command=lambda: self.adjust_value(attribute_name, -15, label),
            bg="#8a2be2", fg="#ffffff"
        )
        down_button.pack(side=tk.RIGHT, padx=5)

        # Сохраняем ссылку на метку для обновления
        setattr(self, f"{attribute_name}_label", label)

    def setup_secret_control(self):
        """
        Создает чекбокс для управления состоянием "Секрет раскрыт".
        """
        frame = tk.Frame(self.controls_frame, bg="#2c2c2c")
        frame.pack(fill=tk.X, pady=5)

        self.secret_var = tk.BooleanVar(value=self.model.secretExposed)

        secret_checkbox = tk.Checkbutton(
            frame, text="Секрет раскрыт", variable=self.secret_var,
            bg="#2c2c2c", fg="#ffffff", command=self.adjust_secret
        )
        secret_checkbox.pack(side=tk.LEFT, padx=5)

    def adjust_value(self, attribute_name, delta, label):
        """
        Обновляет значение атрибута модели и метки.
        :param attribute_name: Имя атрибута модели (например, "attitude").
        :param delta: Изменение значения (например, +15 или -15).
        :param label: Метка, которую нужно обновить.
        """
        current_value = getattr(self.model, attribute_name)
        new_value = current_value + delta
        setattr(self.model, attribute_name, new_value)

        # Обновляем текст метки
        label.config(text=f"{label.cget('text').split(':')[0]}: {new_value}")

    def adjust_secret(self):
        """
        Обновляет состояние "Секрет раскрыт" в модели.
        """
        self.model.secretExposed = self.secret_var.get()

    def setup_history_controls(self, parent):
        history_frame = tk.Frame(parent, bg="#2c2c2c")
        history_frame.pack(fill=tk.X, pady=4)

        clear_button = tk.Button(
            history_frame, text="Очистить историю", command=self.clear_history,
            bg="#8a2be2", fg="#ffffff"
        )
        clear_button.pack(side=tk.LEFT, padx=5)

        save_history_button = tk.Button(
            history_frame, text="Сохранить историю", command=self.model.save_chat_history,
            bg="#8a2be2", fg="#ffffff"
        )
        save_history_button.pack(side=tk.LEFT, padx=10)

        reload_prompts_button = tk.Button(
            history_frame, text="Перезагрузить промпты", command=self.model.reload_promts,
            bg="#8a2be2", fg="#ffffff"
        )
        reload_prompts_button.pack(side=tk.LEFT, padx=15)

    def load_chat_history(self):

        chat_history = self.model.current_character.load_history()
        for entry in chat_history["messages"]:
            role = entry["role"]
            content = entry["content"]
            self.insert_message(role, content)

        self.update_debug_info()

    def setup_debug_controls(self, parent):
        debug_frame = tk.Frame(parent, bg="#2c2c2c")
        debug_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.debug_window = tk.Text(
            debug_frame, height=5, width=50, bg="#1e1e1e", fg="#ffffff",
            state=tk.NORMAL, wrap=tk.WORD, insertbackground="white"
        )
        self.debug_window.pack(fill=tk.BOTH, expand=True)

        self.update_debug_info()  # Отобразить изначальное состояние переменных

    def setup_api_controls(self, parent):
        api_frame = tk.Frame(parent, bg="#2c2c2c")
        api_frame.pack(fill=tk.X, pady=3)

        self.show_api_var = tk.BooleanVar(value=False)

        api_toggle = tk.Checkbutton(
            api_frame, text="Показать настройки API", variable=self.show_api_var,
            command=lambda: self.pack_unpack(self.show_api_var, self.api_settings_frame), bg="#2c2c2c", fg="#ffffff"
        )
        api_toggle.pack(side=tk.LEFT, padx=4)

        self.api_settings_frame = tk.Frame(parent, bg="#2c2c2c")

        # Элементы в одном столбце
        tk.Label(
            self.api_settings_frame, text="API-ключ:", bg="#2c2c2c", fg="#ffffff"
        ).grid(row=0, column=0, padx=4, pady=4, sticky=tk.W)

        self.api_key_entry = tk.Entry(self.api_settings_frame, width=50, bg="#1e1e1e", fg="#ffffff",
                                      insertbackground="white")
        self.api_key_entry.grid(row=0, column=1, padx=4, pady=4, sticky=tk.W)

        tk.Label(
            self.api_settings_frame, text="резервный API-ключ:", bg="#2c2c2c", fg="#ffffff"
        ).grid(row=1, column=0, padx=4, pady=4, sticky=tk.W)

        self.api_key_res_entry = tk.Entry(self.api_settings_frame, width=50, bg="#1e1e1e", fg="#ffffff",
                                          insertbackground="white")
        self.api_key_res_entry.grid(row=1, column=1, padx=4, pady=4, sticky=tk.W)

        tk.Label(
            self.api_settings_frame, text="Ссылка:", bg="#2c2c2c", fg="#ffffff"
        ).grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)

        self.api_url_entry = tk.Entry(self.api_settings_frame, width=50, bg="#1e1e1e", fg="#ffffff",
                                      insertbackground="white")
        self.api_url_entry.grid(row=2, column=1, padx=4, pady=5, sticky=tk.W)

        tk.Label(
            self.api_settings_frame, text="Модель:", bg="#2c2c2c", fg="#ffffff"
        ).grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)

        self.api_model_entry = tk.Entry(self.api_settings_frame, width=50, bg="#1e1e1e", fg="#ffffff",
                                        insertbackground="white")
        self.api_model_entry.grid(row=3, column=1, padx=4, pady=4, sticky=tk.W)

        tk.Label(
            self.api_settings_frame, text="Telegram API ID:", bg="#2c2c2c", fg="#ffffff"
        ).grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)

        self.api_id_entry = tk.Entry(self.api_settings_frame, width=50, bg="#1e1e1e", fg="#ffffff",
                                     insertbackground="white")
        self.api_id_entry.grid(row=4, column=1, padx=4, pady=4, sticky=tk.W)

        tk.Label(
            self.api_settings_frame, text="Telegram API Hash:", bg="#2c2c2c", fg="#ffffff"
        ).grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)

        self.api_hash_entry = tk.Entry(self.api_settings_frame, width=50, bg="#1e1e1e", fg="#ffffff",
                                       insertbackground="white")
        self.api_hash_entry.grid(row=5, column=1, padx=4, pady=4, sticky=tk.W)

        tk.Label(
            self.api_settings_frame, text="Telegram Phone:", bg="#2c2c2c", fg="#ffffff"
        ).grid(row=6, column=0, padx=5, pady=5, sticky=tk.W)

        self.phone_entry = tk.Entry(self.api_settings_frame, width=50, bg="#1e1e1e", fg="#ffffff",
                                    insertbackground="white")
        self.phone_entry.grid(row=6, column=1, padx=4, pady=4, sticky=tk.W)

        # Переменная для тумблера
        self.makeRequest_entry = tk.Checkbutton(
            self.api_settings_frame, text="openapi or req", variable=self.show_api_var,
            command=self.toggle_makeRequest, bg="#2c2c2c", fg="#ffffff"
        )
        self.makeRequest_entry.grid(row=7, column=0, padx=15, pady=0, sticky=tk.W)
        self.toggle_makeRequest(False)

        save_button = tk.Button(
            self.api_settings_frame, text="Сохранить", command=self.save_api_settings,
            bg="#8a2be2", fg="#ffffff"
        )
        save_button.grid(row=7, column=1, padx=5, sticky=tk.E)

        # Обновляем поля ввода
        self.api_key_entry.insert(0, self.api_key)
        self.api_key_res_entry.insert(0, self.api_key_res)
        self.api_url_entry.insert(0, self.api_url)
        self.api_model_entry.insert(0, self.api_model)
        self.api_id_entry.insert(0, self.api_id)
        self.api_hash_entry.insert(0, self.api_hash)
        self.phone_entry.insert(0, self.phone)

    def setup_silero_controls(self, parent):
        # Основные настройки
        telegram_config = [
            {'label': 'Использовать силеро', 'key': 'SILERO_USE', 'type': 'checkbutton', 'default': True},
            {'label': 'Максимальное ожидание', 'key': 'SILERO_TIME', 'type': 'entry', 'default': 7,
             'validation': self.validate_number}
        ]

        self.create_settings_section(parent, "Настройка силеро", telegram_config)

    def setup_mita_controls(self, parent):
        # Основные настройки
        mita_config = [
            {'label': 'Персонаж', 'key': 'CHARACTER', 'type': 'combobox', 'options': self.model.get_all_mitas(),
             'default': "Mita"}
        ]

        self.create_settings_section(parent, "Выбор персонажа", mita_config)

    def setup_model_controls(self, parent):
        # Основные настройки
        mita_config = [
            {'label': 'Использовать gpt4free', 'key': 'gpt4free', 'type': 'checkbutton', 'default_checkbutton': False},
            {'label': 'Лимит сообщений', 'key': 'MODEL_MESSAGE_LIMIT', 'type': 'entry', 'default': 40},
            {'label': 'Использовать модель', 'key': 'gpt4free_model', 'type': 'entry', 'default': "gpt-4o-mini"}
        ]

        self.create_settings_section(parent, "Настройки модели", mita_config)

    def validate_number(self, new_value):
        return 0 < len(new_value) <= 30  # Пример простой валидации

    def pack_unpack(self, var, frame):
        """
        Показывает или скрывает фрейм в зависимости от состояния var.
        """
        if var.get():
            frame.pack(fill=tk.X, padx=10, pady=10)
        else:
            frame.pack_forget()

    def toggle_makeRequest(self, change_to_opposite=True):
        if change_to_opposite:
            self.makeRequest = not self.makeRequest

        if self.makeRequest:
            self.makeRequest_entry.config(text="Сейчас режим прокси Gemini")
        else:
            self.makeRequest_entry.config(text="Сейчас обычный режим")

    def save_api_settings(self):
        """Собирает данные из полей ввода и сохраняет только непустые значения, не перезаписывая существующие."""
        try:
            # Загружаем текущие настройки, если файл существует
            if os.path.exists(self.config_path):
                with open(self.config_path, "rb") as f:
                    encoded = f.read()
                decoded = base64.b64decode(encoded)
                settings = json.loads(decoded.decode("utf-8"))
            else:
                settings = {}

            # Обновляем настройки новыми значениями, если они не пустые
            if api_key := self.api_key_entry.get().strip():
                print("Сохранение апи ключа")
                settings["NM_API_KEY"] = api_key
            if api_key_res := self.api_key_res_entry.get().strip():
                print("Сохранение резервного апи ключа")
                settings["NM_API_KEY_RES"] = api_key_res
            if api_url := self.api_url_entry.get().strip():
                print("Сохранение апи ссылки")
                settings["NM_API_URL"] = api_url
            else:
                print("Сохранение ссылку по умолчанию, тк она пуста")
                settings["NM_API_URL"] = "https://openrouter.ai/api/v1"
            if api_model := self.api_model_entry.get().strip():
                print("Сохранение апи модели")
                settings["NM_API_MODEL"] = api_model
            else:
                print("Сохранение модель по умолчанию, тк настройка пуста")
                settings["NM_API_MODEL"] = "google/gemini-2.0-pro-exp-02-05:free"

            if api_id := self.api_id_entry.get().strip():
                print("Сохранение тг айди")
                settings["NM_TELEGRAM_API_ID"] = api_id
            if api_hash := self.api_hash_entry.get().strip():
                print("Сохранение тг хеш")
                settings["NM_TELEGRAM_API_HASH"] = api_hash
            if phone := self.phone_entry.get().strip():
                print("Сохранение тг телефона")
                settings["NM_TELEGRAM_PHONE"] = phone

            # Булево значение сохраняем всегда
            settings["NM_API_REQ"] = self.makeRequest

            # Сериализация и кодирование
            json_data = json.dumps(settings, ensure_ascii=False)
            encoded = base64.b64encode(json_data.encode("utf-8"))

            # Сохраняем в файл
            with open(self.config_path, "wb") as f:
                f.write(encoded)
            print("Настройки успешно сохранены в файл")

        except Exception as e:
            print(f"Ошибка сохранения: {e}")

        # Сразу же их загружаем
        self.load_api_settings(update_model=True)

        if not self.silero_connected:
            print("Попытка запустить силеро заново")
            self.start_silero_async()

    def load_api_settings(self, update_model):
        """Загружает настройки из файла"""
        print("Начинаю загрузку настроек")

        if not os.path.exists(self.config_path):
            print("Не найден файл настроек")
            #self.save_api_settings(False)
            return

        try:
            # Читаем закодированные данные из файла
            with open(self.config_path, "rb") as f:
                encoded = f.read()
            # Декодируем из base64
            decoded = base64.b64decode(encoded)
            # Десериализуем JSON
            settings = json.loads(decoded.decode("utf-8"))

            # Устанавливаем значения
            self.api_key = settings.get("NM_API_KEY", "")
            self.api_key_res = settings.get("NM_API_KEY_RES", "")
            self.api_url = settings.get("NM_API_URL", "")
            self.api_model = settings.get("NM_API_MODEL", "")
            self.makeRequest = settings.get("NM_API_REQ", False)
            self.gpt4free_model = settings.get("NM_GPT4FREE_MODEL", "")

            # ТГ
            self.api_id = settings.get("NM_TELEGRAM_API_ID", "")
            self.api_hash = settings.get("NM_TELEGRAM_API_HASH", "")
            self.phone = settings.get("NM_TELEGRAM_PHONE", "")

            print(
                f"Итого загружено {SH(self.api_key)},{SH(self.api_key_res)},{self.api_url},{self.api_model},{self.makeRequest} (Должно быть не пусто)")
            print(f"По тг {SH(self.api_id)},{SH(self.api_hash)},{SH(self.phone)} (Должно быть не пусто если тг)")
            if update_model:
                if self.api_key:
                    self.model.api_key = self.api_key
                if self.api_url:
                    self.model.api_url = self.api_url
                if self.api_model:
                    self.model.api_model = self.api_model

                self.model.makeRequest = self.makeRequest
                self.model.update_openai_client()

            print("Настройки загружены из файла")
        except Exception as e:
            print(f"Ошибка загрузки: {e}")

    def paste_from_clipboard(self, event=None):
        try:
            clipboard_content = self.root.clipboard_get()
            self.user_entry.insert(tk.INSERT, clipboard_content)
        except tk.TclError:
            pass  # Если буфер обмена пуст, ничего не делаем

    def copy_to_clipboard(self, event=None):
        try:
            # Получение выделенного текста из поля ввода
            selected_text = self.user_entry.selection_get()
            # Копирование текста в буфер обмена
            self.root.clipboard_clear()
            self.root.clipboard_append(selected_text)
            self.root.update()  # Обновление буфера обмена
        except tk.TclError:
            # Если текст не выделен, ничего не делать
            pass

    def toggle_api_settings(self):
        if self.show_api_var.get():
            self.api_settings_frame.pack(fill=tk.X, padx=10, pady=10)
        else:
            self.api_settings_frame.pack_forget()

    def update_debug_info(self):
        """Обновить окно отладки с отображением актуальных данных."""
        self.debug_window.delete(1.0, tk.END)  # Очистить старые данные

        debug_info = (self.model.current_character.current_variables_string())

        self.debug_window.insert(tk.END, debug_info)

    def update_token_count(self, event=None):
        if False and self.model.hasTokenizer:
            user_input = self.user_entry.get("1.0", "end-1c")
            token_count, cost = self.model.calculate_cost(user_input)
            self.token_count_label.config(
                text=f"ВЫКЛЮЧЕНО!! Токенов: {token_count}/{self.model.max_input_tokens} | Ориент. стоимость: {cost:.4f} ₽"
            )
            self.update_debug_info()

    def insertDialog(self, input_text="", response=""):
        if input_text != "":
            self.chat_window.insert(tk.END, "Вы: ", "user_name")
            self.chat_window.insert(tk.END, f"{input_text}\n")
        if response != "":
            self.chat_window.insert(tk.END, "Мита: ", "gpt_name")
            self.chat_window.insert(tk.END, f"{response}\n\n")

    def send_message(self, system_input=""):
        user_input = self.user_entry.get("1.0", "end-1c")
        if not user_input.strip() and system_input == "":
            return

        if user_input != "":
            self.insert_message("user", user_input)
            self.user_entry.delete("1.0", "end")

        response = self.model.generate_response(user_input, system_input)

        self.insert_message("assistant", response)
        self.user_entry.delete("1.0", "end")
        self.updateAll()
        # Отправка сообщения на сервер
        if self.server:
            try:
                # Отправляем сообщение клиенту через сервер
                if self.server.client_socket:
                    self.server.send_message_to_server(response)
                    print("Сообщение отправлено на сервер (связь с игрой есть)")
                else:
                    print("Нет активного подключения к клиенту игры")
            except Exception as e:
                print(f"Ошибка при отправке сообщения на сервер: {e}")

    def clear_history(self):
        self.model.current_character.clear_history()
        self.chat_window.delete(1.0, tk.END)
        self.update_debug_info()

    # region Microphone

    """Спасибо Nelxi (distrane25)"""

    def setup_microphone_controls(self, parent):
        mic_frame = tk.Frame(parent, bg="#2c2c2c")
        mic_frame.pack(fill=tk.X, pady=5)

        tk.Label(
            mic_frame,
            text="Микрофон:",
            bg="#2c2c2c",
            fg="#ffffff"
        ).pack(side=tk.LEFT, padx=5)

        self.mic_combobox = ttk.Combobox(
            mic_frame,
            values=self.get_microphone_list(),
            state="readonly",
            width=30
        )
        self.mic_combobox.pack(side=tk.LEFT, padx=5)
        self.mic_combobox.bind("<<ComboboxSelected>>", self.on_mic_selected)

        refresh_btn = tk.Button(
            mic_frame,
            text="↻",
            command=self.update_mic_list,
            bg="#8a2be2",
            fg="#ffffff",
            width=2
        )
        refresh_btn.pack(side=tk.LEFT, padx=5)

        self.create_setting_widget(mic_frame, 'Распознавание', "MIC_ACTIVE", widget_type='checkbutton', default_checkbutton=False)

    def get_microphone_list(self):
        try:
            devices = sd.query_devices()
            input_devices = [
                f"{d['name']} ({i})"
                for i, d in enumerate(devices)
                if d['max_input_channels'] > 0
            ]
            return input_devices
        except Exception as e:
            print(f"Ошибка получения списка микрофонов: {e}")
            return []

    def update_mic_list(self):
        self.mic_combobox['values'] = self.get_microphone_list()

    def on_mic_selected(self, event):
        selection = self.mic_combobox.get()
        if selection:
            self.selected_microphone = selection.split(" (")[0]
            device_id = int(selection.split(" (")[-1].replace(")", ""))
            self.device_id = device_id
            print(f"Выбран микрофон: {self.selected_microphone} (ID: {device_id})")
            self.save_mic_settings(device_id)

    def save_mic_settings(self, device_id):
        try:
            with open(self.config_path, "rb") as f:
                encoded = f.read()
            decoded = base64.b64decode(encoded)
            settings = json.loads(decoded.decode("utf-8"))
        except FileNotFoundError:
            settings = {}

        settings["NM_MICROPHONE_ID"] = device_id
        settings["NM_MICROPHONE_NAME"] = self.selected_microphone

        json_data = json.dumps(settings, ensure_ascii=False)
        encoded = base64.b64encode(json_data.encode("utf-8"))
        with open(self.config_path, "wb") as f:
            f.write(encoded)

    def load_mic_settings(self):
        try:
            with open(self.config_path, "rb") as f:
                encoded = f.read()
            decoded = base64.b64decode(encoded)
            settings = json.loads(decoded.decode("utf-8"))

            device_id = settings.get("NM_MICROPHONE_ID", 0)
            device_name = settings.get("NM_MICROPHONE_NAME", "")

            devices = sd.query_devices()
            if device_id < len(devices):
                self.selected_microphone = device_name
                self.device_id = device_id
                self.mic_combobox.set(f"{device_name} ({device_id})")
                print(f"Загружен микрофон: {device_name} (ID: {device_id})")

        except Exception as e:
            print(f"Ошибка загрузки настроек микрофона: {e}")

    # endregion

    #region SettingGUI

    def all_settings_actions(self, key, value):
        ...
        if key == "SILERO_TIME":
            self.bot_handler.silero_time_limit = int(value)

        elif key == "CHARACTER":
            self.model.current_character_to_change = value

        elif key == "MODEL_MESSAGE_LIMIT":
            self.model.memory_limit = value
        
        elif key == "gpt4free_model":
            self.model.gpt4free_model = value

        elif key == "MIC_ACTIVE":
            SpeechRecognition.active = value


    def create_settings_section(self, parent, title, settings_config):
        section = CollapsibleSection(parent, title)
        section.pack(fill=tk.X, padx=5, pady=5, expand=True)

        for config in settings_config:
            widget = self.create_setting_widget(
                parent=section.content_frame,
                label=config['label'],
                setting_key=config['key'],
                widget_type=config.get('type', 'entry'),
                options=config.get('options', None),
                default=config.get('default', ''),
                validation=config.get('validation', None)
            )
            section.add_widget(widget)

        return section

    def create_setting_widget(self, parent, label, setting_key, widget_type='entry',
                              options=None, default='', default_checkbutton=False, validation=None, tooltip=None,
                              width=None, height=None, command=None):
        """
        Создает виджет настройки с различными параметрами.

        Параметры:
            parent: Родительский контейнер
            label: Текст метки
            setting_key: Ключ настройки
            widget_type: Тип виджета ('entry', 'combobox', 'checkbutton', 'button', 'scale', 'text')
            options: Опции для combobox
            default: Значение по умолчанию
            validation: Функция валидации
            tooltip: Текст подсказки
            width: Ширина виджета
            height: Высота виджета (для текстовых полей)
            command: Функция, вызываемая при изменении значения
        """
        frame = tk.Frame(parent, bg="#2c2c2c")
        frame.pack(fill=tk.X, pady=2)

        # Label
        lbl = tk.Label(frame, text=label, bg="#2c2c2c", fg="#ffffff", width=20, anchor='w')
        lbl.pack(side=tk.LEFT, padx=5)

        # Widgets
        if widget_type == 'entry':
            entry = tk.Entry(frame, bg="#1e1e1e", fg="#ffffff", insertbackground="white")
            if width:
                entry.config(width=width)
            entry.insert(0, self.settings.get(setting_key, default))
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

            def save_entry():
                self._save_setting(setting_key, entry.get())
                if command:
                    command(entry.get())

            entry.bind("<FocusOut>", lambda e: save_entry())
            entry.bind("<Return>", lambda e: save_entry())

            if validation:
                entry.config(validate="key", validatecommand=(parent.register(validation), '%P'))

        elif widget_type == 'combobox':
            var = tk.StringVar(value=self.settings.get(setting_key, default))
            cb = ttk.Combobox(frame, textvariable=var, values=options, state="readonly")
            if width:
                cb.config(width=width)
            cb.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

            def save_combobox():
                self._save_setting(setting_key, var.get())
                if command:
                    command(var.get())

            cb.bind("<<ComboboxSelected>>", lambda e: save_combobox())

        elif widget_type == 'checkbutton':
            var = tk.BooleanVar(value=self.settings.get(setting_key, default_checkbutton))
            cb = tk.Checkbutton(frame, variable=var, bg="#2c2c2c",
                                command=lambda: [self._save_setting(setting_key, var.get()),
                                                 command(var.get()) if command else None])
            cb.pack(side=tk.LEFT, padx=5)

        elif widget_type == 'button':
            btn = tk.Button(frame, text=label, bg="#8a2be2", fg="#ffffff",
                            command=lambda: [self._save_setting(setting_key, True),
                                             command() if command else None])
            if width:
                btn.config(width=width)
            btn.pack(side=tk.LEFT, padx=5)

        elif widget_type == 'scale':
            var = tk.DoubleVar(value=self.settings.get(setting_key, default))
            scale = tk.Scale(frame, from_=options[0], to=options[1], orient=tk.HORIZONTAL,
                             variable=var, bg="#2c2c2c", fg="#ffffff", highlightbackground="#2c2c2c",
                             length=200 if not width else width)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

            def save_scale(value):
                self._save_setting(setting_key, float(value))
                if command:
                    command(float(value))

            scale.config(command=save_scale)

        elif widget_type == 'text':
            text = tk.Text(frame, bg="#1e1e1e", fg="#ffffff", insertbackground="white",
                           height=height if height else 5, width=width if width else 50)
            text.insert('1.0', self.settings.get(setting_key, default))
            text.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

            def save_text():
                self._save_setting(setting_key, text.get('1.0', 'end-1c'))
                if command:
                    command(text.get('1.0', 'end-1c'))

            text.bind("<FocusOut>", lambda e: save_text())

        # Добавляем tooltip если указан
        if tooltip:
            self.create_tooltip(frame, tooltip)

        return frame

    def create_tooltip(self, widget, text):
        """Создает всплывающую подсказку для виджета"""
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry("+0+0")
        tooltip.withdraw()

        label = tk.Label(tooltip, text=text, bg="#ffffe0", relief='solid', borderwidth=1)
        label.pack()

        def enter(event):
            x = widget.winfo_rootx() + widget.winfo_width() + 5
            y = widget.winfo_rooty()
            tooltip.wm_geometry(f"+{x}+{y}")
            tooltip.deiconify()

        def leave(event):
            tooltip.withdraw()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def _save_setting(self, key, value):
        self.settings.set(key, value)
        self.settings.save_settings()

        self.all_settings_actions(key, value)

    #endregion

    def run(self):
        self.root.mainloop()

    def on_closing(self):
        self.delete_all_wav_files()
        self.stop_server()
        print("Закрываемся")
        self.root.destroy()

    def close_app(self):
        """Закрытие приложения корректным образом."""
        print("Завершение программы...")
        self.root.destroy()  # Закрывает GUI

    def delete_all_wav_files(self):
        # Получаем список всех .wav файлов в корневой директории
        wav_files = glob.glob("*.wav")

        # Проходим по каждому файлу и удаляем его
        for wav_file in wav_files:
            try:
                os.remove(wav_file)
                print(f"Удален файл: {wav_file}")
            except Exception as e:
                print(f"Ошибка при удалении файла {wav_file}: {e}")
