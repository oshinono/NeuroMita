import time
from io import BytesIO
import asyncio
import logging
import soundfile as sf
import numpy as np
import speech_recognition as sr
import sounddevice as sd
from collections import deque
from threading import Lock
from Logger import logger
import httpx
import json
import wave
import sys
from vosk import Model, KaldiRecognizer, SetLogLevel
import io

# You can set log level to -1 to disable debug messages
SetLogLevel(1) # Возвращено к 0

class AudioState:
    def __init__(self):
        self.is_recording = False
        self.audio_buffer = []
        self.last_sound_time = time.time()
        self.is_playing = False
        self.lock = asyncio.Lock()
        self.vc = None
        self.max_buffer_size = 9999999

    async def add_to_buffer(self, data):
        async with self.lock:
            if len(self.audio_buffer) >= self.max_buffer_size:
                self.audio_buffer = self.audio_buffer[-self.max_buffer_size // 2:]  # Сохраняем последние 50%
            self.audio_buffer.append(data.copy())


audio_state = AudioState()


class SpeechRecognition:
    user_input = ""
    microphone_index = 0
    active = True
    _recognizer_type = "google"  # 'google' или 'vosk'
   # vosk_model = "vosk-model-ru-0.10" #vosk-model-small-ru
    vosk_model = "vosk-model-small-ru-0.22"

    SAMPLE_RATE = 32000
    CHUNK_SIZE = 1024 # Увеличено для уменьшения переполнения буфера
    TIMEOUT_MESSAGE = True
    SILENCE_THRESHOLD = 0.02  # Порог тишины
    SILENCE_DURATION = 4  # Длительность тишины для завершения записи
    MAX_RECORDING_DURATION = 15 # Максимальная длительность записи (сек)
    SILENCE_TIMEOUT = 2.0
    MIN_RECORDING = 1.0
    MIN_RECORDING_DURATION = 1  # Минимальная длительность записи
    BUFFER_TIMEOUT = 0.05
    VOSK_PROCESS_INTERVAL = 0.3 # Увеличено для уменьшения переполнения буфера
    _text_lock = Lock()
    _text_buffer = deque(maxlen=15)  # Храним последние 10 фраз
    _current_text = ""
    _last_delimiter = ". "

    @staticmethod
    def set_recognizer_type(recognizer_type: str):
        if recognizer_type in ["google", "vosk"]:
            SpeechRecognition._recognizer_type = recognizer_type
            logger.info(f"Тип распознавателя установлен на: {recognizer_type}")
        else:
            logger.warning(f"Неизвестный тип распознавателя: {recognizer_type}. Используется 'google'.")


    @staticmethod
    def receive_text() -> str:
        """Получение и сброс текста (синхронный метод)"""
        with SpeechRecognition._text_lock:
            result = " ".join(SpeechRecognition._text_buffer).strip()
            SpeechRecognition._text_buffer.clear()
            SpeechRecognition._current_text = ""
            #logger.debug(f"Returned text: {result}")
            return result

    @staticmethod
    def list_microphones():
        return sr.Microphone.list_microphone_names()

    @staticmethod
    async def handle_voice_message(recognized_text: str) -> None:
        """Асинхронная обработка текста"""
        text_clean = recognized_text.strip()
        if text_clean:
            with SpeechRecognition._text_lock:
                # Определение разделителя
                last_char = SpeechRecognition._current_text[-1] if SpeechRecognition._current_text else ""
                delimiter = "" if last_char in {'.', '!', '?', ','} else " "

                SpeechRecognition._text_buffer.append(text_clean)
                SpeechRecognition._current_text += f"{delimiter}{text_clean}"

    @staticmethod
    def _stereo_to_mono(audio_data):
        return np.mean(audio_data, axis=1, dtype=audio_data.dtype)

    _vosk_model_instance = None
    _vosk_rec_instance = None

    @staticmethod
    def _init_vosk_recognizer():
        if SpeechRecognition._vosk_model_instance is None:
            model_path = f"SpeechRecognitionModels/Vosk/{SpeechRecognition.vosk_model}"
            try:
                SpeechRecognition._vosk_model_instance = Model(model_path)
                logger.info(f"Модель Vosk загружена из: {model_path}")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели Vosk из {model_path}: {e}")
                return False
        
        if SpeechRecognition._vosk_rec_instance is None:
            SpeechRecognition._vosk_rec_instance = KaldiRecognizer(SpeechRecognition._vosk_model_instance, SpeechRecognition.SAMPLE_RATE)
            SpeechRecognition._vosk_rec_instance.SetWords(True)
            SpeechRecognition._vosk_rec_instance.SetPartialWords(True)
            logger.info("Распознаватель Vosk инициализирован.")
        return True

    @staticmethod
    async def recognize_vosk(audio_data: np.ndarray) -> str | None:
        if not SpeechRecognition._init_vosk_recognizer():
            return None

        # Vosk ожидает int16, а sounddevice дает float32.
        # Преобразуем float32 в int16
        audio_data_int16 = (audio_data * 32767).astype(np.int16)

        # Создаем in-memory wave file
        bytes_io = io.BytesIO()
        with wave.open(bytes_io, 'wb') as mono_wf:
            mono_wf.setnchannels(1)
            mono_wf.setsampwidth(2)  # 2 bytes for int16
            mono_wf.setframerate(SpeechRecognition.SAMPLE_RATE)
            mono_wf.writeframes(audio_data_int16.tobytes())
        
        bytes_io.seek(0)
        
        recognized_text = ''
        rec = SpeechRecognition._vosk_rec_instance # Используем инициализированный распознаватель

        # Сбрасываем состояние распознавателя для нового аудио
        rec.Reset()

        # Читаем данные из in-memory wave file
        with wave.open(bytes_io, 'rb') as wf:
            while True:
                data = wf.readframes(4000) # Читаем по 4000 фреймов
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    stepResult = rec.Result()
                    recognized_json = json.loads(stepResult)
                    if 'text' in recognized_json and recognized_json['text']:
                        recognized_text += ' ' + recognized_json['text'] + '.'
                else:
                    # Partial results are not used for final text, but can be logged for debugging
                    pass
        
        final_result = rec.FinalResult()
        final_json = json.loads(final_result)
        if 'text' in final_json and final_json['text']:
            recognized_text += ' ' + final_json['text'] + '.'
        
        return recognized_text.strip()


    @staticmethod
    async def live_recognition() -> None:
        # Этот метод будет работать по-разному в зависимости от выбранного распознавателя.
        # Для Google будет использоваться speech_recognition.Microphone.
        # Для Vosk будет использоваться sounddevice для прямого захвата и отправки в Vosk API.

        if SpeechRecognition._recognizer_type == "google":
            recognizer = sr.Recognizer()
            with sr.Microphone(device_index=SpeechRecognition.microphone_index, sample_rate=SpeechRecognition.SAMPLE_RATE,
                               chunk_size=SpeechRecognition.CHUNK_SIZE) as source:
                logger.info(
                    f"Используется микрофон: {sr.Microphone.list_microphone_names()[SpeechRecognition.microphone_index]}")
                recognizer.adjust_for_ambient_noise(source)
                logger.info("Скажите что-нибудь (Google)...")

                while SpeechRecognition.active:
                    try:
                        audio = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: recognizer.listen(source, timeout=5)
                        )

                        text = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: recognizer.recognize_google(audio, language="ru-RU")
                        )
                        if not text:
                            text = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: recognizer.recognize_google(audio, language="en-EN")
                            )

                        if text:
                            await SpeechRecognition.handle_voice_message(text)

                    except sr.WaitTimeoutError:
                        if SpeechRecognition.TIMEOUT_MESSAGE:
                            ...
                    except sr.UnknownValueError:
                        ...
                    except Exception as e:
                        logger.error(f"Ошибка при распознавании Google: {e}")
                        break
        elif SpeechRecognition._recognizer_type == "vosk":
            if not SpeechRecognition._init_vosk_recognizer():
                logger.error("Не удалось инициализировать Vosk распознаватель. Отмена live_recognition.")
                return

            logger.info(f"Используется микрофон: {sr.Microphone.list_microphone_names()[SpeechRecognition.microphone_index]}")
            logger.info("Скажите что-нибудь (Vosk)...")

            # Используем sounddevice для захвата аудио в реальном времени
            with sd.InputStream(
                samplerate=SpeechRecognition.SAMPLE_RATE,
                channels=1,
                dtype='float32',
                blocksize=SpeechRecognition.CHUNK_SIZE,
                device=SpeechRecognition.microphone_index
            ) as stream:
                while SpeechRecognition.active:
                    try:
                        # Читаем данные из потока
                        data, overflowed = stream.read(SpeechRecognition.CHUNK_SIZE)
                        if overflowed:
                            logger.warning("Переполнение буфера аудиопотока!")

                        # Логируем информацию о полученных аудиоданных
                        if data.size > 0:
                            rms_val = np.sqrt(np.mean(data ** 2))
                            logger.debug(f"Получены аудиоданные. Размер: {data.size}, RMS: {rms_val:.4f}")
                            if rms_val < SpeechRecognition.SILENCE_THRESHOLD:
                                logger.debug("Обнаружена тишина.")
                        else:
                            logger.debug("Получены пустые аудиоданные.")

                        # Преобразуем float32 в int16 для Vosk
                        audio_data_int16 = (data * 32767).astype(np.int16)
                        
                        # Передаем данные в Vosk
                        if SpeechRecognition._vosk_rec_instance.AcceptWaveform(audio_data_int16.tobytes()):
                            result_json = json.loads(SpeechRecognition._vosk_rec_instance.Result())
                            if 'text' in result_json and result_json['text']:
                                # Добавляем ucfirst для единообразия с примером
                                recognized_text = result_json['text']
                                if recognized_text:
                                    recognized_text = recognized_text[:1].upper() + recognized_text[1:]
                                await SpeechRecognition.handle_voice_message(recognized_text)
                                logger.info(f"Vosk распознал: {recognized_text}")
                        else:
                            # Обработка частичных результатов (опционально)
                            partial_result_json = json.loads(SpeechRecognition._vosk_rec_instance.PartialResult())
                            if 'partial' in partial_result_json and partial_result_json['partial']:
                                # Включаем логирование частичных результатов для отладки
                                logger.debug(f"Vosk частичный: {partial_result_json['partial']}")
                                pass
                        
                        await asyncio.sleep(SpeechRecognition.VOSK_PROCESS_INTERVAL) # Небольшая задержка для предотвращения перегрузки

                    except Exception as e:
                        logger.error(f"Ошибка при распознавании Vosk в реальном времени: {e}")
                        break
            
            # Получаем окончательный результат после завершения записи
            final_result = SpeechRecognition._vosk_rec_instance.FinalResult()
            final_json = json.loads(final_result)
            if 'text' in final_json and final_json['text']:
                recognized_text = final_json['text']
                if recognized_text:
                    recognized_text = recognized_text[:1].upper() + recognized_text[1:]
                await SpeechRecognition.handle_voice_message(recognized_text)
                logger.info(f"Vosk окончательный: {recognized_text}")


    @staticmethod
    async def async_audio_callback(indata):
        try:
            current_time = time.time()
            # Преобразуем indata в numpy array
            audio_data = np.frombuffer(indata, dtype=np.float32)
            rms = np.sqrt(np.mean(audio_data ** 2))

            async with audio_state.lock:
                if rms > SpeechRecognition.SILENCE_THRESHOLD:
                    audio_state.last_sound_time = current_time
                    if not audio_state.is_recording:
                        logger.debug("🟢 Начало записи")
                        audio_state.is_recording = True
                    await audio_state.add_to_buffer(audio_data)

                elif audio_state.is_recording:
                    silence_duration = 4
                    audio_state.is_recording = False
                    await SpeechRecognition.process_audio()
                else:
                    logger.debug("❌ Слишком короткая запись, сброс")
                    audio_state.audio_buffer.clear()
                    audio_state.is_recording = False

        except Exception as e:
            logger.error(f"Ошибка в колбэке: {str(e)}")

    @staticmethod
    async def process_audio():
        try:
            async with audio_state.lock:
                if not audio_state.audio_buffer:
                    return

                audio_data = np.concatenate(audio_state.audio_buffer)
                audio_state.audio_buffer.clear()

                text = None
                if SpeechRecognition._recognizer_type == "google":
                    with BytesIO() as buffer:
                        sf.write(buffer, audio_data, SpeechRecognition.SAMPLE_RATE, format='WAV')
                        buffer.seek(0)

                        try:
                            recognizer = sr.Recognizer()
                            with sr.AudioFile(buffer) as source:
                                audio = recognizer.record(source)
                                text = recognizer.recognize_google(audio, language="ru-RU")
                                logger.info(f"Google распознал: {text}")
                        except sr.UnknownValueError:
                            logger.warning("Google не распознал речь.")
                        except Exception as e:
                            logger.error(f"Ошибка распознавания Google: {str(e)}")
                elif SpeechRecognition._recognizer_type == "vosk":
                    text = await SpeechRecognition.recognize_vosk(audio_data)

                if text:
                    await SpeechRecognition.handle_voice_message(text)
        except Exception as e:
            logger.error(f"Ошибка обработки: {str(e)}")

    @staticmethod
    async def recognize_speech(audio_buffer):
        # Этот метод используется для распознавания из буфера,
        # который уже является AudioFile-подобным объектом.
        # Для Vosk API нам нужен numpy array.
        # Поэтому, если выбран Vosk, нужно будет преобразовать audio_buffer в numpy array.
        # Или же этот метод будет использоваться только для Google.
        # Пока оставим его для Google, так как он принимает AudioFile.
        # Если потребуется Vosk здесь, нужно будет пересмотреть.
        recognizer = sr.Recognizer()
        text = None

        if SpeechRecognition._recognizer_type == "google":
            try:
                with sr.AudioFile(audio_buffer) as source:
                    audio = recognizer.record(source)

                text = recognizer.recognize_google(audio, language="ru-RU")
                if not text:
                    text = recognizer.recognize_google(audio, language="en-EN")
                return text
            except sr.UnknownValueError:
                logger.error("Google: Не удалось распознать речь")
                return None
            except sr.RequestError as e:
                logger.error(f"Google: Ошибка API: {e}")
                return None
        elif SpeechRecognition._recognizer_type == "vosk":
            # Здесь нужно будет преобразовать audio_buffer в numpy array
            # Это сложнее, так как audio_buffer может быть BytesIO или другим объектом
            # Для простоты, пока этот метод будет работать только с Google
            logger.warning("recognize_speech не поддерживает Vosk напрямую с текущим типом audio_buffer.")
            return None

    @staticmethod
    async def speach_recognition_start_async_other_system():
        while SpeechRecognition.active:
            try:
                await SpeechRecognition.async_audio_callback(0)
                await asyncio.sleep(0.1)  # Уменьшим интервал
            except Exception as e:
                logger.error(f"Ошибка в speach_recognition_start_async_other_system: {e}")

    @staticmethod
    async def speach_recognition_start_async():
        await SpeechRecognition.live_recognition()

    @staticmethod
    def speach_recognition_start(device_id: int, loop):
        SpeechRecognition.microphone_index = device_id
        asyncio.run_coroutine_threadsafe(SpeechRecognition.speach_recognition_start_async(), loop)


    @staticmethod
    async def audio_monitoring():
        try:
            logger.info("🚀 Запуск аудиомониторинга")
            loop = asyncio.get_event_loop()
            with sd.InputStream(
                    callback=lambda indata, *_: asyncio.run_coroutine_threadsafe(SpeechRecognition.async_audio_callback(indata), loop),
                    channels=1,
                    samplerate=SpeechRecognition.SAMPLE_RATE,
                    blocksize=SpeechRecognition.CHUNK_SIZE,
                    device=SpeechRecognition.microphone_index
            ):
                while SpeechRecognition.active:
                    await asyncio.sleep(0.1)
        except Exception as e:
            logger.critical(f"Критическая ошибка: {str(e)}")

    @staticmethod
    async def get_current_text() -> str:
        async with SpeechRecognition._text_lock:
            return SpeechRecognition._current_text.strip()
