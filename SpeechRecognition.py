import time
from io import BytesIO

import soundfile as sf
import numpy as np
import speech_recognition as sr


import asyncio


class AudioState:
    def __init__(self):
        self.is_recording = False
        self.audio_buffer = []
        self.last_sound_time = None
        self.is_playing = False
        self.lock = asyncio.Lock()
        self.vc = None
        self.max_buffer_size = 50  # Максимум 50 фреймов в буфере

    async def add_to_buffer(self, data):
        async with self.lock:
            if len(self.audio_buffer) >= self.max_buffer_size:
                self.audio_buffer.pop(0)  # Удаляем старые данные
            self.audio_buffer.append(data.copy())


class SpeechRecognition:
    user_input = ""
    microphone_index = 0
    active = True
    audio_state = AudioState()

    SAMPLE_RATE = 32000
    CHUNK_SIZE = 256
    TIMEOUT_MESSAGE = False
    SILENCE_THRESHOLD = 0.01
    SILENCE_DURATION = 2

    @staticmethod
    def receive_text():
        user_input_to_send = SpeechRecognition.user_input
        SpeechRecognition.user_input = ""
        return user_input_to_send

    @staticmethod
    def list_microphones():
        """Возвращает список доступных микрофонов"""
        return sr.Microphone.list_microphone_names()

    @staticmethod
    async def handle_voice_message(recognized_text: str) -> None:
        print(f"Распознанный текст: {recognized_text}")
        SpeechRecognition.user_input += f" {recognized_text}"

    @staticmethod
    async def live_recognition() -> None:
        """Потоковое распознавание с выбранного микрофона"""
        recognizer = sr.Recognizer()

        # Используем выбранный микрофон
        with sr.Microphone(device_index=SpeechRecognition.microphone_index, sample_rate=SpeechRecognition.SAMPLE_RATE,
                           chunk_size=SpeechRecognition.CHUNK_SIZE) as source:
            print(f"Используется микрофон: {sr.Microphone.list_microphone_names()[SpeechRecognition.microphone_index]}")
            recognizer.adjust_for_ambient_noise(source)
            print("Скажите что-нибудь...")

            while True:
                try:
                    if not SpeechRecognition.active:
                        return

                    audio = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: recognizer.listen(source)
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
                        print("Таймаут ожидания речи...")
                except sr.UnknownValueError:
                    ...
                    # print("Речь не распознана")
                except Exception as e:
                    print(f"Ошибка при распознавании: {e}")
                    break

    #region Попытка2
    @staticmethod
    async def async_audio_callback(indata):

        async with SpeechRecognition.audio_state.lock:

            if SpeechRecognition.audio_state.is_playing:
                return

            current_volume = np.max(np.abs(indata))
            #print(f"Громкость: {current_volume}")  # Логирование громкости

            if current_volume > SpeechRecognition.SILENCE_THRESHOLD:
                SpeechRecognition.audio_state.audio_buffer.append(indata.copy())
                SpeechRecognition.audio_state.last_sound_time = time.time()
                SpeechRecognition.audio_state.is_recording = True
                print("Данные добавлены в буфер.")
            elif SpeechRecognition.audio_state.is_recording:
                silence_time = time.time() - SpeechRecognition.audio_state.last_sound_time
                if silence_time > SpeechRecognition.SILENCE_DURATION:
                    SpeechRecognition.audio_state.is_recording = False
                    print("Запись завершена: тишина.")
                    await SpeechRecognition.process_audio()  # Запуск обработки сразу после тишины

    @staticmethod
    async def process_audio():
        try:
            async with SpeechRecognition.audio_state.lock:
                if len(SpeechRecognition.audio_state.audio_buffer) == 0:
                    print("Аудиобуфер пуст!")
                    return
                # Проверка данных перед конкатенацией
                non_empty_chunks = [chunk for chunk in SpeechRecognition.audio_state.audio_buffer if chunk.size > 0]
                if not non_empty_chunks:
                    print("Нет данных для обработки.")
                    return
                audio_data = np.concatenate(non_empty_chunks)
                SpeechRecognition.audio_state.audio_buffer.clear()

                with BytesIO() as audio_buffer:
                    sf.write(audio_buffer, audio_data, SpeechRecognition.SAMPLE_RATE, format='WAV')
                    audio_buffer.seek(0)
                    recognized_text = SpeechRecognition.recognize_speech(audio_buffer)

                    if recognized_text:
                        await SpeechRecognition.handle_voice_message(recognized_text)
        except Exception as e:
            print(f"Ошибка обработки аудио: {e}")

    @staticmethod
    def recognize_speech(audio_buffer):
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_buffer) as source:
                audio = recognizer.record(source)

            text = recognizer.recognize_google(audio, language="ru-RU")
            if not text:
                text = recognizer.recognize_google(audio, language="en-EN")
            return text
        except sr.UnknownValueError:
            print("Не удалось распознать речь")
            return None
        except sr.RequestError as e:
            print(f"Ошибка API: {e}")
            return None

    #print(f"Размер audio_buffer: {len(SpeechRecognition.audio_state.audio_buffer)}")
    @staticmethod
    async def speach_recognition_start_async_other_system():
        while True:
            #SpeechRecognition.audio_state.is_playing = True
            try:

                if not SpeechRecognition.active:
                    return

                await SpeechRecognition.async_audio_callback(4)

                #await asyncio.sleep(0.25)
            except:
                ...

    #endregion

    @staticmethod
    async def speach_recognition_start_async():
        """Асинхронный запуск распознавания с указанным микрофоном"""
        await SpeechRecognition.live_recognition()
        #await  SpeechRecognition.speach_recognition_start_async_other_system()

    @staticmethod
    def speach_recognition_start(device_id: int, loop):
        SpeechRecognition.microphone_index = device_id
        """Синхронный запуск распознавания с указанным микрофоном"""
        asyncio.run_coroutine_threadsafe(SpeechRecognition.speach_recognition_start_async(), loop)
