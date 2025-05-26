import mss
import mss.tools
from PIL import Image
import numpy as np
import time
import threading
from Logger import logger

class ScreenCapture:
    def __init__(self):
        self._running = False
        self._thread = None
        self._latest_frame = None # Оставляем для совместимости, но будем использовать _frame_history
        self._frame_history = [] # Список для хранения последних кадров
        self._max_history_frames = 1 # Максимальное количество кадров в истории
        self._quality = 25  # По умолчанию 25%
        self._fps = 1  # По умолчанию 1 кадр в секунду
        self._interval_seconds = 1.0  # По умолчанию 1 кадр в секунду
        self._sct = None  # Инициализируем здесь как None, чтобы создавать в потоке

    def start_capture(self, interval_seconds: float = 1.0, quality: int = 25, fps: int = 1, max_history_frames: int = 1):
        if self._running:
            logger.warning("Захват экрана уже запущен.")
            return

        self._quality = max(1, min(quality, 100))  # Ограничение качества от 1 до 100
        self._fps = max(1, fps)  # Минимальный FPS = 1
        self._interval_seconds = 1.0 / self._fps
        self._interval_seconds = max(0.1, self._interval_seconds)  # Минимальный интервал 0.1 секунды
        self._max_history_frames = max(1, max_history_frames) # Минимум 1 кадр в истории

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Захват экрана запущен с качеством {self._quality}, {self._fps} FPS (интервал {self._interval_seconds} секунд).")

    def stop_capture(self):
        if not self._running:
            logger.warning("Захват экрана не запущен.")
            return

        self._running = False
        if self._thread:
            self._thread.join()  # Ждем завершения потока
        logger.info("Захват экрана остановлен.")

    def _capture_loop(self):
        # Инициализируем mss.mss() внутри потока, где он будет использоваться
        # Это решает проблему 'srcdc' object has no attribute
        with mss.mss() as sct:
            while self._running:
                try:
                    # Захват всего экрана
                    sct_img = sct.grab(sct.monitors[0])  # monitors[0] - основной монитор

                    # Конвертация в PIL Image
                    img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)

                    # Сжатие изображения (уменьшение размера)
                    max_size = (1024, 768)  # Максимальное разрешение, можно сделать настраиваемым
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)  # Использование LANCZOS для лучшего качества

                    # Сохраняем в байтовый буфер в формате JPEG для лучшего сжатия
                    from io import BytesIO
                    byte_arr = BytesIO()
                    img.save(byte_arr, format='JPEG', quality=self._quality)  # Качество JPEG
                    current_frame_bytes = byte_arr.getvalue()

                    # Добавляем кадр в историю, поддерживая лимит
                    self._frame_history.append(current_frame_bytes)
                    if len(self._frame_history) > self._max_history_frames:
                        self._frame_history.pop(0) # Удаляем самый старый кадр

                    # Обновляем _latest_frame для совместимости, если нужно
                    self._latest_frame = current_frame_bytes

                except Exception as e:
                    logger.error(f"Ошибка при захвате экрана: {e}")
                    # При ошибке не добавляем кадр, но не сбрасываем историю

                time.sleep(self._interval_seconds)

    def get_latest_frame(self) -> bytes | None:
        """Возвращает последний захваченный кадр в формате JPEG байтов (для совместимости)."""
        if self._frame_history:
            return self._frame_history[-1]
        return None

    def get_recent_frames(self, limit: int) -> list[bytes]:
        """Возвращает список последних захваченных кадров в формате JPEG байтов."""
        # Возвращаем не более limit кадров из истории, начиная с самого старого из запрошенных
        return self._frame_history[max(0, len(self._frame_history) - limit):]

    def is_running(self) -> bool:
        return self._running

if __name__ == "__main__":
    # Пример использования
    logger.info("Запуск примера ScreenCapture...")
    capture = ScreenCapture()
    capture.start_capture(interval_seconds=2.0)  # Захват каждые 2 секунды

    try:
        for i in range(10):
            frame = capture.get_latest_frame()
            if frame:
                logger.info(f"Получен кадр размером: {len(frame)} байт")
                # Здесь можно было бы сохранить кадр в файл для проверки
                # with open(f"screenshot_{i}.png", "wb") as f:
                #     f.write(frame)
            else:
                logger.info("Кадр пока не готов или произошла ошибка.")
            time.sleep(1)  # Ждем, чтобы не запрашивать слишком часто

    except KeyboardInterrupt:
        logger.info("Пример остановлен пользователем.")
    finally:
        capture.stop_capture()
        logger.info("Захват экрана завершен.")
