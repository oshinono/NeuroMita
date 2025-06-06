# File: NeuroMita/characters.py
import logging
from character import Character # From NeuroMita/character.py
from typing import Dict, Any, Optional
import re
# import multiprocessing # Заменяем на threading для GUI, но Queue пока оставим от multiprocessing
import multiprocessing # Оставляем для Queue, т.к. они process-safe и thread-safe
import threading # Добавляем для запуска GUI в потоке
import importlib 
import os 

logger = logging.getLogger("NeuroMita.Characters")

class CrazyMita(Character):
    DEFAULT_OVERRIDES: Dict[str, Any] = {
        "attitude": 50.0,
        "boredom": 20.0,
        "stress": 8.0,
        "current_fsm_state": "Hello",
    }

    def __init__(self, char_id: str, name: str, silero_command: str, short_name: str,
                 miku_tts_name: str = "Player", silero_turn_off_video: bool = False,
                 initial_vars_override: Dict[str, Any] | None = None):
        super().__init__(char_id, name, silero_command, short_name,
                         miku_tts_name, silero_turn_off_video, initial_vars_override)

        # Только БМ имеет возможность. 
        self.set_variable("playingChess", False) 
        self.chess_gui_thread: Optional[threading.Thread] = None # Изменено с Process на Thread
        self.chess_command_queue: Optional[multiprocessing.Queue] = None # Оставляем multiprocessing.Queue, он thread-safe
        self.chess_state_queue: Optional[multiprocessing.Queue] = None   # Оставляем multiprocessing.Queue
        self.current_chess_elo: Optional[int] = None
        self.elo_mapping: Dict[str, int] = {"easy": 1100, "medium": 1500, "hard": 1900}

        logger.info(f"CrazyMita '{char_id}' fully initialized with overrides and chess attributes.")

    def _start_chess_game_process(self, elo: int, player_is_white: bool = True): # Название метода можно оставить для совместимости
        if self.chess_gui_thread and self.chess_gui_thread.is_alive():
            logger.warning(f"[{self.char_id}] Chess game thread already running. Stopping it first.")
            self._stop_chess_game_process()

        try:
            from Modules.Chess.chess_board import run_chess_gui_process

            self.current_chess_elo = elo
            self.set_variable("playingChess", True) # Устанавливаем флаг игры

            # Инициализируем очереди здесь, если они еще не созданы или для новой игры
            # Важно, чтобы controller внутри GUI потока получал эти очереди
            if self.chess_command_queue is None:
                self.chess_command_queue = multiprocessing.Queue()
            if self.chess_state_queue is None:
                self.chess_state_queue = multiprocessing.Queue()

            logger.info(f"[{self.char_id}] Starting chess GUI in a new thread. ELO: {elo}, Player White: {player_is_white}")
            
            self.chess_gui_thread = threading.Thread(
                target=run_chess_gui_process,
                args=(self.chess_command_queue, self.chess_state_queue, elo, player_is_white),
                daemon=True # Поток-демон завершится, когда завершится основной поток
            )
            self.chess_gui_thread.start()
            
            logger.info(f"[{self.char_id}] Chess GUI thread started.")

            # Убираем отладочный прямой вызов, который передавал None в качестве очередей
            # logger.info(f"[{self.char_id}] DEBUG: Calling run_chess_gui_process directly for debugging.")
            # dummy_command_q = None 
            # dummy_state_q = None   
            # try:
            #     run_chess_gui_process(dummy_command_q, dummy_state_q, elo, player_is_white)
            #     logger.info(f"[{self.char_id}] DEBUG: run_chess_gui_process finished (if it's blocking).")
            # except Exception as e_direct_run:
            #     logger.error(f"[{self.char_id}] DEBUG: Error calling run_chess_gui_process directly: {e_direct_run}", exc_info=True)

        except ImportError as e_imp:
            logger.error(
                f"[{self.char_id}] Failed to import chess_board module from 'Modules.Chess.chess_board'. Chess game cannot start. Error details: {e_imp}",
                exc_info=True
            )
            self._cleanup_chess_resources()
        except AttributeError as e_attr: # Например, если run_chess_gui_process не найдена
            logger.error(f"[{self.char_id}] Attribute error related to chess_board module (e.g., 'run_chess_gui_process' not found). Error: {e_attr}. Chess game cannot start.", exc_info=True)
            self._cleanup_chess_resources()
        except Exception as e:
            logger.error(f"[{self.char_id}] Error starting chess game thread: {e}", exc_info=True)
            self._cleanup_chess_resources()


    def _send_chess_command(self, command_data: Dict[str, Any]):
        if self.get_variable("playingChess", False) and self.chess_command_queue and self.chess_gui_thread and self.chess_gui_thread.is_alive():
            try:
                self.chess_command_queue.put(command_data)
                logger.debug(f"[{self.char_id}] Sent command to chess thread: {command_data}")
            except Exception as e:
                logger.error(f"[{self.char_id}] Error sending command to chess command_queue: {e}")
        elif not self.get_variable("playingChess", False):
            logger.warning(f"[{self.char_id}] Cannot send chess command: game not active.")
        else:
            logger.warning(f"[{self.char_id}] Cannot send chess command: command_queue or thread not available/alive.")


    def _stop_chess_game_process(self, resign: bool = False): # Название метода можно оставить
        if self.get_variable("playingChess", False):
            logger.info(f"[{self.char_id}] Stopping chess game (resign={resign}). Sending command to GUI thread.")
            # Отправляем команду на закрытие GUI процесса/потока через очередь
            # run_chess_gui_process должен обработать "stop_gui_process" или "resign"/"stop"
            # и корректно завершить свой цикл и ресурсы Tkinter.
            if resign:
                self._send_chess_command({"action": "resign"}) # Это приведет к закрытию GUI в run_chess_gui_process
            else:
                # Если просто останавливаем, то GUI должен получить команду "stop_gui_process"
                # или "stop", чтобы знать, что нужно закрыться.
                # "stop_gui_process" - более явная команда для самого GUI потока
                self._send_chess_command({"action": "stop_gui_process"}) 

            if self.chess_gui_thread and self.chess_gui_thread.is_alive():
                logger.info(f"[{self.char_id}] Waiting for chess GUI thread to join...")
                self.chess_gui_thread.join(timeout=10) # Даем потоку время на завершение
                if self.chess_gui_thread.is_alive():
                    logger.warning(f"[{self.char_id}] Chess GUI thread did not terminate gracefully after 10s.")
                    # Для потоков нет метода terminate(). Поток должен сам завершиться.
                    # Если он не завершился, это может указывать на проблему в run_chess_gui_process.
        self._cleanup_chess_resources() # Это установит playingChess в False и обнулит ресурсы

    def _cleanup_chess_resources(self):
        logger.debug(f"[{self.char_id}] Cleaning up chess resources.")
        self.set_variable("playingChess", False) 
        
        # Закрывать очереди multiprocessing.Queue нужно осторожно,
        # особенно если поток-читатель еще может быть жив.
        # Но если поток уже завершен (или мы считаем его таковым), то можно.
        # if self.chess_command_queue:
        #     try:
        #         self.chess_command_queue.close() # Предотвращает добавление новых элементов
        #         # self.chess_command_queue.join_thread() # Ожидает, пока все элементы из буфера не будут обработаны
        #     except Exception as e:
        #         logger.warning(f"[{self.char_id}] Error closing command_queue: {e}")
        # if self.chess_state_queue:
        #     try:
        #         self.chess_state_queue.close()
        #         # self.chess_state_queue.join_thread()
        #     except Exception as e:
        #         logger.warning(f"[{self.char_id}] Error closing state_queue: {e}")

        # Просто обнуляем ссылки. Потоки-демоны и сборщик мусора должны справиться.
        # Или, если очереди создаются заново каждый раз, это тоже решает проблему.
        self.chess_gui_thread = None
        self.chess_command_queue = None # Можно пересоздавать их в _start_chess_game_process
        self.chess_state_queue = None
        self.current_chess_elo = None


    def process_response_nlp_commands(self, response: str) -> str:
        response = super().process_response_nlp_commands(response)

        # --- Chess Game Tags ---
        start_match = re.search(r"<StartChessGame>(.*?)</StartChessGame>", response, re.DOTALL)
        if start_match:
            difficulty_str = start_match.group(1).strip().lower()
            elo = self.elo_mapping.get(difficulty_str, self.elo_mapping["medium"])
            # Важно: playingChess установится в True внутри _start_chess_game_process
            self._start_chess_game_process(elo=elo, player_is_white=True) 
            response = response.replace(start_match.group(0), "", 1).strip()
            logger.info(f"[{self.char_id}] Chess game start requested with difficulty '{difficulty_str}' (ELO: {elo}).")

        change_diff_match = re.search(r"<ChangeChessDifficulty>(.*?)</ChangeChessDifficulty>", response, re.DOTALL)
        if change_diff_match:
            if self.get_variable("playingChess", False):
                difficulty_str = change_diff_match.group(1).strip().lower()
                new_elo = self.elo_mapping.get(difficulty_str)
                if new_elo:
                    # self.current_chess_elo = new_elo # ELO изменится внутри контроллера после команды
                    self._send_chess_command({"action": "change_elo", "elo": new_elo})
                    logger.info(f"[{self.char_id}] Requested chess difficulty change to '{difficulty_str}' (ELO: {new_elo}).")
                else:
                    logger.warning(f"[{self.char_id}] Invalid difficulty for ChangeChessDifficulty: {difficulty_str}")
            else:
                logger.warning(f"[{self.char_id}] Received ChangeChessDifficulty but no game is active.")
            response = response.replace(change_diff_match.group(0), "", 1).strip()

        if "<RequestBestChessMove!>" in response:
            if self.get_variable("playingChess", False):
                self._send_chess_command({"action": "engine_move"})
                logger.info(f"[{self.char_id}] Requested best chess move from Maia engine.")
            else:
                logger.warning(f"[{self.char_id}] Received RequestBestChessMove but no game is active.")
            response = response.replace("<RequestBestChessMove!>", "", 1).strip()

        llm_move_match = re.search(r"<MakeChessMoveAsLLM>(.*?)</MakeChessMoveAsLLM>", response, re.DOTALL)
        if llm_move_match:
            if self.get_variable("playingChess", False):
                uci_move = llm_move_match.group(1).strip().lower()
                if uci_move:
                    self._send_chess_command({"action": "force_engine_move", "move": uci_move})
                    logger.info(f"[{self.char_id}] LLM specified chess move: {uci_move}.")
                else:
                    logger.warning(f"[{self.char_id}] LLM specified empty chess move.")
            else:
                logger.warning(f"[{self.char_id}] Received MakeChessMoveAsLLM but no game is active.")
            response = response.replace(llm_move_match.group(0), "", 1).strip()

        if "<ResignChessGame!>" in response:
            if self.get_variable("playingChess", False):
                logger.info(f"[{self.char_id}] LLM resigns the chess game. Stopping process...")
                self._stop_chess_game_process(resign=True) # Это вызовет cleanup и установит playingChess=False
            else:
                logger.warning(f"[{self.char_id}] Received ResignChessGame but no game is active.")
            response = response.replace("<ResignChessGame!>", "", 1).strip()

        if "<StopChessGame!>" in response:
            if self.get_variable("playingChess", False):
                logger.info(f"[{self.char_id}] LLM stops the chess game. Stopping process...")
                self._stop_chess_game_process(resign=False) # Это вызовет cleanup и установит playingChess=False
            else:
                logger.debug(f"[{self.char_id}] Received StopChessGame but no game was active. Tag removed.")
            response = response.replace("<StopChessGame!>", "", 1).strip()


        if "<Secret!>" in response:
            if not self.get_variable("secretExposedFirst", False):
                self.set_variable("secretExposed", True)
                logger.info(f"[{self.char_id}] Secret revealed via <Secret!> tag. Attitude/boredom may be set by DSL or <p>.")
            response = response.replace("<Secret!>", "").strip()
        return response

class KindMita(Character):
    DEFAULT_OVERRIDES: Dict[str, Any] = {
        "attitude": 90.0,
        "stress": 0.0,
        "current_fsm_state": "Default",
    }

class ShortHairMita(Character):
    DEFAULT_OVERRIDES: Dict[str, Any] = {
        "attitude": 70.0,
        "boredom": 15.0,
        "stress": 10.0,
        "current_fsm_state": "Default",
    }

class CappyMita(Character):
    DEFAULT_OVERRIDES: Dict[str, Any] = {
        "boredom": 25.0,
        "current_fsm_state": "Default",
    }

class MilaMita(Character):
    DEFAULT_OVERRIDES: Dict[str, Any] = {
        "attitude": 75.0,
        "current_fsm_state": "Default",
    }

class CreepyMita(Character):
    DEFAULT_OVERRIDES: Dict[str, Any] = {
        "attitude": 40.0,
        "stress": 30.0,
        "current_fsm_state": "Default",
    }

class SleepyMita(Character):
    DEFAULT_OVERRIDES: Dict[str, Any] = {
        "boredom": 40.0,
        "current_fsm_state": "Sleeping",
    }

class SpaceCartridge(Character):
    DEFAULT_OVERRIDES: Dict[str, Any] = {"attitude": 50.0, "current_fsm_state": "Space"}
    # def __init__(self, char_id: str, name: str, silero_command: str, short_name: str,
    #              miku_tts_name: str = "Player", silero_turn_off_video: bool = False,
    #              initial_vars_override: Dict[str, Any] | None = None):
    #     # Cartridges might not need all the same params, adjust as necessary
    #     # Or, they are instantiated with default names if not interactive in the same way.
    #     # For now, keeping constructor consistent.
    #     super().__init__(char_id, name, silero_command, short_name,
    #                      miku_tts_name, silero_turn_off_video, initial_vars_override)

class DivanCartridge(Character):
    DEFAULT_OVERRIDES: Dict[str, Any] = {"attitude": 50.0, "current_fsm_state": "Divan"}
    # def __init__(self, char_id: str, name: str, silero_command: str, short_name: str, 
    #              miku_tts_name: str = "Player", silero_turn_off_video: bool = False,
    #              initial_vars_override: Dict[str, Any] | None = None):
    #     super().__init__(char_id, name, silero_command, short_name,
    #                      miku_tts_name, silero_turn_off_video, initial_vars_override)

class GameMaster(Character):
    DEFAULT_OVERRIDES: Dict[str, Any] = {"attitude": 100.0, "boredom": 0.0, "stress": 0.0}

    def _process_behavior_changes_from_llm(self, response: str) -> str:
        logger.debug(f"[{self.char_id}] GameMaster is not processing <p> tags for self.")
        response = re.sub(r"<p>.*?</p>", "", response).strip()
        return response

    def get_llm_system_prompt_string(self) -> str:
        try:
            from SettingsManager import SettingsManager as settings
            current_instruction = settings.get("GM_SMALL_PROMPT", "")
            self.set_variable("GM_INSTRUCTION", current_instruction)
        except ImportError:
            logger.warning(f"[{self.char_id}] SettingsManager not found for GameMaster's GM_INSTRUCTION.")
            self.set_variable("GM_INSTRUCTION", "")
        except Exception as e:
            logger.error(f"[{self.char_id}] Error accessing GM_SMALL_PROMPT: {e}")
            self.set_variable("GM_INSTRUCTION", "")
        return super().get_llm_system_prompt_string()