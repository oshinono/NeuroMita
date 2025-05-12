# File: chat_model.py
import concurrent.futures
import time
import requests
import tiktoken # Keep for token counting if used by calculate_cost_for_combined_messages
from openai import OpenAI
import re
import importlib
from typing import List, Dict, Any

from Logger import logger
from characters import CrazyMita, KindMita, ShortHairMita, CappyMita, MilaMita, CreepyMita, SleepyMita, GameMaster, SpaceCartridge, DivanCartridge # Updated imports
from character import Character # Character base
from utils.PipInstaller import PipInstaller
from utils import SH, save_combined_messages, calculate_cost_for_combined_messages, replace_numbers_with_words # Keep utils
# from promptPart import PromptPart, PromptType # No longer needed


class ChatModel:
    def __init__(self, gui, api_key, api_key_res, api_url, api_model, api_make_request, pip_installer: PipInstaller):
        self.last_key = 0
        self.gui = gui
        self.pip_installer = pip_installer
        self.g4fClient = None
        self.g4f_available = False
        self._initialize_g4f() # Keep g4f initialization

        self.api_key = api_key
        self.api_key_res = api_key_res
        self.api_url = api_url
        self.api_model = api_model
        self.gpt4free_model = self.gui.settings.get("gpt4free_model")
        self.makeRequest = api_make_request # This seems to be a boolean flag

        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
            logger.info("OpenAI client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None

        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o-mini") # Or your preferred model
            self.hasTokenizer = True
        except Exception:
            logger.info("Tiktokenizer failed to initialize. Token counting might be affected.")
            self.hasTokenizer = False

        self.max_response_tokens = int(self.gui.settings.get("MODEL_MAX_RESPONSE_TOKENS", 3200))
        self.temperature = float(self.gui.settings.get("MODEL_TEMPERATURE", 0.5))
        self.presence_penalty = float(self.gui.settings.get("MODEL_PRESENCE_PENALTY", 0.0))
        
        # Cost calculation variables (if used)
        self.cost_input_per_1000 = 0.0432 
        self.cost_response_per_1000 = 0.1728

        self.memory_limit = int(self.gui.settings.get("MODEL_MESSAGE_LIMIT", 40)) # For historical messages

        self.current_character: Character = None
        self.current_character_to_change = str(self.gui.settings.get("CHARACTER"))
        self.characters: Dict[str, Character] = {}

        # Game-specific state - these should ideally be passed to character or managed elsewhere if possible
        # For now, keeping them here as per original. DSL might need them injected into character.variables.
        self.distance = 0.0
        self.roomPlayer = -1
        self.roomMita = -1
        self.nearObjects = ""
        self.actualInfo = ""
        
        self.infos_to_add_to_history: List[Dict] = [] # For temporary system messages to be added to history

        self.init_characters()
        self.HideAiData = True # Unused?
        self.max_request_attempts = int(self.gui.settings.get("MODEL_MESSAGE_ATTEMPTS_COUNT", 5))
        self.request_delay = float(self.gui.settings.get("MODEL_MESSAGE_ATTEMPTS_TIME", 0.20))

    def _initialize_g4f(self):
        # ... (keep existing _initialize_g4f method from original file) ...
        logger.info("Проверка и инициализация g4f (после возможного обновления при запуске)...")
        try:
            from g4f.client import Client as g4fClient
            logger.info("g4f найден (при проверке), попытка инициализации клиента...")
            try:
                self.g4fClient = g4fClient()
                self.g4f_available = True
                logger.info("g4fClient успешно инициализирован.")
            except Exception as e:
                logger.error(f"Ошибка при инициализации g4fClient: {e}")
                self.g4fClient = None
                self.g4f_available = False
        except ImportError:
            logger.info("Модуль g4f не найден (при проверке). Попытка первоначальной установки...")

            target_version = self.gui.settings.get("G4F_VERSION", "4.7.7") # Ensure settings has G4F_VERSION
            package_spec = f"g4f=={target_version}" if target_version != "latest" else "g4f"

            if self.pip_installer:
                success = self.pip_installer.install_package(
                    package_spec,
                    description=f"Первоначальная установка g4f версии {target_version}..."
                )
                if success:
                    logger.info("Первоначальная установка g4f (файлы) прошла успешно. Очистка кэша импорта...")
                    try:
                        importlib.invalidate_caches()
                        logger.info("Кэш импорта очищен.")
                    except Exception as e_invalidate:
                        logger.error(f"Ошибка при очистке кэша импорта: {e_invalidate}")

                    logger.info("Повторная попытка импорта и инициализации...")
                    try:
                        from g4f.client import Client as g4fClient # Re-import
                        logger.info("Повторный импорт g4f успешен. Попытка инициализации клиента...")
                        try:
                            self.g4fClient = g4fClient()
                            self.g4f_available = True
                            logger.info("g4fClient успешно инициализирован после установки.")
                        except Exception as e_init_after_install: # More specific exception name
                            logger.error(f"Ошибка при инициализации g4fClient после установки: {e_init_after_install}")
                            self.g4fClient = None
                            self.g4f_available = False
                    except ImportError:
                        logger.error("Не удалось импортировать g4f даже после успешной установки и очистки кэша.")
                        self.g4fClient = None
                        self.g4f_available = False
                    except Exception as e_import_after:
                        logger.error(f"Непредвиденная ошибка при повторном импорте/инициализации g4f: {e_import_after}")
                        self.g4fClient = None
                        self.g4f_available = False
                else:
                    logger.error("Первоначальная установка g4f не удалась (ошибка pip).")
                    self.g4fClient = None
                    self.g4f_available = False
            else:
                logger.error("Экземпляр PipInstaller не передан в ChatModel, установка g4f невозможна.")
                self.g4fClient = None
                self.g4f_available = False
        except Exception as e_initial:
            logger.error(f"Непредвиденная ошибка при первичной инициализации g4f: {e_initial}")
            self.g4fClient = None
            self.g4f_available = False


    def init_characters(self):
        # Character instantiation remains largely the same
        self.crazy_mita_character = CrazyMita("Crazy", "Crazy Mita", "/speaker mita", short_name="CrazyMita", miku_tts_name="/set_person CrazyMita", silero_turn_off_video=True)
        self.kind_mita_character = KindMita("Kind", "Kind Mita", "/speaker kind", short_name="MitaKind", miku_tts_name="/set_person KindMita", silero_turn_off_video=True)
        self.cappy_mita_character = CappyMita("Cappy","Cappy Mita", "/speaker cap", short_name="CappieMita", miku_tts_name="/set_person CapMita", silero_turn_off_video=True)
        self.shorthair_mita_character = ShortHairMita("ShortHair","ShortHair Mita", "/speaker shorthair", short_name="ShorthairMita", miku_tts_name="/set_person ShortHairMita", silero_turn_off_video=True)
        self.mila_character = MilaMita("Mila","Mila", "/speaker mila", short_name="Mila", miku_tts_name="/set_person MilaMita", silero_turn_off_video=True)
        self.sleepy_character = SleepyMita("Sleepy","Sleepy Mita", "/speaker dream", short_name="SleepyMita", miku_tts_name="/set_person SleepyMita", silero_turn_off_video=True)
        self.creepy_character = CreepyMita("Creepy","Creepy Mita", "/speaker ghost", short_name="GhostMita", miku_tts_name="/set_person GhostMita", silero_turn_off_video=True)
        
        self.cart_space = SpaceCartridge("Cart_portal", "Cart_portal", "/speaker wheatley", short_name="Player", miku_tts_name="/set_person Player", silero_turn_off_video=True) # Corrected silero speaker
        self.cart_divan = DivanCartridge("Cart_divan", "Cart_divan", "/speaker engineer", short_name="Player", miku_tts_name="/set_person Player", silero_turn_off_video=True)
        self.GameMaster = GameMaster("GameMaster", "GameMaster", "/speaker dryad", short_name="PhoneMita", miku_tts_name="/set_person PhoneMita", silero_turn_off_video=True)

        self.characters = {
            self.crazy_mita_character.name: self.crazy_mita_character,
            self.kind_mita_character.name: self.kind_mita_character,
            self.cappy_mita_character.name: self.cappy_mita_character,
            self.shorthair_mita_character.name: self.shorthair_mita_character,
            self.mila_character.name: self.mila_character,
            self.sleepy_character.name: self.sleepy_character,
            self.creepy_character.name: self.creepy_character,
            self.cart_space.name: self.cart_space,
            self.cart_divan.name: self.cart_divan,
            self.GameMaster.name: self.GameMaster,
        }
        self.current_character = self.characters.get(self.current_character_to_change) or self.crazy_mita_character


    def get_all_mitas(self):
        logger.info(f"Available characters: {list(self.characters.keys())}")
        return list(self.characters.keys())

    def update_openai_client(self, reserve_key_token=None): # Parameter name more descriptive
        logger.info("Attempting to update OpenAI client.")
        key_to_use = reserve_key_token if reserve_key_token else self.api_key

        if not key_to_use:
            logger.error("No API key available to update OpenAI client.")
            self.client = None
            return

        try:
            if self.api_url:
                logger.info(f"Using API key (masked): {SH(key_to_use)} and base URL: {self.api_url}")
                self.client = OpenAI(api_key=key_to_use, base_url=self.api_url)
            else:
                logger.info(f"Using API key (masked): {SH(key_to_use)} (no custom base URL)")
                self.client = OpenAI(api_key=key_to_use)
            logger.info("OpenAI client updated successfully.")
        except Exception as e:
            logger.error(f"Failed to update OpenAI client: {e}")
            self.client = None


    def generate_response(self, user_input: str, system_input: str = "") -> str:
        self.check_change_current_character()

        history_data = self.current_character.history_manager.load_history()
        llm_messages_history = history_data.get("messages", [])

        if self.infos_to_add_to_history:
            llm_messages_history.extend(self.infos_to_add_to_history)
            self.infos_to_add_to_history.clear()
            
        self.current_character.variables["GAME_DISTANCE"] = self.distance
        self.current_character.variables["GAME_ROOM_PLAYER"] = self.roomPlayer
        self.current_character.variables["GAME_ROOM_MITA"] = self.roomMita
        self.current_character.variables["GAME_NEAR_OBJECTS"] = self.nearObjects
        self.current_character.variables["GAME_ACTUAL_INFO"] = self.actualInfo
        
        combined_messages = self.current_character.get_full_system_setup_for_llm()

        if self.current_character != self.GameMaster:
            llm_messages_history_limited = llm_messages_history[-self.memory_limit:]
        else:
            llm_messages_history_limited = llm_messages_history[-8:]
        
        combined_messages.extend(llm_messages_history_limited) # Use limited history for context

        user_message_for_history = None
        if system_input:
            combined_messages.append({"role": "system", "content": system_input})
        if user_input:
            user_message_for_history = {"role": "user", "content": user_input}
            combined_messages.append(user_message_for_history)
        
        try:
            llm_response_content, success = self._generate_chat_response(combined_messages)

            if not success or not llm_response_content:
                logger.warning("LLM generation failed or returned empty.")
                return "..." 

            assistant_message = {"role": "assistant", "content": llm_response_content}
            
            processed_response_text = self.current_character.process_response_nlp_commands(llm_response_content)

            if user_message_for_history: 
                llm_messages_history.append(user_message_for_history)
            llm_messages_history.append(assistant_message) 
            
            self.current_character.save_character_state_to_history(llm_messages_history)

            if self.current_character != self.GameMaster or bool(self.gui.settings.get("GM_VOICE")):
                self.gui.textToTalk = self.process_text_to_voice(processed_response_text)
                self.gui.textSpeaker = self.current_character.silero_command
                self.gui.textSpeakerMiku = self.current_character.miku_tts_name
                self.gui.silero_turn_off_video = self.current_character.silero_turn_off_video
                logger.info(f"TTS Text: {self.gui.textToTalk}, Speaker: {self.gui.textSpeaker}")
            
            self.gui.update_debug_info() 
            return processed_response_text

        except Exception as e:
            logger.error(f"Error during LLM response generation or processing: {e}", exc_info=True)
            return f"Ошибка: {e}"


    def check_change_current_character(self):
        if not self.current_character_to_change:
            return
        if self.current_character_to_change in self.characters:
            if not self.current_character or self.current_character.name != self.current_character_to_change:
                logger.info(f"Changing character to {self.current_character_to_change}")
                self.current_character = self.characters[self.current_character_to_change]
                # self.current_character.load_character_state_from_history() # Ensure new char's state is loaded
                # This load is implicitly handled when character is first accessed or if generate_response loads it.
                # Best to ensure it's loaded upon switch.
                self.current_character.reload_character_data() # Reloads history and resets vars for the new char
            self.current_character_to_change = ""
        else:
            logger.warning(f"Attempted to change to unknown character: {self.current_character_to_change}")
            self.current_character_to_change = ""


    def _generate_chat_response(self, combined_messages):
        # ... (keep existing _generate_chat_response method from original file) ...
        # This method handles retries, different API providers (OpenAI, g4f, custom request)
        # Ensure it uses self.client (for OpenAI) and self.g4fClient (for g4f)
        # and correctly formats messages for Gemini if GEMINI_CASE is true.
        max_attempts = self.max_request_attempts
        retry_delay = self.request_delay
        request_timeout = 45 # seconds

        self._log_generation_start()
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Generation attempt {attempt}/{max_attempts}")
            response_text = None # Changed from response to avoid conflict

            # Log messages being sent (be careful with sensitive data in logs)
            save_combined_messages(combined_messages, f"Attempt_{attempt}") # Util function

            try:
                if bool(self.gui.settings.get("NM_API_REQ", False)): # Custom request
                    formatted_for_request = combined_messages
                    if bool(self.gui.settings.get("GEMINI_CASE", False)):
                        formatted_for_request = self._format_messages_for_gemini(combined_messages)
                    
                    response_text = self._execute_with_timeout(
                        self._generate_request_response, # This calls generate_request_gemini or common
                        args=(formatted_for_request,),
                        timeout=request_timeout
                    )
                else: # OpenAI or g4f
                    use_gpt4free_for_this_attempt = bool(self.gui.settings.get("gpt4free")) or \
                                                 (bool(self.gui.settings.get("GPT4FREE_LAST_ATTEMPT")) and attempt >= max_attempts)
                    
                    if use_gpt4free_for_this_attempt:
                        logger.info("Using gpt4free for this attempt.")
                    elif attempt > 1 and self.api_key_res: # Try reserve key if main fails
                        logger.info("Attempting with reserve API key.")
                        self.update_openai_client(reserve_key_token=self.GetOtherKey()) # Use GetOtherKey logic
                    # else: use current self.client (already set with main key or updated)

                    response_text = self._generate_openapi_response(combined_messages, use_gpt4free=use_gpt4free_for_this_attempt)

                if response_text:
                    cleaned_response = self._clean_response(response_text)
                    logger.info(f"Successful response received (attempt {attempt}).")
                    if cleaned_response: # Ensure not empty after cleaning
                        return cleaned_response, True
                    else:
                        logger.warning("Response became empty after cleaning.")
                else:
                    logger.warning(f"Attempt {attempt} yielded no response or an error handled within generation.")

            except concurrent.futures.TimeoutError:
                logger.error(f"Attempt {attempt} timed out after {request_timeout}s.")
            except Exception as e:
                logger.error(f"Error during generation attempt {attempt}: {str(e)}", exc_info=True)

            if attempt < max_attempts:
                logger.info(f"Waiting {retry_delay}s before next attempt...")
                time.sleep(retry_delay)
        
        logger.error("All generation attempts failed.")
        return None, False


    def _execute_with_timeout(self, func, args=(), kwargs={}, timeout=30):
        # ... (keep existing _execute_with_timeout method) ...
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor: # Ensure single thread for sequential if needed
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {timeout} seconds.")
                raise # Re-raise TimeoutError to be caught by _generate_chat_response
            except Exception as e:
                logger.error(f"Exception in function {func.__name__} executed with timeout: {e}")
                raise # Re-raise other exceptions


    def _log_generation_start(self):
        # ... (keep existing _log_generation_start method) ...
        logger.info("Preparing to generate LLM response.")
        # Log current settings that affect generation
        logger.info(f"Max Response Tokens: {self.max_response_tokens}, Temperature: {self.temperature}")
        logger.info(f"Presence Penalty: {self.presence_penalty} (Used: {bool(self.gui.settings.get('USE_MODEL_PRESENCE_PENALTY'))})")
        logger.info(f"API URL: {self.api_url}, API Model: {self.api_model}")
        logger.info(f"g4f Enabled: {bool(self.gui.settings.get('gpt4free'))}, g4f Model: {self.gpt4free_model}")
        logger.info(f"Custom Request (NM_API_REQ): {bool(self.gui.settings.get('NM_API_REQ', False))}")
        if bool(self.gui.settings.get('NM_API_REQ', False)):
            logger.info(f"  Custom Request Model (NM_API_MODEL): {self.gui.settings.get('NM_API_MODEL')}")
            logger.info(f"  Gemini Case for Custom Req: {bool(self.gui.settings.get('GEMINI_CASE', False))}")


    def _format_messages_for_gemini(self, combined_messages):
        # ... (keep existing _format_messages_for_gemini method) ...
        formatted_messages = []
        # Gemini prefers alternating user/model roles, and no system role.
        # Convert system messages to user messages with a prefix.
        # Ensure the conversation starts with a user role if possible, or handle appropriately.
        
        # Simple conversion: prefix system messages
        for i, msg in enumerate(combined_messages):
            if msg["role"] == "system":
                # If it's the very first message and it's system, Gemini might prefer it as 'user'
                # Or if the previous was 'model', the next system should be 'user'
                # This needs careful handling based on Gemini's exact requirements for conversation structure.
                # A common pattern: user, model, user, model ...
                # For now, a direct conversion:
                formatted_messages.append({"role": "user", "content": f"[System Instruction]: {msg['content']}"})
            elif msg["role"] == "assistant":
                 formatted_messages.append({"role": "model", "content": msg['content']})
            else: # user
                formatted_messages.append(msg)
        
        # Ensure roles alternate if strictly required by the Gemini API endpoint.
        # This might involve merging consecutive 'user' messages if system messages were converted.
        # For simplicity now, this basic conversion is assumed.
        # save_combined_messages(formatted_messages, "ForGemini") # For debugging
        return formatted_messages


    def _generate_request_response(self, formatted_messages):
        # ... (keep existing _generate_request_response method) ...
        # This method dispatches to generate_request_gemini or generate_request_common
        try:
            if bool(self.gui.settings.get("GEMINI_CASE", False)):
                logger.info("Dispatching to Gemini request generation.")
                return self.generate_request_gemini(formatted_messages)
            else:
                logger.info("Dispatching to common request generation.")
                return self.generate_request_common(formatted_messages)
        except Exception as e:
            logger.error(f"Error in _generate_request_response dispatcher: {str(e)}", exc_info=True)
            return None


    def _generate_openapi_response(self, combined_messages, use_gpt4free=False):
        # ... (keep existing _generate_openapi_response method) ...
        # This method handles calls to OpenAI compatible APIs (including self.client or self.g4fClient)
        target_client = None
        model_to_use = ""

        if use_gpt4free:
            if not self.g4f_available or not self.g4fClient:
                logger.error("gpt4free selected, but client is not available.")
                return None
            target_client = self.g4fClient
            model_to_use = self.gui.settings.get("gpt4free_model", "gpt-3.5-turbo") # Default g4f model
            logger.info(f"Using g4f client with model: {model_to_use}")
        else:
            if not self.client:
                logger.info("OpenAI client not initialized. Attempting to re-initialize.")
                self.update_openai_client() # Try with default key
                if not self.client:
                    logger.error("OpenAI client is not available after re-initialization attempt.")
                    return None
            target_client = self.client
            model_to_use = self.api_model
            logger.info(f"Using OpenAI compatible client with model: {model_to_use}")

        try:
            # Model-specific message adjustments (e.g., for Gemini-like models via OpenAI API)
            self.change_last_message_to_user_for_gemini(model_to_use, combined_messages)
            
            final_params = self.get_final_params(model_to_use, combined_messages)
            # save_combined_messages(final_params['messages'], f"ToOpenAI_{model_to_use.replace('/','_')}")

            logger.info(f"Requesting completion from {model_to_use} with temp={final_params.get('temperature')}, max_tokens={final_params.get('max_tokens')}")
            completion = target_client.chat.completions.create(**final_params)
            
            if completion and completion.choices:
                response_content = completion.choices[0].message.content
                logger.info("Completion successful.")
                return response_content.strip() if response_content else None
            else:
                logger.warning("No completion choices received or completion object is empty.")
                if completion: self.try_print_error(completion) # Log error if available
                return None
        except Exception as e:
            logger.error(f"Error during OpenAI/g4f API call: {str(e)}", exc_info=True)
            if hasattr(e, 'response') and e.response: # For openai.APIError
                 logger.error(f"API Error details: Status={e.response.status_code}, Body={e.response.text}")
            return None


    def change_last_message_to_user_for_gemini(self, api_model, combined_messages):
        # ... (keep existing method) ...
        # This is for models that don't like 'system' as the last role if they are Gemini-based
        if combined_messages and ("gemini" in api_model.lower() or "gemma" in api_model.lower()) and \
           combined_messages[-1]["role"] == "system":
            logger.info(f"Adjusting last message for {api_model}: system -> user with [SYSTEM INFO] prefix.")
            combined_messages[-1]["role"] = "user"
            combined_messages[-1]["content"] = f"[SYSTEM INFO] {combined_messages[-1]['content']}"


    def try_print_error(self, completion_or_error): # Renamed for clarity
        # ... (keep existing method, ensure it handles different error structures) ...
        # This method attempts to log detailed error information from an API response.
        # It needs to be robust to various error formats from different providers (OpenAI, g4f, etc.)
        logger.warning("Attempting to print error details from API response/error object.")
        if not completion_or_error:
            logger.warning("No error object or completion data to parse.")
            return

        # Example for OpenAI-like error structure (adjust as needed for g4f or others)
        if hasattr(completion_or_error, 'error') and completion_or_error.error:
            error_data = completion_or_error.error
            logger.warning(f"API Error: Code={getattr(error_data, 'code', 'N/A')}, Message='{getattr(error_data, 'message', 'N/A')}', Type='{getattr(error_data, 'type', 'N/A')}'")
            if hasattr(error_data, 'param') and error_data.param:
                logger.warning(f"  Param: {error_data.param}")
        elif isinstance(completion_or_error, dict) and 'error' in completion_or_error: # Generic dict check
             error_data = completion_or_error['error']
             logger.warning(f"API Error (from dict): {error_data}")
        elif hasattr(completion_or_error, 'message'): # Simple error object
             logger.warning(f"API Error: {completion_or_error.message}")
        else:
            logger.warning(f"Could not parse detailed error. Raw object: {str(completion_or_error)[:500]}") # Log snippet


    def _clean_response(self, response_text: str) -> str:
        # ... (keep existing _clean_response method) ...
        if not isinstance(response_text, str):
            logger.warning(f"Clean response expected string, got {type(response_text)}. Returning as is.")
            return response_text
        
        cleaned = response_text
        # Remove Markdown code blocks if they wrap the entire response
        if cleaned.startswith("```json\n") and cleaned.endswith("\n```"):
            cleaned = cleaned[len("```json\n"):-len("\n```")]
        elif cleaned.startswith("```\n") and cleaned.endswith("\n```"):
            cleaned = cleaned[len("```\n"):-len("\n```")]
        elif cleaned.startswith("```") and cleaned.endswith("```"): # More generic ```
            cleaned = cleaned[3:-3]
            
        return cleaned.strip() # General strip for leading/trailing whitespace


    def generate_request_gemini(self, combined_messages):
        # ... (keep existing generate_request_gemini method) ...
        # Ensure this formats the 'contents' correctly for Gemini API
        # and uses the correct 'generationConfig' parameters.
        params_for_gemini = self.get_params(model="gemini-pro") # Use a representative Gemini model name for param mapping
        
        # Gemini API expects 'contents' as a list of turns, where each turn has 'role' and 'parts'.
        # 'role' should be 'user' or 'model'.
        gemini_contents = []
        for msg in combined_messages: # combined_messages should already be formatted by _format_messages_for_gemini
            role = "model" if msg["role"] == "assistant" else msg["role"] # map assistant to model
            if role not in ["user", "model"]: # Gemini only accepts user/model
                logger.warning(f"Invalid role '{role}' for Gemini, converting to 'user'. Content: {msg['content'][:50]}")
                role = "user" # Or skip/handle error
            gemini_contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        data = {
            "contents": gemini_contents,
            "generationConfig": params_for_gemini
        }
        # safetySettings can be added here if needed

        headers = {"Content-Type": "application/json"} # Auth is usually part of URL for Gemini API key or service account

        api_url_with_key = self.api_url # Assuming api_url for Gemini includes the key or is a proxy
        if ":generateContent" not in api_url_with_key and not api_url_with_key.endswith("/generateContent"):
             api_url_with_key = api_url_with_key.replace("/v1beta/models/", "/v1beta/models/") + ":generateContent"
             if "?key=" not in api_url_with_key and self.api_key: # Add key if not in URL
                 api_url_with_key += f"?key={self.api_key}"


        logger.info(f"Sending request to Gemini API: {api_url_with_key}")
        # save_combined_messages(data, "ToGeminiAPI") # For debugging request payload

        try:
            response = requests.post(api_url_with_key, headers=headers, json=data, timeout=40)
            response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)

            response_data = response.json()
            if response_data.get("candidates"):
                generated_text = response_data["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text", "")
                logger.info("Gemini response successful.")
                return generated_text
            else:
                logger.warning(f"Gemini response missing candidates. Full response: {response_data}")
                # Check for promptFeedback if candidates are missing
                if "promptFeedback" in response_data:
                    logger.warning(f"Gemini Prompt Feedback: {response_data['promptFeedback']}")
                return None
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"Gemini API HTTP error: {http_err} - Response: {http_err.response.text}")
            return None
        except Exception as e:
            logger.error(f"Error during Gemini API request: {str(e)}", exc_info=True)
            return None


    def generate_request_common(self, combined_messages):
        # ... (keep existing generate_request_common method) ...
        # This is for other custom API endpoints that mimic OpenAI structure.
        # Ensure it uses the model from settings: NM_API_MODEL
        model_name = self.gui.settings.get("NM_API_MODEL", self.api_model) # Fallback to general api_model
        params_for_common = self.get_params(model=model_name)

        data = {
            "model": model_name,
            "messages": combined_messages, # Assumes combined_messages are in standard role/content format
            **params_for_common # Spread other parameters like temperature, max_tokens
        }

        headers = {
            "Content-Type": "application/json",
            # Authorization might be needed depending on the common API
            # "Authorization": f"Bearer {self.api_key}" # Uncomment if API uses Bearer token
        }
        if self.api_key: # Add auth header if api_key is present and API likely needs it
            headers["Authorization"] = f"Bearer {self.api_key}"


        logger.info(f"Sending request to common API: {self.api_url} with model: {model_name}")
        # save_combined_messages(data, "ToCommonAPI")

        try:
            response = requests.post(self.api_url, headers=headers, json=data, timeout=40)
            response.raise_for_status()
            
            response_data = response.json()
            if response_data.get("choices"):
                generated_text = response_data["choices"][0].get("message", {}).get("content", "")
                logger.info("Common API response successful.")
                return generated_text
            else:
                logger.warning(f"Common API response missing choices. Full response: {response_data}")
                return None
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"Common API HTTP error: {http_err} - Response: {http_err.response.text}")
            return None
        except Exception as e:
            logger.error(f"Error during common API request: {str(e)}", exc_info=True)
            return None


    def _get_provider_key(self, model_name: str) -> str:
        # ... (keep existing _get_provider_key method) ...
        if not model_name: return 'openai' # Default if no model name
        model_name_lower = model_name.lower()
        if 'gpt-4' in model_name_lower or 'gpt-3.5' in model_name_lower: return 'openai'
        if 'gemini' in model_name_lower or 'gemma' in model_name_lower: return 'gemini' # Gemma often uses Gemini params
        if 'claude' in model_name_lower: return 'anthropic'
        if 'deepseek' in model_name_lower: return 'deepseek'
        # Add more known provider keywords
        logger.info(f"Unknown provider for model '{model_name}', defaulting to 'openai' parameter naming conventions.")
        return 'openai' # Default for unknown models


    def get_params(self, model: str = None) -> Dict[str, Any]:
        # ... (keep existing get_params method, ensure it checks USE_MODEL_PRESENCE_PENALTY) ...
        current_model_name = model if model is not None else self.api_model
        provider_key = self._get_provider_key(current_model_name)
        
        params: Dict[str, Any] = {}

        if self.temperature is not None:
            params['temperature'] = self.temperature

        if self.max_response_tokens is not None:
            if provider_key in ['openai', 'deepseek', 'anthropic']: # Anthropic uses max_tokens_to_sample or max_tokens
                params['max_tokens'] = self.max_response_tokens
            elif provider_key == 'gemini':
                params['maxOutputTokens'] = self.max_response_tokens
            # Add other provider mappings for max tokens if different

        if self.presence_penalty is not None and bool(self.gui.settings.get("USE_MODEL_PRESENCE_PENALTY", False)):
            if provider_key in ['openai', 'deepseek']:
                params['presence_penalty'] = self.presence_penalty
            elif provider_key == 'gemini': # Gemini does not have a direct presence_penalty equivalent in genConfig
                logger.info(f"Presence penalty not directly supported by Gemini config for model {current_model_name}. Skipping.")
            # Anthropic also doesn't have a direct equivalent.
        
        # Example: top_p (aka nucleus sampling)
        # top_p_value = float(self.gui.settings.get("MODEL_TOP_P", 0.9)) # Get from settings
        # if top_p_value > 0 and top_p_value <=1.0: # Check if valid
        #     if provider_key in ['openai', 'deepseek', 'gemini']: # Gemini supports topP
        #         params['top_p'] = top_p_value
        #     elif provider_key == 'anthropic':
        #         params['top_p'] = top_p_value # Anthropic supports top_p

        # Remove parameters not supported by specific model variants if known
        params = self.remove_unsupported_params(current_model_name, params)
        return params

    def get_final_params(self, model_name: str, messages: List[Dict]) -> Dict[str, Any]:
        # ... (keep existing get_final_params method) ...
        # This combines model, messages, and other parameters from get_params()
        final_params = {
            "model": model_name,
            "messages": messages,
            **self.get_params(model=model_name) # Spread parameters
        }
        return final_params

    def remove_unsupported_params(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        # ... (keep existing remove_unsupported_params method) ...
        # Example: Some specific Gemini models might not support all general Gemini params.
        # model_name_lower = model_name.lower()
        # if "gemini-1.5-pro-preview" in model_name_lower: # Fictional example
        #     if 'presencePenalty' in params: # If it was mistakenly added
        #         del params['presencePenalty']
        #         logger.info("Removed 'presencePenalty' as it's not supported by gemini-1.5-pro-preview.")
        return params


    def process_text_to_voice(self, text_to_speak: str) -> str:
        # ... (keep existing process_text_to_voice method) ...
        # This method cleans text for TTS by removing tags, etc.
        if not isinstance(text_to_speak, str):
            logger.warning(f"process_text_to_voice expected string, got {type(text_to_speak)}. Converting to string.")
            text_to_speak = str(text_to_speak)

        # Remove content within all tags like <tag>content</tag>
        # This regex finds <tag>...</tag> and replaces the whole thing.
        clean_text = re.sub(r"<[^>]+>.*?</[^>]+>", "", text_to_speak, flags=re.DOTALL)
        
        # Remove any remaining standalone tags like <tag>
        clean_text = re.sub(r"<[^>]+>", "", clean_text)
        
        # Replace numbers with words (if this utility function exists and is desired)
        try:
            clean_text = replace_numbers_with_words(clean_text)
        except NameError: # If replace_numbers_with_words is not defined globally
            logger.debug("replace_numbers_with_words utility not found or used.")
            pass 
            
        # Transliterate (if desired and function exists)
        # clean_text = transliterate_english_to_russian(clean_text)

        if not clean_text.strip():
            clean_text = "..." # Default for empty speech
            logger.info("TTS text was empty after cleaning, using default '...'")
            
        return clean_text.strip()


    def reload_promts(self): # Renamed in original, keep consistent if intended
        logger.info("Reloading current character data.")
        if self.current_character:
            self.current_character.reload_character_data()
            logger.info(f"Character {self.current_character.name} data reloaded.")
        else:
            logger.warning("No current character selected to reload.")

    def add_temporary_system_info(self, content: str):
        """
        Adds a system message that will be included in the *next* call to the LLM
        and then saved to history.
        """
        system_info_message = {"role": "system", "content": content}
        self.infos_to_add_to_history.append(system_info_message)
        logger.info(f"Queued temporary system info: {content[:100]}...")

    #region TokensCounting (Kept from original, ensure it works with new structure)
    def calculate_cost(self, user_input_text: str): # Parameter is text, not full message
        if not self.hasTokenizer:
            logger.warning("Tokenizer not available, cannot calculate cost accurately.")
            return 0, 0.0

        # To calculate cost, we need the messages that *would be sent*
        # This is complex as it involves DSL processing.
        # For a rough estimate, we can count tokens for current history + new user input.
        
        temp_messages_for_costing = []
        if self.current_character:
            history_data = self.current_character.history_manager.load_history()
            temp_messages_for_costing.extend(history_data.get("messages", []))
        
        temp_messages_for_costing.append({"role": "user", "content": user_input_text})
        
        token_count = self.count_tokens(temp_messages_for_costing)
        cost = (token_count / 1000) * self.cost_input_per_1000 # Example input cost
        
        logger.info(f"Estimated token count for input '{user_input_text[:50]}...': {token_count}, Estimated cost: {cost:.5f}")
        return token_count, cost

    def count_tokens(self, messages_list: List[Dict]) -> int:
        if not self.hasTokenizer:
            return 0 # Or estimate based on char count

        total_tokens = 0
        for msg in messages_list:
            if isinstance(msg, dict) and "content" in msg and isinstance(msg["content"], str):
                try:
                    total_tokens += len(self.tokenizer.encode(msg["content"]))
                except Exception as e:
                    logger.warning(f"Error encoding content for token counting: {e}. Content snippet: {msg['content'][:50]}")
            # Add handling for other message formats if necessary (e.g., Gemini's 'parts')
        return total_tokens
    #endregion

    def GetOtherKey(self) -> str | None: # Added return type hint
        # ... (keep existing GetOtherKey method) ...
        # This method provides a way to cycle through API keys.
        # Ensure NM_API_KEY_RES is correctly fetched from settings.
        
        # Fallback to main API key if it's the only one or if list is misconfigured.
        # Ensure self.api_key is part of the list or considered.
        
        all_keys = []
        if self.api_key: # Prioritize the primary key
            all_keys.append(self.api_key)
        
        reserve_keys_str = self.gui.settings.get("NM_API_KEY_RES", "")
        if reserve_keys_str:
            all_keys.extend([key.strip() for key in reserve_keys_str.split() if key.strip()])
        
        # Remove duplicates while preserving order (somewhat)
        seen = set()
        unique_keys = [x for x in all_keys if not (x in seen or seen.add(x))]

        if not unique_keys:
            logger.warning("No API keys configured (main or reserve).")
            return None
        
        if len(unique_keys) == 1:
            self.last_key = 0 # Reset index if only one key
            return unique_keys[0]

        # Cycle through keys
        self.last_key = (self.last_key + 1) % len(unique_keys)
        selected_key = unique_keys[self.last_key]
        
        logger.info(f"Selected API key index: {self.last_key} (masked: {SH(selected_key)}) from {len(unique_keys)} unique keys.")
        return selected_key
    
    # region невошедшие
    def get_room_name(self, room_id):
        room_names = {
            0: "Кухня",
            1: "Зал",
            2: "Комната",
            3: "Туалет",
            4: "Подвал"
        }
        return room_names.get(room_id, "?")
    
    def add_temporary_system_message(self, messages: List[Dict], content: str):
        if not isinstance(messages, list):
            logger.error("add_temporary_system_message ожидает список сообщений.")
            return

        system_message = {
            "role": "system",
            "content": content
        }
        messages.append(system_message)
        logger.debug(f"Временно добавлено системное сообщение в переданный список: {content[:100]}...")

    # endregion