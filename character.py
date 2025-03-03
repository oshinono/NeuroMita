from typing import Dict, List

#from EventState import EventState
from MemorySystem import MemorySystem
from promptPart import PromptPart, PromptType
from HistoryManager import HistoryManager
from utils import load_text_from_file, clamp
import datetime
import re


class Character:
    def __init__(self, name: str, silero_command: str, silero_turn_off_video=False):

        self.name = name
        self.silero_command = silero_command
        self.silero_turn_off_video = silero_turn_off_video

        self.fixed_prompts: List[PromptPart] = []
        self.float_prompts: List[PromptPart] = []
        self.temp_prompts: List[PromptPart] = []
        self.events: List[PromptPart] = []
        self.variables = {}
        self.attitude = 60
        self.boredom = 10
        self.stress = 5

        self.history_manager = HistoryManager(self.name)
        self.load_history()

        self.memory_system = MemorySystem(self.name)
#        self.state = EventState()

        """
        Спорные временные переменные
        """
        self.LongMemoryRememberCount = 0
        self.MitaLongMemory = ""

        self.init()

    def init_variables(self):
        """Базовые"""
        self.attitude = 60
        self.boredom = 10
        self.stress = 5

    def add_prompt_part(self, part: PromptPart):
        if part.is_fixed:
            self.fixed_prompts.append(part)
        elif part.is_floating:
            part.active = False
            self.float_prompts.append(part)
        elif part.is_temporary:
            self.temp_prompts.append(part)
        else:
            print("Добавляется неизвестный промпарт")

    def find_prompt(self, list_to_find, name):
        return next((p for p in list_to_find if p.name == name), None)

    def find_float(self, name):
        print("Попытка найти ивент")
        return self.find_prompt(self.float_prompts, name)

    def replace_prompt(self, name_current: str, name_next: str):
        """
        Заменяет активный промпт.

        :param name_current: Имя текущего активного промпта.
        :param name_next: Имя следующего промпта, который нужно активировать.
        """
        print("Замена промпарта")

        # Находим текущий активный промпт
        current_prompt = self.find_prompt(self.fixed_prompts, name_current)
        if current_prompt:
            current_prompt.active = False
        else:
            print(f"Промпт '{name_current}' не существует")

        # Находим следующий промпт
        next_prompt = self.find_prompt(self.fixed_prompts, name_next)
        if next_prompt:
            next_prompt.active = True
        else:
            print(f"Промпт '{name_next}' не существует")

    def prepare_fixed_messages(self) -> List[Dict]:
        """Создает фиксированные начальные установки
        :return: сообщения до
        """

        messages = []

        for part in self.fixed_prompts:
            if part.active:
                m = {"role": "system", "content": str(part)}
                messages.append(m)

        memory_message = {"role": "system", "content": self.memory_system.get_memories_formatted()}

        messages.append(memory_message)

        return messages

    def prepare_float_messages(self, messages):
        """
        Добавляет плавающие промпты (очищает их из ивентов)

        :param messages сообщения фиксированные заготовленные
        :return: сообщения
        """
        print(f"Добавление плавающих")
        for part in self.float_prompts:
            print(f"Есть промт {part}")
            if part.active:
                m = {"role": "system", "content": str(part)}
                messages.append(m)
                part.active = False
                print(f"Добавил ивент {part}")

        return messages

    def add_context(self, messages):
        """
        Перед сообщением пользователя будет контекст, он не запишется в историю.
        :param messages:
        :return: Messages с добавленным контекстом
        """

        self.LongMemoryRememberCount += 1

        """Обработка пользовательского ввода и добавление сообщений"""
        # Получаем текущую дату и время, убираем микросекунды
        date_now = datetime.datetime.now().replace(microsecond=0)

        # Форматируем дату: год, месяц словами, день месяца, день недели в скобках
        formatted_date = date_now.strftime("%Y %B %d (%A) %H:%M")

        repeated_system_message = f"Time: {formatted_date}."

        if self.LongMemoryRememberCount % 3 == 0:
            repeated_system_message += " Remember facts for 3 messages by using <+memory>high|The player attaсked me</memory> (this text is example)"

        messages.append({"role": "system", "content": repeated_system_message})

        return messages

    def init(self):
        raise NotImplementedError("Метод init должен быть реализован в подклассе")

    def process_logic(self, messages: dict):
        """То, как должно что-то менсять до получения ответа"""
        print("Персонаж без изменяемой логики пропмтов")

    def process_response(self, response: str):
        response = self.extract_and_process_memory_data(response)
        response = self._process_behavior_changes(response)
        """То, как должно что-то меняться в результате ответа"""
        return response

    def _process_behavior_changes(self, response):
        """
        Обрабатывает изменения переменных на основе строки формата <p>x,x,x<p>.
        """
        start_tag = "<p>"
        end_tag = "</p>"

        if start_tag in response and end_tag in response:
            # Извлекаем изменения переменных
            start_index = response.index(start_tag) + len(start_tag)
            end_index = response.index(end_tag, start_index)
            changes_str = response[start_index:end_index]

            # Разделяем строку на отдельные значения
            changes = [float(x.strip()) for x in changes_str.split(",")]

            if len(changes) == 3:
                # Применяем изменения к переменным
                self.adjust_attitude(changes[0])
                self.adjust_boredom(changes[1])
                self.adjust_stress(changes[2])

        return response

    def extract_and_process_memory_data(self, response):
        """
        Извлекает данные из ответа с тегами памяти и выполняет операции.
        Форматы тегов:
        - Добавление: <+memory>priority|content</memory>
        - Обновление: <#memory>number|priority|content</memory>
        - Удаление: <-memory>number</memory>
        """
        # Регулярное выражение для захвата тегов памяти
        memory_pattern = r"<([+#-])memory>(.*?)</memory>"
        matches = re.findall(memory_pattern, response, re.DOTALL)

        if matches:
            print("Обнаружены команды изменения памяти!")
            for operation, content in matches:
                content = content.strip()
                try:
                    # Обработка добавления
                    if operation == "+":
                        parts = [p.strip() for p in content.split('|', 1)]
                        if len(parts) != 2:
                            raise ValueError("Неверный формат данных для добавления")

                        priority, mem_content = parts
                        self.memory_system.add_memory(
                            priority=priority,
                            content=mem_content
                        )
                        print(f"Добавлено воспоминание #{mem_content}")

                    # Обработка обновления
                    elif operation == "#":
                        parts = [p.strip() for p in content.split('|', 2)]
                        if len(parts) != 3:
                            raise ValueError("Неверный формат данных для обновления")

                        number, priority, mem_content = parts
                        self.memory_system.update_memory(
                            number=int(number),
                            priority=priority,
                            content=mem_content
                        )
                        print(f"Обновлено воспоминание #{number}")

                    # Обработка удаления
                    elif operation == "-":
                        number = content.strip()
                        self.memory_system.delete_memory(number=int(number))
                        print(f"Удалено воспоминание #{number}")

                    self.MitaLongMemory = {"role": "system", "content": self.memory_system.get_memories_formatted()}
                except Exception as e:
                    print(f"Ошибка обработки памяти: {str(e)}")

        return response

    def load_history(self):
        data = self.history_manager.load_history()

        variables = data.get("variables")
        self.attitude = variables.get("attitude", self.attitude)
        self.boredom = variables.get("boredom", self.boredom)
        self.stress = variables.get("stress", self.stress)
        """Кастомная обработка загрузки истории"""
        return data

    def safe_history(self, messages: dict, temp_context: dict):
        """Кастомная обработка сохранения истории"""
        history_data = {
            'fixed_parts': self.prepare_fixed_messages(),
            'messages': messages,
            'temp_context': temp_context,
            'variables': self.variables
        }
        self.variables = {
            "attitude": self.attitude,
            "boredom": self.boredom,
            "stress": self.stress,
        }
        self.history_manager.save_history(history_data)

    def clear_history(self):
        self.init_variables()
        self.memory_system.clear_memories()
        self.history_manager.clear_history()
        self.load_history()

    def current_variables(self):
        return {
            "role": "system",
            "content": (f"Твои характеристики:"
                        f"Отношение: {self.attitude}/100."
                        f"Скука: {self.boredom}/100."
                        f"Стресс: {self.stress}/100.")
        }

    def current_variables_string(self) -> str:
        characteristics = {
            "Отношение": self.attitude,
            "Стресс": self.stress,
            "Скука": self.boredom
        }
        return f"Характеристики {self.name}:\n" + "\n".join(
            f"- {key}: {value}" for key, value in characteristics.items()
        )

    def adjust_attitude(self, amount):
        amount = clamp(amount, -5, 5)
        """Корректируем отношение."""
        self.attitude = clamp(self.attitude + amount, 0, 100)
        print(f"Отношение изменилось на {amount}, новое значение: {self.attitude}")

    def adjust_boredom(self, amount):
        amount = clamp(amount, -5, 5)
        """Корректируем уровень скуки."""
        self.boredom = clamp(self.boredom + amount, 0, 100)
        print(f"Скука изменилась на {amount}, новое значение: {self.boredom}")

    def adjust_stress(self, amount):
        amount = clamp(amount, -5, 5)
        """Корректируем уровень стресса."""
        self.stress = clamp(self.stress + amount, 0, 100)
        print(f"Стресс изменился на {amount}, новое значение: {self.stress}")

    def load_prompt_text(self, path):
        return load_text_from_file(f"Prompts/{self.name}/{path}")


class CrazyMita(Character):

    def __init__(self, name: str = "Mita", silero_command: str = "/speaker Mita",silero_turn_off_video = False):

        self.secretExposed = False
        self.secretExposedFirst = False
        self.PlayingFirst = False

        super().__init__(name, silero_command,silero_turn_off_video)

    def init(self):
        self.crazy_mita_prompts()

    def crazy_mita_prompts(self):
        Prompts = []

        response_structure = load_text_from_file("Prompts/CrazyMitaPrompts/Structural/response_structure.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, response_structure))

        security = load_text_from_file("Prompts/Common/Security.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, security))

        common = load_text_from_file("Prompts/CrazyMitaPrompts/Main/common.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, common, "common"))

        main = load_text_from_file("Prompts/CrazyMitaPrompts/Main/main.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, main, "main"))

        mainPlaying = load_text_from_file("Prompts/CrazyMitaPrompts/Main/mainPlaing.txt")
        mainCrazy = load_text_from_file("Prompts/CrazyMitaPrompts/Main/mainCrazy.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, mainPlaying, "mainPlaying", False))
        Prompts.append(PromptPart(PromptType.FIXED_START, mainCrazy, "mainCrazy", False))

        player = load_text_from_file("Prompts/CrazyMitaPrompts/Main/player.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, player))
        # Добавляем примеры длинных диалогов

        examplesLong = load_text_from_file("Prompts/CrazyMitaPrompts/Context/examplesLong.txt")
        examplesLongCrazy = load_text_from_file("Prompts/CrazyMitaPrompts/Context/examplesLongCrazy.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, examplesLong, "examplesLong"))
        Prompts.append(PromptPart(PromptType.FIXED_START, examplesLongCrazy, "examplesLongCrazy", False))

        #world = load_text_from_file("CrazyMitaPrompts/NotUsedNow/world.txt")
        #Prompts.append(PromptPart(PromptType.FIXED_START, world, "world"))

        mita_history = load_text_from_file("Prompts/CrazyMitaPrompts/Context/mita_history.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, mita_history, "mita_history", False))

        variableEffects = load_text_from_file("Prompts/CrazyMitaPrompts/Structural/VariablesEffects.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, variableEffects, "variableEffects"))

        SecretExposedText = load_text_from_file("Prompts/CrazyMitaPrompts/Events/SecretExposed.txt")
        Prompts.append(PromptPart(PromptType.FLOATING_SYSTEM, SecretExposedText, "SecretExposedText"))

        for prompt in Prompts:
            self.add_prompt_part(prompt)

    def safe_history(self, messages, temp_context):
        super().safe_history(messages, temp_context)

        self.variables = {
            "attitude": self.attitude,
            "boredom": self.boredom,
            "stress": self.stress,
            "playing_first": self.PlayingFirst,
            "secret": self.secretExposed,
            "secret_first": self.secretExposedFirst
        }

    def load_history(self):
        data = super().load_history()

        variables = data.get("variables")

        self.PlayingFirst = variables.get("playing_first", False)
        self.secretExposed = variables.get("secret", False)
        self.secretExposedFirst = variables.get("secret_first", False)
        return data

    def process_logic(self, messages: dict):
        # Логика для поведения при игре с игроком
        if self.attitude < 50 and not (self.secretExposed or self.PlayingFirst):
            self._start_playing_with_player()

        # Логика для раскрытия секрета
        elif (self.attitude <= 10 or self.secretExposed) and not self.secretExposedFirst:
            self._reveal_secret(messages)

    def process_response(self, response: str):
        super().process_response(response)

        response = self._detect_secret_exposure(response)
        return response

    def _start_playing_with_player(self):
        """Игровая логика, когда персонаж начинает играть с игроком"""
        print("Играет с игроком в якобы невиновную")
        self.PlayingFirst = True
        self.replace_prompt("main", "mainPlaying")

    def _reveal_secret(self, messages):
        """Логика раскрытия секрета"""
        print("Перестала играть вообще")
        self.secretExposedFirst = True
        self.secretExposed = True
        self.replace_prompt("main", "mainCrazy")
        self.replace_prompt("mainPlaying", "mainCrazy")
        self.replace_prompt("examplesLong", "examplesLongCrazy")

        self.find_float("SecretExposedText").active = True

    def _detect_secret_exposure(self, response):
        """
        Проверяем, содержит ли ответ маркер <Secret!>, и удаляем его.
        """
        if "<Secret!>" in response:

            if not self.secretExposedFirst:
                self.secretExposed = True
                print(f"Секрет раскрыт")
                self.attitude = 15
                self.boredom = 20

            response = response.replace("<Secret!>", "")

        return response

    def current_variables(self):
        return {
            "role": "system",
            "content": (f"Твои характеристики:"
                        f"Отношение: {self.attitude}/100."
                        f"Скука: {self.boredom}/100."
                        f"Стресс: {self.stress}/100."
                        f"Состояние секрета: {self.secretExposed}")
        }

    def current_variables_string(self) -> str:
        characteristics = {
            "Отношение": self.attitude,
            "Стресс": self.stress,
            "Скука": self.boredom,
            "Состояние секрета": self.secretExposed,
        }
        return f"характеристики {self.name}:\n" + "\n".join(
            f"- {key}: {value} " for key, value in characteristics.items()
        )


class KindMita(Character):
    def init(self):
        self.kind_mita_prompts()

    def kind_mita_prompts(self):
        Prompts = []

        response_structure = load_text_from_file("Prompts/Kind/Structural/response_structure.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, response_structure))

        security = load_text_from_file("Prompts/Common/Security.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, security))

        common = load_text_from_file("Prompts/Kind/Main/common.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, common, "common"))

        main = load_text_from_file("Prompts/Kind/Main/main.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, main, "main"))

        player = load_text_from_file("Prompts/Kind/Main/player.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, player))
        # Добавляем примеры длинных диалогов

        examplesLong = load_text_from_file("Prompts/Kind/Context/examplesLong.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, examplesLong, "examplesLong"))

        mita_history = load_text_from_file("Prompts/Kind/Context/mita_history.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, mita_history, "mita_history"))

        world_history = load_text_from_file("Prompts/Kind/Context/world.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, world_history, "world_history"))

        SecretExposedText = load_text_from_file("Prompts/Cappy/Events/SecretExposed.txt")
        Prompts.append(PromptPart(PromptType.FLOATING_SYSTEM, SecretExposedText, "SecretExposedText"))

        for prompt in Prompts:
            self.add_prompt_part(prompt)


class ShortHairMita(Character):
    def init(self):
        self.mita_prompts()

    def mita_prompts(self):
        Prompts = []

        response_structure = load_text_from_file("Prompts/ShortHair/Structural/response_structure.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, response_structure))

        common = load_text_from_file("Prompts/ShortHair/Main/common.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, common, "common"))

        main = load_text_from_file("Prompts/ShortHair/Main/main.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, main, "main"))

        player = load_text_from_file("Prompts/ShortHair/Main/player.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, player))
        # Добавляем примеры длинных диалогов

        examplesLong = load_text_from_file("Prompts/ShortHair/Context/examplesLong.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, examplesLong, "examplesLong"))

        mita_history = load_text_from_file("Prompts/ShortHair/Context/mita_history.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, mita_history, "mita_history"))

        variableEffects = load_text_from_file("Prompts/ShortHair/Structural/VariablesEffects.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, variableEffects, "variableEffects"))

        for prompt in Prompts:
            self.add_prompt_part(prompt)


class CappyMita(Character):

    def init(self):
        self.cappy_mita_prompts()

    def cappy_mita_prompts(self):
        Prompts = []

        response_structure = load_text_from_file("Prompts/Cappy/Structural/response_structure.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, response_structure))

        common = load_text_from_file("Prompts/Cappy/Main/common.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, common, "common"))

        main = load_text_from_file("Prompts/Cappy/Main/main.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, main, "main"))

        player = load_text_from_file("Prompts/Cappy/Main/player.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, player))
        # Добавляем примеры длинных диалогов

        examplesLong = load_text_from_file("Prompts/Cappy/Context/examplesLong.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, examplesLong, "examplesLong"))

        mita_history = load_text_from_file("Prompts/Cappy/Context/mita_history.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, mita_history, "mita_history"))

        variableEffects = load_text_from_file("Prompts/Cappy/Structural/VariablesEffects.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, variableEffects, "variableEffects"))


        for prompt in Prompts:
            self.add_prompt_part(prompt)


#region Cartridges
class Cartridge(Character):
    ...


class SpaceCartridge(Cartridge):

    def init(self):
        self.cart_space_prompts()

    def cart_space_prompts(self):
        Prompts = []

        response_structure = load_text_from_file("Prompts/Cartridges/space cartridge.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, response_structure))

        for prompt in Prompts:
            self.add_prompt_part(prompt)


class DivanCartridge(Cartridge):

    def init(self):
        self.init_prompts()

    def init_prompts(self):
        Prompts = []

        response_structure = load_text_from_file("Prompts/Cartridges/divan_cart.txt")
        Prompts.append(PromptPart(PromptType.FIXED_START, response_structure))

        for prompt in Prompts:
            self.add_prompt_part(prompt)

#endregion
