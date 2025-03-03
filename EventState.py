from abc import ABC, abstractmethod


class State(ABC):
    """Абстрактный базовый класс для состояний"""

    def __init__(self, fsm=None):
        self.fsm = fsm  # Ссылка на конечный автомат

    @abstractmethod
    def on_enter(self):
        """Вызывается при входе в состояние"""
        pass

    @abstractmethod
    def on_exit(self):
        """Вызывается при выходе из состояния"""
        pass

    @abstractmethod
    def handle_event(self, event):
        """Обработка событий, должен возвращать следующее состояние"""
        pass


class FiniteStateMachine:
    """Класс конечного автомата"""

    def __init__(self, initial_state):
        self.current_state = initial_state
        self.current_state.fsm = self
        self.current_state.on_enter()

    def transition_to(self, state):
        """Выполнить переход к указанному состоянию"""
        self.current_state.on_exit()
        self.current_state = state
        self.current_state.fsm = self
        self.current_state.on_enter()

    def handle_event(self, event):
        """Обработать событие"""
        next_state = self.current_state.handle_event(event)
        if next_state is not None and next_state != self.current_state:
            self.transition_to(next_state)


# Пример использования
class IdleState(State):
    def on_enter(self):
        print("Вход в состояние Idle")

    def on_exit(self):
        print("Выход из состояния Idle")

    def handle_event(self, event):
        if event == 'start':
            return RunningState()
        return self  # Остаемся в текущем состоянии


class RunningState(State):
    def on_enter(self):
        print("Вход в состояние Running")

    def on_exit(self):
        print("Выход из состояния Running")

    def handle_event(self, event):
        if event == 'stop':
            return IdleState()
        elif event == 'pause':
            return PausedState()
        return self


class PausedState(State):
    def on_enter(self):
        print("Вход в состояние Paused")

    def on_exit(self):
        print("Выход из состояния Paused")

    def handle_event(self, event):
        if event == 'resume':
            return RunningState()
        elif event == 'stop':
            return IdleState()
        return self


# Пример использования
if __name__ == "__main__":
    fsm = FiniteStateMachine(IdleState())

    # Тестирование переходов
    fsm.handle_event('start')  # Idle -> Running
    fsm.handle_event('pause')  # Running -> Paused
    fsm.handle_event('resume')  # Paused -> Running
    fsm.handle_event('stop')  # Running -> Idle


class Activity:
    """Недолгая активность"""

    def __init__(self):
        ...
