import threading
import berserk
from camera_chess.constants import LICHESS_TOKEN_1


class LichessGame(threading.Thread):
    def __init__(self, token, **kwargs):
        super().__init__(**kwargs)

        session = berserk.TokenSession(token)
        self.client = berserk.Client(session=session)

        games = self.client.games.get_ongoing()
        if len(games) != 1:
            raise ValueError(f'Need exactly 1 game but have {len(games)}')
        game = games[0]
        self.colour = game['color']
        self.game_id = game['gameId']

        self.stream = self.client.board.stream_game_state(self.game_id)
        self.current_state = next(self.stream)

        self.moves = []

    def run(self):
        for event in self.stream:
            if event['type'] == 'gameState':
                self.handle_state_change(event)

    def handle_state_change(self, game_state):
        self.moves = game_state['moves'].split()
        print(f'Lichess: {self.moves[-1]}')

    def make_move(self, move):
        self.client.board.make_move(self.game_id, move)


def main():
    LichessGame(LICHESS_TOKEN_1)


if __name__ == '__main__':
    main()
