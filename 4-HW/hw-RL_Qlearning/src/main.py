"""Pygame UI for the Q-learning GridWorld project.

Launch: python main.py
Controls: Train (T), Play (P/Space), Reset (R), ESC to exit.
"""

import csv
import sys
import pygame

from gridworld import GridWorld
from qlearning import QLearningAgent

SCREEN_W, SCREEN_H = 930, 620
CELL = 57
MAP_X, MAP_Y = 43, 91
PANEL_X = 548
FPS = 60
TRAIN_EPISODES = 5000
EPISODES_PER_FRAME = 14
MOVE_INTERVAL_MS = 360

BG = (247, 249, 252)
DARK = (28, 43, 63)
TEXT_MUTED = (94, 109, 128)
WALL = (65, 80, 99)
EMPTY = (233, 239, 246)
BORDER = (207, 216, 229)
BLUE = (44, 107, 222)
GREEN = (38, 154, 94)
GOLD = (235, 168, 55)
WHITE = (255, 255, 255)


def font(size, bold=False):
    return pygame.font.SysFont("arial", size, bold=bold)


def text(surface, label, x, y, size=20, color=DARK, bold=False):
    surface.blit(font(size, bold).render(str(label), True, color), (x, y))


def make_button(x, y, width, height):
    """Tạo vùng hình chữ nhật dùng làm nút bấm."""
    return pygame.Rect(x, y, width, height)


def draw_button(surface, rect, label, color, enabled=True):
    """Vẽ một nút bấm. Không cần tạo class riêng cho nút."""
    if not enabled:
        color = (172, 182, 193)
    pygame.draw.rect(surface, color, rect, border_radius=9)
    letters = font(17, bold=True).render(label, True, WHITE)
    surface.blit(letters, letters.get_rect(center=rect.center))


class RobotApp:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Q-Learning Robot - GridWorld")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.env = GridWorld()
        self.agent = QLearningAgent(self.env)
        self.train_button = make_button(PANEL_X, 143, 145, 43)
        self.play_button = make_button(PANEL_X + 158, 143, 145, 43)
        self.reset_button = make_button(PANEL_X, 198, 303, 43)
        self.training = False
        self.playing = False
        self.robot_state = self.env.start
        self.path = []
        self.path_ok = False
        self.path_index = 0
        self.last_move = 0
        self.status = "Press TRAIN to teach the robot."
        self.bfs_steps = len(self.env.shortest_path()) - 1

    def reset(self):
        self.agent.reset()
        self.training = False
        self.playing = False
        self.robot_state = self.env.start
        self.path = []
        self.path_ok = False
        self.path_index = 0
        self.status = "Q-table cleared. Press TRAIN."

    def start_train(self):
        if self.training:
            self.training = False
            self.status = "Training paused. Press TRAIN to resume."
        elif len(self.agent.history) < TRAIN_EPISODES:
            self.playing = False
            self.robot_state = self.env.start
            self.training = True
            self.status = "Training: exploring and updating Q..."
        else:
            self.status = "Already trained. RESET to start over."

    def start_play(self):
        if self.training:
            self.status = "Pause training before playing."
            return
        if not self.agent.history:
            self.status = "Please TRAIN first."
            return
        self.path, self.path_ok = self.agent.greedy_path()
        self.path_index = 0
        self.robot_state = self.env.start
        if not self.path_ok:
            self.playing = False
            self.status = "No valid greedy route yet. TRAIN more."
            return
        self.playing = True
        self.last_move = pygame.time.get_ticks()
        self.status = f"Playing: {len(self.path)-1} steps to the goal."

    def save_csv(self):
        with open("training_log.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "steps", "success", "epsilon"])
            for i, item in enumerate(self.agent.history, start=1):
                writer.writerow([i, item.reward, item.steps, int(item.success), item.epsilon])

    def update(self):
        if self.training:
            remaining = TRAIN_EPISODES - len(self.agent.history)
            for _ in range(min(EPISODES_PER_FRAME, remaining)):
                self.agent.train_episode()
            if len(self.agent.history) == TRAIN_EPISODES:
                self.training = False
                self.path, self.path_ok = self.agent.greedy_path()
                self.save_csv()
                self.status = "Training done! Press PLAY to watch."
                if not self.path_ok:
                    self.status = "Training done; route not found. RESET and retry."
        if self.playing and pygame.time.get_ticks() - self.last_move >= MOVE_INTERVAL_MS:
            self.last_move = pygame.time.get_ticks()
            self.path_index += 1
            if self.path_index >= len(self.path):
                self.playing = False
                return
            self.robot_state = self.path[self.path_index]
            if self.robot_state == self.env.goal:
                self.playing = False
                self.status = "Success! Robot reached the green goal."

    def center(self, state):
        row, col = self.env.decode(state)
        return MAP_X + col * CELL + CELL // 2, MAP_Y + row * CELL + CELL // 2

    def draw_grid(self):
        for row in range(self.env.height):
            for col in range(self.env.width):
                ch = self.env.layout[row][col]
                rect = pygame.Rect(MAP_X + col*CELL, MAP_Y + row*CELL, CELL-3, CELL-3)
                color = WALL if ch == "#" else EMPTY
                if ch == "S": color = (218, 231, 252)
                if ch == "G": color = (197, 238, 215)
                pygame.draw.rect(self.screen, color, rect, border_radius=6)
                if ch == "#":
                    pygame.draw.line(self.screen, (95, 108, 125), rect.topleft, rect.bottomright, 1)
                elif ch == "G":
                    text(self.screen, "GOAL", rect.x + 5, rect.y + 19, 12, (20, 119, 73), True)
                elif ch == "S":
                    text(self.screen, "START", rect.x + 4, rect.y + 19, 11, BLUE, True)
        if self.path_ok and len(self.path) > 1:
            centers = [self.center(s) for s in self.path]
            pygame.draw.lines(self.screen, GOLD, False, centers, 4)
            for pt in centers[1:-1]:
                pygame.draw.circle(self.screen, GOLD, pt, 4)
        x, y = self.center(self.robot_state)
        pygame.draw.circle(self.screen, WHITE, (x, y), 18)
        pygame.draw.circle(self.screen, BLUE, (x, y), 15)
        pygame.draw.circle(self.screen, WHITE, (x-5, y-3), 3)
        pygame.draw.circle(self.screen, WHITE, (x+5, y-3), 3)
        pygame.draw.arc(self.screen, WHITE, (x-6, y-2, 12, 12), 0.3, 2.8, 2)

    def draw_panel(self):
        text(self.screen, "Q-LEARNING ROBOT", PANEL_X, 58, 27, DARK, True)
        text(self.screen, "8x8 GridWorld  |  4 actions", PANEL_X, 100, 17, TEXT_MUTED)
        draw_button(self.screen, self.train_button, "TRAIN (T)", BLUE,
                    len(self.agent.history) < TRAIN_EPISODES)
        draw_button(self.screen, self.play_button, "PLAY (P)", GREEN,
                    bool(self.agent.history) and not self.training)
        draw_button(self.screen, self.reset_button, "RESET (R)", WALL)
        n = len(self.agent.history)
        pygame.draw.rect(self.screen, BORDER, (PANEL_X, 272, 305, 10), border_radius=5)
        if n:
            pygame.draw.rect(self.screen, GREEN, (PANEL_X, 272, max(1, round(305*n/TRAIN_EPISODES)), 10), border_radius=5)
        text(self.screen, f"Episodes: {n} / {TRAIN_EPISODES}", PANEL_X, 294, 18)
        text(self.screen, f"Epsilon: {self.agent.epsilon:.3f}", PANEL_X, 330, 18)
        if self.agent.history:
            recent = self.agent.history[-100:]
            rate = 100 * sum(t.success for t in recent) / len(recent)
            avg_reward = sum(t.reward for t in recent) / len(recent)
            text(self.screen, f"Recent success: {rate:.0f}%", PANEL_X, 366, 18)
            text(self.screen, f"Avg reward (100): {avg_reward:.1f}", PANEL_X, 402, 18)
        else:
            text(self.screen, "Recent success: --", PANEL_X, 366, 18)
            text(self.screen, "Avg reward (100): --", PANEL_X, 402, 18)
        text(self.screen, f"BFS minimum: {self.bfs_steps} steps", PANEL_X, 438, 18)
        if self.path_ok:
            text(self.screen, f"Learned route: {len(self.path)-1} steps", PANEL_X, 474, 18, GREEN, True)
        else:
            text(self.screen, "Learned route: --", PANEL_X, 474, 18)
        text(self.screen, "T: train / pause | P: play", PANEL_X, 518, 15, TEXT_MUTED)
        text(self.screen, "R: reset | ESC: exit", PANEL_X, 540, 15, TEXT_MUTED)

    def draw(self):
        self.screen.fill(BG)
        text(self.screen, "ROBOT PATH PLANNING", MAP_X, 28, 29, DARK, True)
        self.draw_grid()
        self.draw_panel()
        pygame.draw.line(self.screen, BORDER, (0, 578), (SCREEN_W, 578), 1)
        text(self.screen, self.status, 42, 589, 17, TEXT_MUTED)
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_t:
                        self.start_train()
                    elif event.key in (pygame.K_p, pygame.K_SPACE):
                        self.start_play()
                    elif event.key == pygame.K_r:
                        self.reset()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if self.train_button.collidepoint(event.pos):
                        self.start_train()
                    elif self.play_button.collidepoint(event.pos):
                        self.start_play()
                    elif self.reset_button.collidepoint(event.pos):
                        self.reset()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        pygame.quit()
        return 0


if __name__ == "__main__":
    sys.exit(RobotApp().run())
