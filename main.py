import pygame
import numpy as np
import random
from collections import defaultdict

# --- CONFIGURATION ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 600
GRID_SIZE = 20
CELL_W = SCREEN_WIDTH / GRID_SIZE
CELL_H = SCREEN_HEIGHT / GRID_SIZE
FPS = 30

# Load Assets (Ensure these filenames match your uploaded files)
try:
    IMG_BG = pygame.image.load("univ2_bg.png")
    IMG_AGENT = pygame.image.load("uni2_agent.png")
    IMG_ENEMY = pygame.image.load("uni2_enemy.png")
    # Resize assets to fit grid
    IMG_BG = pygame.transform.scale(IMG_BG, (SCREEN_WIDTH, SCREEN_HEIGHT))
    IMG_AGENT = pygame.transform.scale(IMG_AGENT, (int(CELL_W*1.2), int(CELL_H*1.2)))
    IMG_ENEMY = pygame.transform.scale(IMG_ENEMY, (int(CELL_W), int(CELL_H)))
except:
    print("Error: Image files not found. Ensure assets are in the same folder.")
    exit()

# --- AGENT LOGIC ---

def discretize_universe2(observation):
    ax, ay, ex, ey, gx, gy = observation
    dist_exit = np.sqrt((ex - ax)**2 + (ey - ay)**2)
    dist_guard = np.sqrt((gx - ax)**2 + (gy - ay)**2)
    dx = 1 if ex - ax > 0.5 else (-1 if ex - ax < -0.5 else 0)
    dy = 1 if ey - ay > 0.5 else (-1 if ey - ay < -0.5 else 0)
    danger = 0 if dist_guard > 4.5 else (1 if dist_guard >= 2.0 else 2)
    return (int(dist_exit / 1.5), int(dist_guard / 1.0), dx, dy, danger)

def create_universe2_agent():
    alpha, gamma = 0.08, 0.97
    epsilon = [1.0] # Mutable list to keep track across calls
    Q_table = defaultdict(lambda: np.zeros(4))

    def select_action(observation, training=True):
        state = discretize_universe2(observation)
        if training and random.random() < epsilon[0]:
            return random.randint(0, 3)
        return int(np.argmax(Q_table[state]))

    def update(obs, action, reward, next_obs, done):
        state = discretize_universe2(obs)
        next_state = discretize_universe2(next_obs)
        reward = np.clip(reward, -200, 600)
        best_next_q = 0 if done else np.max(Q_table[next_state])
        Q_table[state][action] += alpha * (reward + gamma * best_next_q - Q_table[state][action])
        if done: epsilon[0] = max(0.05, epsilon[0] * 0.993)

    return select_action, update, Q_table, epsilon

# --- GAME ENGINE ---

class HawkinsGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT + 60))
        pygame.display.set_caption("Universe 2: Escape Hawkins Lab")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 18, bold=True)
        
        self.agent_pos = np.array([2.0, 18.0])
        self.exit_pos = np.array([18.5, 2.5])
        self.guards = [np.array([10.0, 10.0]), np.array([5.0, 5.0]), np.array([15.0, 15.0])]
        self.steps = 0
        self.total_wins = 0

    def reset(self):
        self.agent_pos = np.array([2.0, 18.0])
        self.steps = 0
        return self._get_obs()

    def _get_obs(self):
        dists = [np.linalg.norm(self.agent_pos - g) for g in self.guards]
        nearest_g = self.guards[np.argmin(dists)]
        return np.array([*self.agent_pos, *self.exit_pos, *nearest_g])

    def step(self, action):
        # 0: Right, 1: Up, 2: Left, 3: Down
        moves = {0: [0.5, 0], 1: [0, -0.5], 2: [-0.5, 0], 3: [0, 0.5]}
        self.agent_pos += moves[action]
        self.agent_pos = np.clip(self.agent_pos, 0, 19)
        
        # Simple Guard Patrol
        for i in range(len(self.guards)):
            self.guards[i] += np.random.uniform(-0.1, 0.1, 2)
            self.guards[i] = np.clip(self.guards[i], 2, 17)

        obs = self._get_obs()
        dist_exit = np.linalg.norm(self.agent_pos - self.exit_pos)
        dist_guard = np.linalg.norm(self.agent_pos - obs[4:])

        reward = -0.8
        if dist_exit < 12.0: reward += 0.8
        if dist_exit < 1.5: reward += 600
        
        done = (dist_exit < 1.5) or (self.steps >= 300) or (dist_guard < 1.2)
        if dist_exit < 1.5: self.total_wins += 1
        self.steps += 1
        return obs, reward, done

    def draw(self, ep, eps, rew):
        self.screen.blit(IMG_BG, (0, 0))
        
        # Draw Guard Danger Zones (Visual Cue)
        for g in self.guards:
            s = pygame.Surface((CELL_W*3, CELL_H*3), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 0, 0, 40), (int(CELL_W*1.5), int(CELL_H*1.5)), int(CELL_W*1.5))
            self.screen.blit(s, (g[0]*CELL_W - CELL_W*1.5, g[1]*CELL_H - CELL_H*1.5))
            self.screen.blit(IMG_ENEMY, (g[0]*CELL_W - CELL_W/2, g[1]*CELL_H - CELL_H/2))

        # Draw Exit (The Gate)
        pygame.draw.circle(self.screen, (0, 255, 200), (int(self.exit_pos[0]*CELL_W), int(self.exit_pos[1]*CELL_H)), 25, 3)

        # Draw Agent
        self.screen.blit(IMG_AGENT, (self.agent_pos[0]*CELL_W - CELL_W/2, self.agent_pos[1]*CELL_H - CELL_H/2))

        # Dashboard
        pygame.draw.rect(self.screen, (10, 10, 15), (0, SCREEN_HEIGHT, SCREEN_WIDTH, 60))
        stats = f"EPISODE: {ep} | WINS: {self.total_wins} | EPSILON: {eps:.2f} | REWARD: {int(rew)}"
        text_surf = self.font.render(stats, True, (0, 255, 150))
        self.screen.blit(text_surf, (20, SCREEN_HEIGHT + 20))
        
        pygame.display.flip()

# --- RUN ---

if __name__ == "__main__":
    game = HawkinsGame()
    select_action, update, Q_table, eps_ptr = create_universe2_agent()
    
    episode = 0
    running = True
    
    while running:
        obs = game.reset()
        done = False
        ep_reward = 0
        
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False; done = True

            action = select_action(obs)
            next_obs, reward, done = game.step(action)
            update(obs, action, reward, next_obs, done)
            
            obs = next_obs
            ep_reward += reward
            
            game.draw(episode, eps_ptr[0], ep_reward)
            game.clock.tick(FPS)
            
        episode += 1

    pygame.quit()
