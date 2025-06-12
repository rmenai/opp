import sys

import pygame

pygame.init()
screen = pygame.display.set_mode((400, 200))
pygame.display.set_caption("Hello World in Red")

WHITE = (255, 255, 255)
RED = (255, 0, 0)

font = pygame.font.SysFont(None, 48)
text = font.render("Hello, World!", True, RED, WHITE)
text_rect = text.get_rect(center=(200, 100))

screen.fill(WHITE)
screen.blit(text, text_rect)
pygame.display.flip()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
