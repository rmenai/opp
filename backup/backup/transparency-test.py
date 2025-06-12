import sys

import pygame

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 300

# Colors
TRANSPARENT_BLACK = (0, 0, 0, 0)  # RGBA, A=0 for full transparency
OPAQUE_RED = (255, 0, 0, 255)  # RGBA, A=255 for full opacity

# Create the screen with per-pixel alpha
try:
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
    pygame.display.set_caption("Transparency Test")
except pygame.error as e:
    print(f"Error creating display: {e}")
    print("This might happen if your system doesn't support SRCALPHA or if SDL has issues.")
    sys.exit()

# Simple rectangle to draw
rect_color = OPAQUE_RED
rect_to_draw = pygame.Rect(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 4, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

running = True
clock = pygame.time.Clock()

print("Transparency Test Running. Press ESC or close window to quit.")
print("If the window background is transparent, you should see your desktop behind it.")
print("A red rectangle should be visible in the center.")

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # Fill the screen with a transparent color
    # This is crucial for the background to be transparent
    screen.fill(TRANSPARENT_BLACK)

    # Draw an opaque rectangle
    pygame.draw.rect(screen, rect_color, rect_to_draw)

    # Update the display
    pygame.display.flip()

    clock.tick(30)

pygame.quit()
sys.exit()
