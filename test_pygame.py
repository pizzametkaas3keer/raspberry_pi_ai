import pygame
import sys

print("Testing pygame only...")

try:
    pygame.init()
    print("Pygame init OK")
    
    screen = pygame.display.set_mode((800, 600))
    print("Window OK")
    
    screen.fill((50, 50, 100))
    print("Fill OK")
    
    pygame.display.flip()
    print("Flip OK")
    
    print("SUCCESS")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
    
    pygame.quit()
    
except Exception as e:
    print(f"ERROR: {e}")
    pygame.quit()
    sys.exit(1)
