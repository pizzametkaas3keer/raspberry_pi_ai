import pygame
import moderngl

print("Testing pygame + moderngl...")

try:
    pygame.init()
    print("Pygame init OK")
    
    screen = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
    print("OpenGL window OK")
    
    ctx = moderngl.create_context()
    print("moderngl context OK")
    
    ctx.clear(0.5, 0.5, 0.5, 1.0)
    print("OpenGL clear OK")
    
    pygame.display.flip()
    print("Display flip OK")
    
    print("SUCCESS: Alles werkt!")
    print("Druk op scherm om te sluiten...")
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
    
    pygame.quit()
    print("Done")
    
except Exception as e:
    print(f"ERROR: {e}")
    pygame.quit()
