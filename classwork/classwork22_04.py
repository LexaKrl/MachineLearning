from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
import pygame

def main():
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    screen.fill("white")
    pygame.display.update()

    play = True
    r = 5
    X, y = [], []
    while(play):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                X.append(event.pos)
                if event.button == 1:
                    pygame.draw.circle(screen, "red", event.pos, r)
                    y.append("red")
                if event.button == 3:
                    pygame.draw.circle(screen, "blue", event.pos, r)
                    y.append("blue")
                pygame.display.update()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if len(X) > 0 and len(y) > 0:
                        lr = LogisticRegression()
                        lr.fit(X, y)

                        for i in range(0, 600, 5):
                            for j in range(0, 400, 5):
                                pygame.draw.circle(screen, lr.predict([[i, j]])[0], (i, j), 1)
                                pygame.display.update()





if __name__ == "__main__":
    main()