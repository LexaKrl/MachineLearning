import numpy as np
import vlc
from PIL import Image

def main():
    path = "C:\\Users\\Mi\\PycharmProjects\\MachineLearning\\1512667842_hl-2017-12-06-21-13-03-749.jpg"
    image = Image.open(path)
    print(image)
    print(np.array(image))

    music_path = "C:\\Users\\Mi\\PycharmProjects\\MachineLearning\\hos5.mp3"
    music = vlc.MediaPlayer(music_path)
    print(music)
    print(np.array(music))


if __name__ == "__main__":
    main()