import os
import imageio

class GifWriter:
    def __init__(self):
        pass
    
    def save_gif(self, frames, filename):
        file_path = os.path.join(os.getcwd(), 'gifs', filename)
        imageio.mimsave(file_path + '.gif', frames)
