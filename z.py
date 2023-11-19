import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from Methods.M1.generator import Generator
vis = Generator(chunk_size=10000)

vis.generate()