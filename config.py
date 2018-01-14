import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_ROOT = os.path.join(ROOT_DIR, 'data')
EMBD_FILE = os.path.join(ROOT_DIR, 'data/saved_embd.pt')


if __name__ == '__main__':
    print(EMBD_FILE)

# /home/easonnie/projects/publiced_code/ResEncoder/saved_model/12-04-23:22:31_[600,600,600]-3stack-bilstm-maxout-residual-1-relu-seed(12)-dr(0.1)-mlpd(800)